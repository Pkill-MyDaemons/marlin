pub mod background;
pub mod budget;
pub mod context;
pub mod loop_guard;
pub mod subagent;

use base64::{engine::general_purpose::STANDARD as B64, Engine as _};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::Result;
use marlin_core::skill::{Skill, SkillDef};
use marlin_core::tasks::{TaskStatus, TaskStep};
use tokio::sync::mpsc;

use config::{AstMode, Config, ModelTier, SandboxMode};
use context::{estimate_tokens, maybe_prune_history};
use history::{from_session_message, to_session_message, InputHistory, Session};
use index::Index;
use loop_guard::LoopGuard;
use marlin_checkpoint as checkpoint;
use marlin_commands as commands;
use marlin_config as config;
use marlin_history as history;
use marlin_index as index;
use marlin_mcp as mcp;
use marlin_preflight as preflight;
use marlin_providers::{
    registry::Registry, Message, Provider, RateLimitState, StreamChunk, StreamRequest, ToolCall,
    ToolCallMsg,
};
use marlin_skills as skills;
use marlin_snapshots as snapshots;
use marlin_styles as styles;
use marlin_tools::{all_tools, executor, policy};

pub use marlin_core::ui::{Action, ConfigState, HistoryEntry, StatusInfo, UiUpdate};

// ── Channel types ────────────────────────────────────────────────────────────

/// Read the current git branch from `<dir>/.git/HEAD`. Returns None when the
/// directory isn't a git repo or the file is unreadable.
pub fn detect_git_branch(dir: &str) -> Option<String> {
    let head = std::path::Path::new(dir).join(".git").join("HEAD");
    let content = std::fs::read_to_string(&head).ok()?;
    let branch = content.trim();
    if let Some(name) = branch.strip_prefix("ref: refs/heads/") {
        Some(name.to_string())
    } else {
        // Detached HEAD — the content is a commit hash; show a short form.
        Some(branch.chars().take(7).collect())
    }
}

/// Map a file extension to its MIME type if it's a supported image. Returns
/// None for anything that isn't an image (so /attach can fall through to the
/// text-attachment path).
fn image_mime(path: &str) -> Option<String> {
    let ext = Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "png" => Some("image/png".into()),
        "jpg" | "jpeg" => Some("image/jpeg".into()),
        "gif" => Some("image/gif".into()),
        "webp" => Some("image/webp".into()),
        "bmp" => Some("image/bmp".into()),
        "svg" => Some("image/svg+xml".into()),
        _ => None,
    }
}

// ── Engine ───────────────────────────────────────────────────────────────────

pub struct Engine {
    cfg: Config,
    registry: Registry,
    marlin_dir: PathBuf,
    work_dir: String,

    pub history: Vec<Message>,
    code_index: Option<Index>,
    /// mtime/size snapshot used to detect external file changes for the
    /// background index refresh.
    index_refresh: index::RefreshState,
    session: Option<Session>,
    input_history: InputHistory,

    active_goal: String,
    tool_iterations: usize,
    stall_nudges: usize,
    attachments: Vec<(String, String)>, // (filename, content)
    /// Image attachments for the next user message: (path, mime_type, base64).
    /// Injected as multimodal content blocks by the providers.
    image_attachments: Vec<(String, String, String)>,
    allowed_commands: Vec<String>,
    /// Retry counter for mid-stream transient network disconnects within the
    /// current turn. Reset to 0 at the start of each new message/slash command
    /// (see SendMessage / SlashCommand) so a fresh request gets a clean slate.
    mid_stream_retries: usize,

    rate_limit_state: Option<RateLimitState>,
    loop_guard: LoopGuard,
    cancel_flag: Arc<std::sync::atomic::AtomicBool>,

    /// Live task list for the sidebar
    task_steps: Vec<TaskStep>,
    /// Upfront plan generated before the tool loop starts (see `maybe_generate_plan`).
    /// Separate from `task_steps`: this is a coarse, ordered checklist shown above
    /// the granular tool-call log, not a replacement for it.
    plan_steps: Vec<TaskStep>,
    /// Index of the next `plan_steps` entry to resolve as tool-call batches complete.
    plan_cursor: usize,
    /// Approximate token budget ceiling (from config or 100k default)
    token_budget: usize,
    /// AST-driven context mode
    ast_mode: AstMode,

    /// Loaded skill definitions
    skills: Vec<Skill>,
    /// User-defined slash commands from ~/.marlin/commands/
    user_commands: Vec<commands::UserCommand>,
    /// User-defined LLM tools from ~/.marlin/tools/
    external_tools: Vec<marlin_tools::external::ExternalTool>,
    /// MCP server configs from ~/.marlin/mcp/ (loaded at construction; servers
    /// are actually spawned in `run()`, since spawning is async).
    mcp_server_configs: Vec<mcp::McpServerConfig>,
    /// Live MCP connections, keyed by server name.
    mcp_clients: HashMap<String, Arc<mcp::client::McpClient>>,
    /// Tools discovered from `mcp_clients` via `tools/list`, as (server_name, tool)
    /// pairs — mirrors `external_tools`' role for `tools::all_tools`.
    mcp_tools: Vec<(String, mcp::client::McpTool)>,
    /// Provider/model selected for the current agentic request (may be tier-routed)
    req_provider: String,
    req_model: String,
    /// Backup provider/model to use if req_provider is rate-limited
    req_backup_provider: String,
    req_backup_model: String,

    /// Token count at last LLM compaction — prevents immediate re-triggering.
    compact_guard_tokens: usize,

    /// Registry of long-running background processes (see `engine::background`).
    /// The model starts/polls/kills them via the bg_start/bg_log/bg_status/bg_kill
    /// tools; they survive across turns.
    background: background::BackgroundRegistry,

    /// Diagnostics collected at construction time (skill validation issues,
    /// missing binaries, unparsable config files, stale index) — emitted once
    /// `run()` has a UI channel to surface them on, since eprintln! during
    /// startup is invisible once the TUI takes over the terminal.
    startup_diagnostics: Vec<String>,
}

impl Engine {
    pub fn new(cfg: Config) -> Result<Self> {
        let marlin_dir = marlin_config::marlin_dir()?;
        // Install the default provider file before building the registry so it's
        // picked up on this same startup, not just the next one.
        marlin_providers::user_providers::install_defaults(&marlin_dir);
        let registry = Registry::new(&cfg, Some(&marlin_dir));
        let work_dir = cfg.work_dir.clone();
        let allowed = cfg.allowed_commands.clone();

        let input_history = InputHistory::load(&marlin_dir);
        let code_index = index::load(&marlin_dir, &work_dir).ok();
        // Seed the mtime/size baseline so the first background refresh only
        // re-indexes files that actually change after startup — not everything.
        let mut index_refresh = index::RefreshState::default();
        if code_index.is_some() {
            let (_, next) = index::diff_against(&index_refresh, &work_dir);
            index_refresh = next;
        }

        let project_name = Path::new(&work_dir)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string();
        let session = Some(Session::new(&project_name, &work_dir));

        let ast_mode = cfg.ast_mode.clone();
        let req_provider = cfg.active_provider.clone();
        let req_model = cfg.active_model.clone();
        let token_budget = cfg.token_budget;

        // Install built-in skills/commands/tools if not present, then load all.
        skills::install_defaults(&marlin_dir);
        let (loaded_skills, mut startup_diagnostics) = skills::load_all(&marlin_dir);
        commands::install_defaults(&marlin_dir);
        let loaded_commands = commands::load_all(&marlin_dir);
        marlin_tools::external::install_defaults(&marlin_dir);
        let loaded_external_tools = marlin_tools::external::load_all(&marlin_dir);
        // No install_defaults here — unlike skills/commands/tools there's no safe
        // offline-runnable default MCP server to ship. Servers themselves are
        // spawned later in `run()` (async); this just loads the configs.
        let loaded_mcp_configs = mcp::load_all(&marlin_dir);
        marlin_config::install_default_themes(&marlin_dir);

        startup_diagnostics.extend(preflight::startup(
            &cfg,
            &marlin_dir,
            &work_dir,
            code_index.as_ref(),
        ));

        let mut engine = Self {
            cfg,
            registry,
            marlin_dir,
            work_dir,
            history: vec![],
            code_index,
            index_refresh,
            session,
            input_history,
            active_goal: String::new(),
            tool_iterations: 0,
            stall_nudges: 0,
            mid_stream_retries: 0,
            attachments: vec![],
            image_attachments: vec![],
            allowed_commands: allowed,
            rate_limit_state: None,
            loop_guard: LoopGuard::new(),
            cancel_flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            task_steps: vec![],
            plan_steps: vec![],
            plan_cursor: 0,
            token_budget,
            ast_mode,
            skills: loaded_skills,
            user_commands: loaded_commands,
            external_tools: loaded_external_tools,
            mcp_server_configs: loaded_mcp_configs,
            mcp_clients: HashMap::new(),
            mcp_tools: Vec::new(),
            req_provider,
            req_model,
            req_backup_provider: String::new(),
            req_backup_model: String::new(),
            compact_guard_tokens: 0,
            background: background::BackgroundRegistry::new(),
            startup_diagnostics,
        };
        engine.apply_project_config();
        Ok(engine)
    }

    /// Preflight startup diagnostics (missing binaries, unparsable config files,
    /// skill validation issues, stale index, ...), computed once in `new()`.
    /// Exposed so the CLI entry point can print them to the real terminal before
    /// the TUI takes over the alternate screen — `run()` also surfaces them as a
    /// system message once the UI channel exists, so they're not lost either way.
    pub fn startup_diagnostics(&self) -> &[String] {
        &self.startup_diagnostics
    }

    /// Load and apply `.marlonrc.toml` from the current work directory.
    /// Called at construction and after every `/cd`. Project config overrides
    /// are session-only — they're never persisted to `config.json`.
    pub fn apply_project_config(&mut self) {
        if let Some(pc) = marlin_config::load_project_config(&self.work_dir) {
            if !pc.system_prompt.is_empty() {
                self.cfg.system_prompt = pc.system_prompt;
            }
            if !pc.allowed_commands.is_empty() {
                self.allowed_commands = pc.allowed_commands.clone();
                self.cfg.allowed_commands = pc.allowed_commands;
            }
            if let Some(vc) = pc.verify_command {
                self.cfg.verify_command = Some(vc);
            }
            if let Some(sm) = pc.sandbox_mode {
                match sm.as_str() {
                    "off" => self.cfg.sandbox_mode = SandboxMode::Off,
                    "permissive" | "on" => self.cfg.sandbox_mode = SandboxMode::Permissive,
                    "mxc" => self.cfg.sandbox_mode = SandboxMode::Mxc,
                    "docker" => self.cfg.sandbox_mode = SandboxMode::Docker,
                    _ => {}
                }
            }
        }
    }

    /// Build the current `StatusInfo` (provider/model/git branch/work dir and the
    /// per-directory status bar color from config, if any).
    fn status_info(&self) -> StatusInfo {
        let git_branch = detect_git_branch(&self.work_dir);
        let status_color = self.cfg.status_colors.get(&self.work_dir).copied();
        StatusInfo {
            provider: self.cfg.active_provider.clone(),
            model: self.cfg.active_model.clone(),
            git_branch,
            work_dir: self.work_dir.clone(),
            status_color,
            bg_count: self.background.running_count(),
        }
    }

    pub async fn run(
        &mut self,
        mut action_rx: mpsc::Receiver<Action>,
        ui_tx: mpsc::Sender<UiUpdate>,
    ) {
        // Send initial status
        let _ = ui_tx.send(UiUpdate::StatusUpdate(self.status_info())).await;

        // If `--resume-last` pre-seeded history before `run()`, repopulate the
        // TUI's chat display so the resumed conversation is visible immediately
        // (not just held in the model's context window).
        if !self.history.is_empty() {
            let _ = ui_tx
                .send(UiUpdate::HistoryLoaded(self.history_entries()))
                .await;
        }

        let _ = ui_tx
            .send(UiUpdate::SystemMsg(
                "marlin ready  /help for commands".into(),
            ))
            .await;
        if !self.startup_diagnostics.is_empty() {
            let body = self.startup_diagnostics.join("\n  ");
            let _ = ui_tx
                .send(UiUpdate::SystemMsg(format!(
                    "preflight startup ({} note(s)) — see /preflight:\n  {body}",
                    self.startup_diagnostics.len()
                )))
                .await;
        }
        if self.ast_mode != AstMode::Off {
            let _ = ui_tx.send(UiUpdate::AstMode(self.ast_mode.clone())).await;
        }
        if let Some(idx) = &self.code_index {
            let _ = ui_tx
                .send(UiUpdate::SystemMsg(format!(
                    "index: {} files, {} terms",
                    idx.file_count, idx.term_count
                )))
                .await;
        }

        // Send skills and user commands to TUI for suggestion panel.
        let skill_defs: Vec<SkillDef> = self.skills.iter().map(SkillDef::from).collect();
        let _ = ui_tx.send(UiUpdate::SkillsLoaded(skill_defs)).await;
        let cmd_defs: Vec<commands::UserCommandDef> = self
            .user_commands
            .iter()
            .map(commands::UserCommandDef::from)
            .collect();
        let _ = ui_tx.send(UiUpdate::UserCommandsLoaded(cmd_defs)).await;

        // Spawn nightly skill-suggestion daemon.
        self.maybe_spawn_daemon(ui_tx.clone());

        // Connect configured MCP servers (~/.marlin/mcp/*.json). Async, so it
        // couldn't happen in `new()`; best-effort per server — one bad/missing
        // server reports a message and doesn't block the others or startup.
        self.connect_mcp_servers(&ui_tx).await;

        // Periodic background index refresh: every 30s, re-index any files that
        // changed on disk (added externally, edited outside the agent, deleted).
        let mut index_tick = tokio::time::Instant::now();

        while let Some(action) = action_rx.recv().await {
            // Keep the index warm in the background between actions.
            if index_tick.elapsed() >= Duration::from_secs(30) {
                index_tick = tokio::time::Instant::now();
                let n = self.maybe_refresh_index();
                if n > 0 {
                    let _ = ui_tx
                        .send(UiUpdate::SystemMsg(format!(
                            "index: refreshed {n} changed file(s)"
                        )))
                        .await;
                }
            }
            match action {
                Action::Quit => break,

                Action::CancelStream => {
                    self.cancel_flag
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    self.active_goal.clear();
                    self.tool_iterations = 0;
                    self.stall_nudges = 0;
                    // Otherwise any plan steps still Pending when the user cancelled
                    // linger in the sidebar indefinitely, implying the cancelled task
                    // is still queued/active.
                    self.plan_steps.clear();
                    self.plan_cursor = 0;
                    let _ = ui_tx.send(UiUpdate::PlanUpdate(vec![])).await;
                    let _ = ui_tx.send(UiUpdate::SystemMsg("Cancelled.".into())).await;
                }

                // Approval actions received outside an agentic loop are no-ops
                Action::Approve | Action::Deny => {}
                // UserAnswer is consumed inside await_ask_user while the agentic
                // loop is blocked; if it arrives here it's a stray no-op.
                Action::UserAnswer(_) => {}
                // Diff preview actions received outside an agentic loop are no-ops
                Action::AcceptDiff { .. } | Action::RejectDiff { .. } => {}

                Action::ConfigSet { key, value } => {
                    self.apply_config_set(&key, &value, &ui_tx).await;
                }

                Action::SaveEditorFile { path, content } => {
                    self.save_editor_file(path, content, &ui_tx, &mut action_rx)
                        .await;
                }

                Action::SendMessage(text) => {
                    self.input_history.add(&text);

                    // Emit skill matches so TUI can show relevant skills.
                    let skill_defs: Vec<SkillDef> =
                        self.skills.iter().map(SkillDef::from).collect();
                    let matches: Vec<(String, String)> =
                        skills::suggest::match_skills(&text, &skill_defs)
                            .into_iter()
                            .map(|m| (m.name, m.description))
                            .collect();
                    if !matches.is_empty() {
                        let _ = ui_tx.send(UiUpdate::SkillMatches(matches)).await;
                    }

                    let msg = self.take_attachments(&text);
                    self.history.push(msg);
                    self.active_goal = text.clone();
                    self.tool_iterations = 0;
                    self.stall_nudges = 0;
                    self.mid_stream_retries = 0;
                    self.task_steps.clear();
                    self.plan_steps.clear();
                    self.plan_cursor = 0;
                    let _ = ui_tx.send(UiUpdate::PlanUpdate(vec![])).await;
                    self.loop_guard.reset();
                    self.cancel_flag
                        .store(false, std::sync::atomic::Ordering::SeqCst);

                    // Select model tier based on difficulty score (if tiers enabled).
                    self.rate_and_route(&text, &ui_tx).await;
                    self.maybe_generate_plan(&text, &ui_tx).await;

                    // Create a git checkpoint before the turn so /undo can roll
                    // it back (opt-in via /checkpoints on).
                    self.maybe_checkpoint(&ui_tx).await;

                    // Broadcast token count immediately so the sidebar isn't stale while waiting.
                    // Prefer exact count from provider API; fall back to heuristic.
                    let system_prompt = self.effective_system_prompt();
                    let turn_tools = all_tools(
                        &self.ast_mode,
                        &self.skill_tool_list(&text),
                        &self.external_tools,
                        self.cfg.skill_subagents,
                        &self.mcp_tools,
                    );
                    let tok = if let Ok(p) = self.registry.get(&self.req_provider) {
                        let req_for_count = StreamRequest {
                            model: self.req_model.clone(),
                            messages: self.history.clone(),
                            system_prompt: system_prompt.clone(),
                            max_tokens: 1,
                            tools: turn_tools.clone(),
                            thinking: false,
                        };
                        p.count_tokens(&req_for_count)
                            .await
                            .unwrap_or_else(|| estimate_tokens(&self.history, &system_prompt))
                    } else {
                        estimate_tokens(&self.history, &system_prompt)
                    };
                    let injection_report = budget::compute(&system_prompt, &turn_tools);
                    let _ = ui_tx
                        .send(UiUpdate::PromptBudget(
                            injection_report
                                .over_budget()
                                .then_some(injection_report.total),
                        ))
                        .await;
                    let _ = ui_tx
                        .send(UiUpdate::TokenUsage {
                            used: tok,
                            budget: self.token_budget,
                        })
                        .await;
                    self.agentic_loop(&ui_tx, &mut action_rx).await;
                }

                Action::SlashCommand(cmd) => {
                    self.input_history.add(&cmd);
                    if let Some(prompt) = self
                        .handle_slash_command(&cmd, &ui_tx, &mut action_rx)
                        .await
                    {
                        // Prompt-type user command: inject expanded template and run agentic loop.
                        let msg = self.take_attachments(&prompt);
                        self.history.push(msg);
                        self.active_goal = prompt.clone();
                        self.tool_iterations = 0;
                        self.stall_nudges = 0;
                        self.mid_stream_retries = 0;
                        self.task_steps.clear();
                        self.plan_steps.clear();
                        self.plan_cursor = 0;
                        let _ = ui_tx.send(UiUpdate::PlanUpdate(vec![])).await;
                        self.loop_guard.reset();
                        self.cancel_flag
                            .store(false, std::sync::atomic::Ordering::SeqCst);
                        self.rate_and_route(&prompt, &ui_tx).await;
                        self.maybe_generate_plan(&prompt, &ui_tx).await;
                        self.maybe_checkpoint(&ui_tx).await;
                        self.agentic_loop(&ui_tx, &mut action_rx).await;
                    }
                }

                // Steering input while the model is idle. If it's a slash command
                // it's executed instantly; otherwise it's shown as a text field in
                // the model's output area (no agentic loop is started).
                Action::Steer(text) => {
                    self.handle_steer(&text, &ui_tx, &mut action_rx).await;
                }
            }
        }
    }

    async fn agentic_loop(
        &mut self,
        ui_tx: &mpsc::Sender<UiUpdate>,
        action_rx: &mut mpsc::Receiver<Action>,
    ) {
        let safety_cap = self.cfg.tool_call_limit;

        loop {
            if self.cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                break;
            }
            // A cancel/steer may have arrived while the previous iteration was
            // busy (tool execution, compaction, rate-limit sleep) — drain it
            // before starting a new stream request.
            if self.drain_pending_actions(ui_tx, action_rx).await {
                break;
            }

            // Proactive rate-limit check
            if let Some(rl) = &self.rate_limit_state {
                let est = estimate_tokens(&self.history, &self.effective_system_prompt());
                let mut wait_secs = 0u32;

                if rl.remaining_tokens >= 0 && est as i64 > rl.remaining_tokens {
                    if let Some(reset) = rl.reset_tokens_at {
                        if let Ok(d) = reset.duration_since(SystemTime::now()) {
                            wait_secs = d.as_secs() as u32 + 1;
                        }
                    }
                }
                if rl.remaining_requests == 0 {
                    if let Some(reset) = rl.reset_requests_at {
                        if let Ok(d) = reset.duration_since(SystemTime::now()) {
                            let s = d.as_secs() as u32 + 1;
                            if s > wait_secs {
                                wait_secs = s;
                            }
                        }
                    }
                }

                if wait_secs > 0 {
                    self.rate_limit_state = None;
                    let _ = ui_tx.send(UiUpdate::RateLimited { secs: wait_secs }).await;
                    tokio::time::sleep(Duration::from_secs(wait_secs as u64)).await;
                    let _ = ui_tx
                        .send(UiUpdate::SystemMsg(
                            "Rate limit cleared — resuming...".into(),
                        ))
                        .await;
                }
            }

            // LLM-based compaction first, then mechanical truncation fallback
            self.maybe_compact_history(ui_tx).await;

            let (compressed, dropped) = maybe_prune_history(&mut self.history, self.token_budget);
            if compressed > 0 || dropped > 0 {
                let _ = ui_tx.send(UiUpdate::SystemMsg(format!(
                    "Context managed: compressed {compressed} messages, dropped {dropped} oldest turns."
                ))).await;
            }

            // Broadcast token usage to sidebar
            let tok_used = estimate_tokens(&self.history, &self.effective_system_prompt());
            let _ = ui_tx
                .send(UiUpdate::TokenUsage {
                    used: tok_used,
                    budget: self.token_budget,
                })
                .await;

            // A cancel may have arrived during compaction / rate-limit sleep
            // above — catch it before spending a new stream request.
            if self.drain_pending_actions(ui_tx, action_rx).await {
                break;
            }

            let provider = match self.registry.get(&self.req_provider) {
                Ok(p) => p,
                Err(e) => {
                    let _ = ui_tx.send(UiUpdate::ErrorMsg(e.to_string())).await;
                    break;
                }
            };

            let req = StreamRequest {
                model: self.req_model.clone(),
                messages: self.history.clone(),
                system_prompt: self.effective_system_prompt(),
                max_tokens: self.cfg.max_tokens,
                tools: all_tools(
                    &self.ast_mode,
                    &self.skill_tool_list(&self.active_goal),
                    &self.external_tools,
                    self.cfg.skill_subagents,
                    &self.mcp_tools,
                ),
                thinking: self.cfg.thinking,
            };

            let mut stream = match self.stream_with_retry(provider, req, ui_tx).await {
                Some(s) => s,
                None => break,
            };

            let mut text_buf = String::new();
            let mut done_chunk = None;

            // Poll every 50 ms so Ctrl+C is felt within one frame rather than
            // waiting for the next network chunk (which can take seconds). Also
            // polls for steering input (Action::Steer) so the user can nudge the
            // model mid-stream without interrupting it.
            'recv: loop {
                let chunk = tokio::select! {
                    maybe = stream.recv() => match maybe {
                        Some(c) => c,
                        None => break 'recv,
                    },
                    maybe_action = action_rx.recv() => {
                        match maybe_action {
                            Some(Action::Steer(text)) => {
                                self.handle_steer(&text, ui_tx, action_rx).await;
                                continue 'recv;
                            }
                            Some(Action::CancelStream) => {
                                self.cancel_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                                self.active_goal.clear();
                                self.tool_iterations = 0;
                                self.stall_nudges = 0;
                                self.plan_steps.clear();
                                self.plan_cursor = 0;
                                let _ = ui_tx.send(UiUpdate::PlanUpdate(vec![])).await;
                                if !text_buf.is_empty() {
                                    let _ = ui_tx.send(UiUpdate::StreamChunk("\n\n*[cancelled]*".into())).await;
                                }
                                return;
                            }
                            // Channel closed — nothing more to do.
                            None => break 'recv,
                            _ => { continue 'recv; }
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_millis(50)) => {
                        if self.cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                            if !text_buf.is_empty() {
                                let _ = ui_tx.send(UiUpdate::StreamChunk("\n\n*[cancelled]*".into())).await;
                            }
                            return;
                        }
                        continue 'recv;
                    }
                };

                if chunk.retry_after > 0 {
                    // Switch to backup provider/model if configured.
                    if !self.req_backup_provider.is_empty() {
                        let bp = std::mem::take(&mut self.req_backup_provider);
                        let bm = std::mem::take(&mut self.req_backup_model);
                        let _ = ui_tx
                            .send(UiUpdate::SystemMsg(format!(
                                "Rate limited — switching to backup: {bp} / {bm}"
                            )))
                            .await;
                        self.req_provider = bp;
                        self.req_model = bm;
                    } else {
                        let _ = ui_tx
                            .send(UiUpdate::RateLimited {
                                secs: chunk.retry_after,
                            })
                            .await;
                        tokio::time::sleep(Duration::from_secs(chunk.retry_after as u64)).await;
                        let _ = ui_tx
                            .send(UiUpdate::SystemMsg(
                                "Rate limit cleared — resuming...".into(),
                            ))
                            .await;
                    }
                    break 'recv;
                }

                if let Some(e) = chunk.error {
                    // Mid-stream transient network disconnect (stream dropped
                    // after the request succeeded). Nothing has been committed
                    // to history yet — text_buf is local — so a full re-request
                    // of this same turn is safe and idempotent. Retry a couple
                    // times, then give up with a clear error.
                    if is_transient_network_error(&e) && self.mid_stream_retries < 2 {
                        self.mid_stream_retries += 1;
                        let _ = ui_tx
                            .send(UiUpdate::SystemMsg(format!(
                                "Stream interrupted — retrying ({}/2): {e}",
                                self.mid_stream_retries
                            )))
                            .await;
                        tokio::time::sleep(Duration::from_millis(600)).await;
                        break 'recv; // outer loop re-requests the stream
                    }
                    let _ = ui_tx.send(UiUpdate::ErrorMsg(e.to_string())).await;
                    return;
                }

                if !chunk.content.is_empty() {
                    text_buf.push_str(&chunk.content);
                    let _ = ui_tx.send(UiUpdate::StreamChunk(chunk.content)).await;
                }

                if chunk.done {
                    if let Some(rl) = chunk.rate_limit {
                        self.rate_limit_state = Some(rl);
                    }
                    done_chunk = Some(chunk.tool_calls);
                    break 'recv;
                }
            }

            let Some(tool_calls) = done_chunk else {
                continue;
            };

            if !tool_calls.is_empty() {
                if self.tool_iterations >= safety_cap {
                    let _ = ui_tx.send(UiUpdate::ErrorMsg(format!(
                        "Safety cap reached ({safety_cap} tool calls). Send a new message to continue."
                    ))).await;
                    self.active_goal.clear();
                    return;
                }

                let text = text_buf.trim().to_string();
                self.history.push(Message {
                    role: "assistant".into(),
                    content: text.clone(),
                    tool_calls: tool_calls
                        .iter()
                        .map(|tc| ToolCallMsg {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            input: tc.input.clone(),
                        })
                        .collect(),
                    tool_use_id: String::new(),
                    tool_call_id: String::new(),
                    images: vec![],
                    is_error: false,
                });

                // Notify TUI of each tool call and add to task list. Calls in a run of
                // 2+ consecutive parallel-safe tools share a group id (see
                // `parallel_group_ids`) so the sidebar can show they ran together.
                let groups = parallel_group_ids(&tool_calls);
                for (tc, group) in tool_calls.iter().zip(groups.iter()) {
                    let _ = ui_tx
                        .send(UiUpdate::ToolCall {
                            name: tc.name.clone(),
                            input: tc.input.clone(),
                        })
                        .await;
                    let desc = tool_short_desc(&tc.name, &tc.input);
                    self.task_steps
                        .push(TaskStep::tool_pending(&tc.name, desc, *group));
                }
                let _ = ui_tx
                    .send(UiUpdate::TaskUpdate(self.task_steps.clone()))
                    .await;

                // Check for destructive commands and await user approval
                let denied = self
                    .run_approval_checks(&tool_calls, ui_tx, action_rx)
                    .await;
                if self.cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }

                // Execute tools (run in blocking thread)
                let results = self
                    .execute_tools(&tool_calls, &denied, ui_tx, action_rx)
                    .await;

                // A cancel may have arrived while the tools were running — stop
                // the loop rather than feeding the results back to the model.
                if self.cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                    return;
                }

                // Track which task step index corresponds to this batch
                let batch_task_start = self.task_steps.len().saturating_sub(tool_calls.len());

                for (i, (tc, (res, real_group))) in
                    tool_calls.iter().zip(results.iter()).enumerate()
                {
                    let _ = ui_tx
                        .send(UiUpdate::ToolResult {
                            name: tc.name.clone(),
                            output: res.output.clone(),
                            is_error: res.is_error,
                        })
                        .await;

                    // When a tool returns an error, offer a quick inline explanation
                    if res.is_error {
                        let hint = error_hint(&tc.name, &res.output);
                        if !hint.is_empty() {
                            let _ = ui_tx.send(UiUpdate::SystemMsg(format!("💡 {hint}"))).await;
                        }
                    }

                    // Update task step status, and correct the speculative group id
                    // (assigned before approval/denial was known — see
                    // parallel_group_ids) with what actually ran concurrently.
                    let step_idx = batch_task_start + i;
                    if step_idx < self.task_steps.len() {
                        self.task_steps[step_idx].status = if res.is_error {
                            TaskStatus::Failed
                        } else {
                            TaskStatus::Completed
                        };
                        self.task_steps[step_idx].parallel_group = *real_group;
                    }

                    // File-hash-aware loop guard for edits
                    let intercept = if tc.name == "edit_file"
                        || tc.name == "write_file"
                        || tc.name == "notebook_edit"
                        || tc.name == "multi_edit"
                    {
                        if let Some(path) = extract_file_path(&tc.input, &self.work_dir) {
                            let content = std::fs::read(&path).unwrap_or_default();
                            self.loop_guard
                                .check_file_edit(&path, &content, res.is_error)
                        } else {
                            self.loop_guard.check(&tc.name, res.is_error)
                        }
                    } else {
                        self.loop_guard.check(&tc.name, res.is_error)
                    };

                    if let Some(msg) = intercept {
                        let _ = ui_tx.send(UiUpdate::SystemMsg(msg.clone())).await;
                        self.history.push(Message {
                            role: "tool".into(),
                            content: msg,
                            tool_calls: vec![],
                            tool_use_id: tc.id.clone(),
                            tool_call_id: tc.id.clone(),
                            images: vec![],
                            is_error: true,
                        });
                    } else {
                        self.history.push(Message {
                            role: "tool".into(),
                            content: res.output.clone(),
                            tool_calls: vec![],
                            tool_use_id: tc.id.clone(),
                            tool_call_id: tc.id.clone(),
                            images: vec![],
                            is_error: res.is_error,
                        });
                    }

                    // Keep index fresh after writes
                    if (tc.name == "write_file"
                        || tc.name == "edit_file"
                        || tc.name == "notebook_edit"
                        || tc.name == "multi_edit")
                        && !res.is_error
                    {
                        if let Some(path) = extract_file_path(&tc.input, &self.work_dir) {
                            if let Some(idx) = &mut self.code_index {
                                index::update_file(idx, &path);
                            }
                        }
                    }
                }

                let _ = ui_tx
                    .send(UiUpdate::TaskUpdate(self.task_steps.clone()))
                    .await;

                // Advance the upfront plan by one step per tool-call batch — a rough
                // approximation of "one plan step per loop iteration" rather than
                // trying to match individual tool calls to plan text.
                if self.plan_cursor < self.plan_steps.len() {
                    let batch_failed = results.iter().any(|(r, _)| r.is_error);
                    self.plan_steps[self.plan_cursor].status = if batch_failed {
                        TaskStatus::Failed
                    } else {
                        TaskStatus::Completed
                    };
                    self.plan_cursor += 1;
                    let _ = ui_tx
                        .send(UiUpdate::PlanUpdate(self.plan_steps.clone()))
                        .await;
                }

                // Write-Test-Fix: run verify_command after any file edit
                let had_file_edit = tool_calls.iter().zip(results.iter()).any(|(tc, (r, _))| {
                    (tc.name == "edit_file"
                        || tc.name == "write_file"
                        || tc.name == "notebook_edit"
                        || tc.name == "multi_edit")
                        && !r.is_error
                });
                if had_file_edit {
                    if let Some(verify_result) = self.run_verify_command(ui_tx).await {
                        self.history.push(verify_result);
                    }
                }

                self.tool_iterations += 1;

                if let Some(done_call) = tool_calls.iter().find(|tc| tc.name == "mark_complete") {
                    let summary = extract_summary_field(&done_call.input);
                    if !summary.is_empty() {
                        let _ = ui_tx.send(UiUpdate::Summary(summary)).await;
                    }
                    // Send desktop notification if the turn took many tool calls
                    if self.tool_iterations >= 5 {
                        self.send_notification("Marlin task complete", &self.active_goal);
                    }
                    self.finish_turn(ui_tx).await;
                    break;
                }
                // Continue loop
            } else {
                // Goal complete — unless the model stopped mid-promise (see
                // `looks_like_unfinished_stall`), in which case nudge it to
                // actually follow through instead of reporting done.
                const MAX_STALL_NUDGES: usize = 2;
                let text = text_buf.trim().to_string();

                if !text.is_empty()
                    && looks_like_unfinished_stall(&text)
                    && self.stall_nudges < MAX_STALL_NUDGES
                {
                    self.stall_nudges += 1;
                    self.history.push(Message {
                        role: "assistant".into(),
                        content: text,
                        tool_calls: vec![],
                        tool_use_id: String::new(),
                        tool_call_id: String::new(),
                        images: vec![],
                        is_error: false,
                    });
                    self.history.push(Message::new_user(
                        "You said you would do that but made no tool call. Either make \
                            the tool call now to actually do it, or say plainly that you're done \
                            or blocked.",
                    ));
                    continue;
                }

                if !text.is_empty() {
                    self.history.push(Message {
                        role: "assistant".into(),
                        content: text,
                        tool_calls: vec![],
                        tool_use_id: String::new(),
                        tool_call_id: String::new(),
                        images: vec![],
                        is_error: false,
                    });
                } else if self.tool_iterations == 0 {
                    let _ = ui_tx.send(UiUpdate::ErrorMsg(
                        "Model returned an empty response. Try rephrasing or check your API key/quota.".into()
                    )).await;
                }

                self.finish_turn(ui_tx).await;
                break;
            }
        }
    }

    /// Poll the action channel for steering/cancel input that arrived while the
    /// agentic loop was busy — during tool execution (`execute_tools`), context
    /// compaction, a rate-limit sleep, or a network retry — anywhere the `'recv`
    /// loop isn't running to read it. Without this, a `CancelStream` sent while a
    /// tool is running sits unread in the channel, `cancel_flag` never gets set,
    /// and the loop just makes another stream request and keeps going.
    ///
    /// Handles `CancelStream` (sets the cancel flag and clears goal/plan state —
    /// the TUI already shows "Cancelled." locally, so no message is sent here)
    /// and `Steer` (executes it). Returns `true` if a cancel was requested, so
    /// the caller should stop the loop; `false` if the channel is empty.
    async fn drain_pending_actions(
        &mut self,
        ui_tx: &mpsc::Sender<UiUpdate>,
        action_rx: &mut mpsc::Receiver<Action>,
    ) -> bool {
        loop {
            match action_rx.try_recv() {
                Ok(Action::CancelStream) => {
                    self.cancel_flag
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    self.active_goal.clear();
                    self.tool_iterations = 0;
                    self.stall_nudges = 0;
                    self.plan_steps.clear();
                    self.plan_cursor = 0;
                    let _ = ui_tx.send(UiUpdate::PlanUpdate(vec![])).await;
                    return true;
                }
                Ok(Action::Steer(text)) => {
                    self.handle_steer(&text, ui_tx, action_rx).await;
                }
                Ok(_) => {} // ignore other actions while the loop is busy
                Err(mpsc::error::TryRecvError::Empty) => return false,
                Err(mpsc::error::TryRecvError::Disconnected) => return false,
            }
        }
    }

    /// Call `provider.stream(req)`, retrying up to `NETWORK_RETRIES` times on
    /// transient network errors (connection refused/timeout, DNS, TLS, mid-stream
    /// disconnect — anything `is_transient_network_error` matches). The request
    /// is deterministic given the (unchanged) history, so a retry is safe and
    /// idempotent. Non-transient errors (auth, 4xx/5xx, bad request) are reported
    /// to the UI and surfaced as `None` so the caller breaks the loop. Also
    /// surfaces a system message on each retry so the user sees why it's hanging.
    async fn stream_with_retry(
        &self,
        provider: Arc<dyn Provider>,
        req: StreamRequest,
        ui_tx: &mpsc::Sender<UiUpdate>,
    ) -> Option<mpsc::Receiver<StreamChunk>> {
        let mut last_err: Option<anyhow::Error> = None;
        let mut attempt = 0;
        while attempt <= NETWORK_RETRIES {
            if self.cancel_flag.load(std::sync::atomic::Ordering::SeqCst) {
                return None;
            }
            match provider.stream(req.clone()).await {
                Ok(stream) => return Some(stream),
                Err(e) => {
                    if !is_transient_network_error(&e) {
                        let _ = ui_tx.send(UiUpdate::ErrorMsg(e.to_string())).await;
                        return None;
                    }
                    last_err = Some(e);
                    attempt += 1;
                    if attempt > NETWORK_RETRIES {
                        break;
                    }
                    let backoff_ms = NETWORK_RETRY_BASE_MS * (1 << (attempt - 1));
                    let _ = ui_tx
                        .send(UiUpdate::SystemMsg(format!(
                            "Network error — retrying in {}ms (attempt {}/{}): {}",
                            backoff_ms,
                            attempt,
                            NETWORK_RETRIES,
                            last_err.as_ref().map(|e| e.to_string()).unwrap_or_default(),
                        )))
                        .await;
                    tokio::time::sleep(Duration::from_millis(backoff_ms)).await;
                }
            }
        }
        let _ = ui_tx
            .send(UiUpdate::ErrorMsg(
                last_err
                    .map(|e| format!("Still failing after {NETWORK_RETRIES} retries: {e}"))
                    .unwrap_or_else(|| format!("Network error after {NETWORK_RETRIES} retries.")),
            ))
            .await;
        None
    }

    /// Ends the current turn: resets iteration/stall counters, clears the
    /// active goal, persists the session, marks any leftover plan steps
    /// done, and tells the TUI the goal is complete. Shared by the
    /// plain-text-completion path and the explicit `mark_complete` tool call.
    async fn finish_turn(&mut self, ui_tx: &mpsc::Sender<UiUpdate>) {
        let tool_count = self.tool_iterations;
        self.tool_iterations = 0;
        self.stall_nudges = 0;
        self.active_goal.clear();
        self.save_session();

        // Any plan steps left un-resolved (model finished early, or the
        // plan overestimated how many turns the task needed) are done.
        if self.plan_cursor < self.plan_steps.len() {
            for step in &mut self.plan_steps[self.plan_cursor..] {
                step.status = TaskStatus::Completed;
            }
            self.plan_cursor = self.plan_steps.len();
            let _ = ui_tx
                .send(UiUpdate::PlanUpdate(self.plan_steps.clone()))
                .await;
        }

        let _ = ui_tx.send(UiUpdate::GoalComplete { tool_count }).await;
    }

    /// Returns each result paired with the group id it actually ran under —
    /// `Some(start)` only for calls that were part of a real 2+-call concurrent
    /// batch, `None` for anything that ran solo (including a parallel-safe call
    /// that happened to be alone, or one denied out of a batch). Callers should
    /// use this — not the pre-execution `parallel_group_ids` guess — as the
    /// source of truth for what actually ran concurrently.
    async fn execute_tools(
        &mut self,
        calls: &[ToolCall],
        denied: &HashSet<String>,
        ui_tx: &mpsc::Sender<UiUpdate>,
        action_rx: &mut mpsc::Receiver<Action>,
    ) -> Vec<(executor::ToolResult, Option<usize>)> {
        let mut results = Vec::with_capacity(calls.len());
        let mut i = 0;
        while i < calls.len() {
            if is_parallel_safe(&calls[i].name) {
                let start = i;
                while i < calls.len() && is_parallel_safe(&calls[i].name) {
                    i += 1;
                }
                let batch = &calls[start..i];
                let group = (batch.len() > 1).then_some(start);

                // Spawn every non-denied call in the batch *before* awaiting any of
                // them — spawn_blocking starts running on the blocking pool as soon
                // as it's called, so this is what actually makes them run
                // concurrently rather than one-at-a-time.
                let handles: Vec<Option<tokio::task::JoinHandle<executor::ToolResult>>> = batch
                    .iter()
                    .map(|c| (!denied.contains(&c.id)).then(|| self.spawn_tool(c, ui_tx)))
                    .collect();

                for h in handles {
                    // A denied call never actually spawned alongside the others —
                    // don't claim it ran in the group.
                    let ran_in_batch = h.is_some();
                    let result = match h {
                        None => executor::ToolResult {
                            output: "Command denied by user.".to_string(),
                            is_error: true,
                        },
                        Some(mut handle) => {
                            // Poll for cancel while the tool runs so Ctrl+C is
                            // felt even during a long-running command. The cancel
                            // flag is only set once the CancelStream action is
                            // read from the channel, so poll the channel here.
                            let res = tokio::select! {
                                r = &mut handle => r.unwrap_or_else(|e| executor::ToolResult {
                                    output: e.to_string(),
                                    is_error: true,
                                }),
                                maybe = action_rx.recv() => {
                                    match maybe {
                                        Some(Action::CancelStream) => {
                                            self.cancel_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                                            executor::ToolResult {
                                                output: "Cancelled by user.".to_string(),
                                                is_error: true,
                                            }
                                        }
                                        Some(Action::Steer(text)) => {
                                            self.handle_steer(&text, ui_tx, action_rx).await;
                                            // Re-await the tool handle after handling
                                            // the steer.
                                            handle.await.unwrap_or_else(|e| executor::ToolResult {
                                                output: e.to_string(),
                                                is_error: true,
                                            })
                                        }
                                        _ => {
                                            // Channel closed or other action — just
                                            // wait for the tool to finish.
                                            handle.await.unwrap_or_else(|e| executor::ToolResult {
                                                output: e.to_string(),
                                                is_error: true,
                                            })
                                        }
                                    }
                                }
                            };
                            res
                        }
                    };
                    results.push((result, if ran_in_batch { group } else { None }));
                }
                continue;
            }

            let call = &calls[i];
            i += 1;

            if denied.contains(&call.id) {
                results.push((
                    executor::ToolResult {
                        output: "Command denied by user.".to_string(),
                        is_error: true,
                    },
                    None,
                ));
                continue;
            }

            // mark_complete is a pure signal, not a real action — short-circuit
            // before spawn_tool/preflight so it never touches the filesystem or
            // approval funnel. The agentic loop below checks for it after this
            // batch finishes and ends the turn instead of continuing.
            if call.name == "mark_complete" {
                results.push((
                    executor::ToolResult {
                        output: "Acknowledged.".into(),
                        is_error: false,
                    },
                    None,
                ));
                continue;
            }

            // ask_user is interactive — send the question to the TUI and block
            // until the user types a reply, which is returned as the tool result.
            if call.name == "ask_user" {
                let input_map: std::collections::HashMap<String, String> =
                    serde_json::from_str(&call.input).unwrap_or_default();
                let question = input_map
                    .get("question")
                    .cloned()
                    .unwrap_or_else(|| "Please answer:".to_string());
                let answer = self.await_ask_user(ui_tx, action_rx, question).await;
                let (output, is_error) = match answer {
                    Some(a) if !a.trim().is_empty() => (a.trim().to_string(), false),
                    Some(_) => ("(no answer)".to_string(), false),
                    None => ("User cancelled.".to_string(), true),
                };
                results.push((executor::ToolResult { output, is_error }, None));
                continue;
            }

            // Handle skill invocations directly — needs access to self.skills
            if call.name == "run_skill" {
                let input_map: std::collections::HashMap<String, String> =
                    serde_json::from_str(&call.input).unwrap_or_default();
                let skill_name = input_map.get("name").cloned().unwrap_or_default();
                let query = input_map.get("query").cloned().unwrap_or_default();

                let result = if let Some(skill) =
                    self.skills.iter().find(|s| s.name == skill_name).cloned()
                {
                    if self.cfg.skill_subagents {
                        self.run_skill_as_subagent(&skill, &query, ui_tx, action_rx)
                            .await
                    } else if skill.is_shell() {
                        self.run_shell_skill(&skill, &query).await
                    } else if skill.is_prompt() {
                        match skills::executor::expand_prompt(&skill, &query) {
                            Ok(expanded) => executor::ToolResult {
                                output: expanded,
                                is_error: false,
                            },
                            Err(e) => executor::ToolResult {
                                output: e.to_string(),
                                is_error: true,
                            },
                        }
                    } else {
                        executor::ToolResult {
                            output: format!(
                                "skill '{skill_name}' has neither a shell chunk nor a prompt body"
                            ),
                            is_error: true,
                        }
                    }
                } else {
                    executor::ToolResult {
                        output: format!("skill '{skill_name}' not found"),
                        is_error: true,
                    }
                };
                results.push((result, None));
                continue;
            }

            // ── Background process tools (bg_start/bg_status/bg_log/bg_kill) ─
            // These operate on the engine's background registry, not the
            // blocking executor. `bg_start` returns immediately with an id so
            // the model can keep working while the process runs.
            if call.name == "bg_start"
                || call.name == "bg_status"
                || call.name == "bg_log"
                || call.name == "bg_kill"
            {
                let input_map: std::collections::HashMap<String, String> =
                    serde_json::from_str(&call.input).unwrap_or_default();
                let result = match call.name.as_str() {
                    "bg_start" => {
                        let cmd = input_map.get("command").cloned().unwrap_or_default();
                        if cmd.trim().is_empty() {
                            executor::ToolResult {
                                output: "bg_start requires 'command'".into(),
                                is_error: true,
                            }
                        } else {
                            match self.background.start(&cmd, &self.work_dir) {
                                Ok(id) => executor::ToolResult { output: format!(
                                    "background process started: id={id}  cmd={cmd:?}\n\
                                     Poll with bg_status id={id} and bg_log id={id}; stop with bg_kill id={id}."
                                ), is_error: false },
                                Err(e) => executor::ToolResult { output: e, is_error: true },
                            }
                        }
                    }
                    "bg_status" => {
                        let id = input_map.get("id").cloned().unwrap_or_default();
                        let statuses = self.background.status(&id);
                        if statuses.is_empty() {
                            executor::ToolResult {
                                output: if id.is_empty() {
                                    "no background processes running".into()
                                } else {
                                    format!("no background process with id '{id}'")
                                },
                                is_error: false,
                            }
                        } else {
                            let mut lines = Vec::new();
                            for s in &statuses {
                                let state = if s.running {
                                    "RUNNING".to_string()
                                } else {
                                    match s.exit_code {
                                        Some(0) => "exited (code 0)".into(),
                                        Some(c) => format!("exited (code {c})"),
                                        None => "exited (no code)".into(),
                                    }
                                };
                                lines.push(format!(
                                    "id={}  {state}  ~{}s  stdout={}B stderr={}B  cmd={:?}",
                                    s.id, s.elapsed_secs, s.stdout_len, s.stderr_len, s.cmd
                                ));
                            }
                            executor::ToolResult {
                                output: lines.join("\n"),
                                is_error: false,
                            }
                        }
                    }
                    "bg_log" => {
                        let id = input_map.get("id").cloned().unwrap_or_default();
                        match self.background.log(&id) {
                            Ok((out, err)) => {
                                let mut text = String::new();
                                if out.trim().is_empty() && err.trim().is_empty() {
                                    text = "(no new output since last poll)".into();
                                } else {
                                    if !out.trim().is_empty() {
                                        text.push_str(&format!("[stdout]\n{out}"));
                                    }
                                    if !err.trim().is_empty() {
                                        text.push_str(&format!("[stderr]\n{err}"));
                                    }
                                }
                                executor::ToolResult {
                                    output: text,
                                    is_error: false,
                                }
                            }
                            Err(e) => executor::ToolResult {
                                output: e,
                                is_error: true,
                            },
                        }
                    }
                    _ => {
                        // bg_kill
                        let id = input_map.get("id").cloned().unwrap_or_default();
                        match self.background.kill(&id) {
                            Ok(msg) => executor::ToolResult {
                                output: msg,
                                is_error: false,
                            },
                            Err(e) => executor::ToolResult {
                                output: e,
                                is_error: true,
                            },
                        }
                    }
                };
                results.push((result, None));
                continue;
            }

            // Route mcp__{server}__{tool} calls to the owning MCP connection.
            // Not treated as parallel-safe (unlike read_file et al.) — an MCP
            // tool's side effects are opaque to marlin, so batching several
            // together by default isn't a safe assumption to bake in.
            if let Some((server, tool_name)) = mcp::parse_tool_name(&call.name) {
                let result = match self.mcp_clients.get(server) {
                    Some(client) => {
                        let args: serde_json::Value =
                            serde_json::from_str(&call.input).unwrap_or_default();
                        match client.call_tool(tool_name, args).await {
                            Ok((output, is_error)) => executor::ToolResult { output, is_error },
                            Err(e) => executor::ToolResult {
                                output: e.to_string(),
                                is_error: true,
                            },
                        }
                    }
                    None => executor::ToolResult {
                        output: format!("mcp server '{server}' is not connected"),
                        is_error: true,
                    },
                };
                results.push((result, None));
                continue;
            }

            // ── Diff preview for file-mutating tools ──────────────────────────
            // Before executing write_file / edit_file / notebook_edit, compute
            // what the file would look like and show a unified diff. The user
            // can accept (tool runs) or reject (tool is skipped with a message).
            // Skipped entirely when `skip_permissions` is on — that mode means
            // "don't ask me about file operations", so the diff dialog would
            // just be noise.
            if !self.cfg.skip_permissions
                && (call.name == "write_file"
                    || call.name == "edit_file"
                    || call.name == "notebook_edit"
                    || call.name == "multi_edit")
            {
                if let Some(path) = extract_file_path(&call.input, &self.work_dir) {
                    let old_content = std::fs::read_to_string(&path).unwrap_or_default();
                    let new_content = match call.name.as_str() {
                        "write_file" => {
                            let input_map: std::collections::HashMap<String, String> =
                                serde_json::from_str(&call.input).unwrap_or_default();
                            input_map.get("content").cloned().unwrap_or_default()
                        }
                        "edit_file" => {
                            let input_map: std::collections::HashMap<String, String> =
                                serde_json::from_str(&call.input).unwrap_or_default();
                            let old_str = input_map.get("old_string").cloned().unwrap_or_default();
                            let new_str = input_map.get("new_string").cloned().unwrap_or_default();
                            old_content.replacen(&old_str, &new_str, 1)
                        }
                        "multi_edit" => {
                            // Apply every edit in order (mirrors the executor arm,
                            // including the empty-old_string guard).
                            let v: serde_json::Value =
                                serde_json::from_str(&call.input).unwrap_or_default();
                            let mut cur = old_content.clone();
                            if let Some(edits) = v["edits"].as_array() {
                                for edit in edits {
                                    let old = edit["old_string"].as_str().unwrap_or("");
                                    let new = edit["new_string"].as_str().unwrap_or("");
                                    if old.is_empty() {
                                        break;
                                    }
                                    cur = cur.replacen(old, new, 1);
                                }
                            }
                            cur
                        }
                        "notebook_edit" => {
                            // For notebooks, just show the raw input as the "new" side
                            let input_map: std::collections::HashMap<String, String> =
                                serde_json::from_str(&call.input).unwrap_or_default();
                            input_map.get("new_source").cloned().unwrap_or_default()
                        }
                        _ => String::new(),
                    };

                    if let Some(diff) = snapshots::diff_lines(&old_content, &new_content) {
                        let _ = ui_tx
                            .send(UiUpdate::DiffPreview {
                                tool_id: call.id.clone(),
                                path: path.clone(),
                                diff,
                            })
                            .await;

                        // Wait for user to accept or reject
                        let accepted = loop {
                            match action_rx.recv().await {
                                Some(Action::AcceptDiff { tool_id }) if tool_id == call.id => {
                                    break true
                                }
                                Some(Action::RejectDiff { tool_id }) if tool_id == call.id => {
                                    break false
                                }
                                Some(Action::CancelStream) => break false,
                                None => break false,
                                _ => {} // ignore other actions while waiting
                            }
                        };

                        if !accepted {
                            results.push((
                                executor::ToolResult {
                                    output: "Edit rejected by user — diff was not accepted."
                                        .to_string(),
                                    is_error: true,
                                },
                                None,
                            ));
                            continue;
                        }
                    }
                }
            }

            let mut handle = self.spawn_tool(call, ui_tx);
            let result = tokio::select! {
                r = &mut handle => r.unwrap_or_else(|e| executor::ToolResult {
                    output: e.to_string(),
                    is_error: true,
                }),
                maybe = action_rx.recv() => {
                    match maybe {
                        Some(Action::CancelStream) => {
                            self.cancel_flag.store(true, std::sync::atomic::Ordering::SeqCst);
                            executor::ToolResult {
                                output: "Cancelled by user.".to_string(),
                                is_error: true,
                            }
                        }
                        Some(Action::Steer(text)) => {
                            self.handle_steer(&text, ui_tx, action_rx).await;
                            handle.await.unwrap_or_else(|e| executor::ToolResult {
                                output: e.to_string(),
                                is_error: true,
                            })
                        }
                        _ => {
                            handle.await.unwrap_or_else(|e| executor::ToolResult {
                                output: e.to_string(),
                                is_error: true,
                            })
                        }
                    }
                }
            };
            results.push((result, None));
        }
        results
    }

    /// Build (but don't await) the blocking-pool task that actually runs one
    /// tool call. Split out of `execute_tools` so a parallel-safe batch can
    /// spawn several of these before awaiting any of them.
    fn spawn_tool(
        &self,
        call: &ToolCall,
        ui_tx: &mpsc::Sender<UiUpdate>,
    ) -> tokio::task::JoinHandle<executor::ToolResult> {
        let name = call.name.clone();
        let input = call.input.clone();
        let work_dir = self.work_dir.clone();
        let allowed = self.allowed_commands.clone();
        let marlin_dir = self.marlin_dir.clone();
        let wd2 = work_dir.clone();
        let logs_dir = marlin_dir.join("logs");
        let sandbox = self.cfg.sandbox_mode.allows_all() || self.cfg.skip_permissions;
        let clean_env = self.cfg.clean_env;
        let ast_mode = self.ast_mode.clone();
        let sandbox_mode = self.cfg.sandbox_mode.clone();

        let idx_clone = self.code_index.clone();
        let ext_tools = self.external_tools.clone();
        let stream_tx = ui_tx.clone();

        tokio::task::spawn_blocking(move || {
            let idx_search = idx_clone.clone();
            let search_fn: Option<Box<executor::SearchFn<'_>>> = idx_search.map(|idx| {
                let f: Box<executor::SearchFn<'_>> = Box::new(move |q: &str, lim: usize| {
                    let results = index::search(&idx, q, lim);
                    index::format_results(&results, q)
                });
                f
            });

            let symbol_search_fn: Option<Box<executor::SymbolSearchFn<'_>>> =
                idx_clone.map(|idx| {
                    let f: Box<executor::SymbolSearchFn<'_>> =
                        Box::new(move |sym: &str, lim: usize| {
                            let results = index::search_symbols(&idx, sym, lim);
                            index::format_symbol_results(&results, sym)
                        });
                    f
                });

            // Streaming callback for run_command: sends chunks to the TUI as they arrive
            let stream_fn: Option<Box<executor::StreamFn<'_>>> = if name == "run_command" {
                Some(Box::new(move |chunk: &str| {
                    let _ = stream_tx.blocking_send(UiUpdate::ToolStreamChunk {
                        chunk: chunk.to_string(),
                    });
                }))
            } else {
                None
            };

            executor::execute(
                &name,
                &input,
                &work_dir,
                &|cmd| sandbox || policy::is_command_allowed(cmd, &allowed),
                search_fn.as_deref(),
                symbol_search_fn.as_deref(),
                Some(&|abs_path: &str, tool: &str| {
                    snapshots::take(&marlin_dir, &wd2, abs_path, tool);
                }),
                stream_fn.as_deref(),
                Some(&logs_dir),
                clean_env,
                ast_mode,
                &sandbox_mode,
                &ext_tools,
            )
        })
    }

    /// Run every one of a skill's resolved chunk commands, in order, through the
    /// *real* preflight funnel (allow-list + sandbox mode) — chunks don't chain,
    /// so each runs independently and the first error stops the rest. Used by
    /// both the LLM tool-call path (execute_tools, above) and the interactive
    /// `/skill run` path (handle_slash_command, below) so the two can't drift
    /// out of sync the way they used to.
    ///
    /// If the skill also has a prompt body (chunks + body — the qmd format's
    /// one genuinely new capability), the expanded prose is prepended to the
    /// combined chunk output so both reach the model together.
    ///
    /// A `NeedApproval` verdict (destructive-but-permitted) is treated as
    /// already cleared here — the LLM tool-call path clears it upstream in
    /// `run_approval_checks` before `execute_tools` ever runs; interactive
    /// callers that haven't already prompted the user must call
    /// `preflight::check` themselves first.
    async fn run_shell_skill(&self, skill: &Skill, query: &str) -> executor::ToolResult {
        let cmds = match skills::executor::resolve_chunks(skill, query) {
            Ok(c) => c,
            Err(e) => {
                return executor::ToolResult {
                    output: e.to_string(),
                    is_error: true,
                }
            }
        };

        let mut outputs = Vec::with_capacity(cmds.len());
        for cmd in cmds {
            match self.preflight_shell(&cmd) {
                Err(result) => return result,
                Ok(_verdict) => {
                    let result = self.run_shell(cmd).await;
                    if result.is_error {
                        return result;
                    }
                    outputs.push(result.output);
                }
            }
        }

        let prose = if skill.is_prompt() {
            skills::executor::expand_prompt(skill, query).unwrap_or_default()
        } else {
            String::new()
        };
        let output = if prose.is_empty() {
            outputs.join("\n\n")
        } else {
            format!("{prose}\n\n{}", outputs.join("\n\n"))
        };
        executor::ToolResult {
            output,
            is_error: false,
        }
    }

    /// Delegate a skill invocation to a subagent (see `engine::subagent`) instead
    /// of running it inline — the current default (`cfg.skill_subagents`).
    /// Every skill shape goes through this uniformly: shell/combined skills
    /// instruct the subagent to run the exact resolved command(s) via
    /// `run_command` (deterministic — the subagent doesn't get to paraphrase
    /// them), prompt-only skills hand it the expanded template directly. The
    /// subagent has its own tools and reports back one final summary.
    async fn run_skill_as_subagent(
        &self,
        skill: &Skill,
        query: &str,
        ui_tx: &mpsc::Sender<UiUpdate>,
        action_rx: &mut mpsc::Receiver<Action>,
    ) -> executor::ToolResult {
        let instructions = match subagent::build_task(skill, query) {
            Ok(s) => s,
            Err(msg) => {
                return executor::ToolResult {
                    output: msg,
                    is_error: true,
                }
            }
        };

        let (provider_name, model) = self.subagent_model();
        let provider = match self.registry.get(&provider_name) {
            Ok(p) => p,
            Err(e) => {
                return executor::ToolResult {
                    output: e.to_string(),
                    is_error: true,
                }
            }
        };

        let result = subagent::run(
            &skill.name,
            &instructions,
            provider,
            &model,
            &self.cfg,
            &self.allowed_commands,
            &self.work_dir,
            &self.marlin_dir,
            self.code_index.as_ref(),
            ui_tx,
            action_rx,
            &self.cancel_flag,
        )
        .await;

        executor::ToolResult {
            output: result.output,
            is_error: result.is_error,
        }
    }

    /// Provider/model a subagent should use: `model_tiers.default` when
    /// tiers are configured (regardless of whether difficulty-based routing
    /// is enabled for the main conversation — this is a separate mechanism),
    /// else whatever the main conversation is using.
    fn subagent_model(&self) -> (String, String) {
        self.cfg
            .model_tiers
            .as_ref()
            .filter(|t| t.enabled)
            .map(|t| (t.default.provider.clone(), t.default.model.clone()))
            .unwrap_or_else(|| {
                (
                    self.cfg.active_provider.clone(),
                    self.cfg.active_model.clone(),
                )
            })
    }

    /// Preflight-check a resolved shell command against the real allow-list and
    /// sandbox mode. `Err` means the command is unconditionally denied and
    /// should never run; `Ok(verdict)` may still be `NeedApproval`.
    fn preflight_shell(&self, cmd: &str) -> Result<preflight::Verdict, executor::ToolResult> {
        let inv = preflight::Invocation::shell("run_command", cmd);
        let verdict = preflight::check(&inv, &self.cfg, &self.allowed_commands);
        if let preflight::Verdict::Deny(reason) = &verdict {
            return Err(executor::ToolResult {
                output: reason.clone(),
                is_error: true,
            });
        }
        Ok(verdict)
    }

    /// Execute a resolved shell command with the engine's real work_dir,
    /// allow-list, clean_env, and sandbox mode.
    async fn run_shell(&self, cmd: String) -> executor::ToolResult {
        let cmd_json = serde_json::json!({"command": cmd}).to_string();
        let wd = self.work_dir.clone();
        let clean_env = self.cfg.clean_env;
        let logs_dir = self.marlin_dir.join("logs");
        let allowed = self.allowed_commands.clone();
        let sandbox = self.cfg.sandbox_mode.allows_all() || self.cfg.skip_permissions;
        let sandbox_mode = self.cfg.sandbox_mode.clone();
        tokio::task::spawn_blocking(move || {
            executor::execute(
                "run_command",
                &cmd_json,
                &wd,
                &|c: &str| sandbox || policy::is_command_allowed(c, &allowed),
                None,
                None,
                None,
                None,
                Some(&logs_dir),
                clean_env,
                marlin_config::AstMode::Off,
                &sandbox_mode,
                &[],
            )
        })
        .await
        .unwrap_or_else(|e| executor::ToolResult {
            output: e.to_string(),
            is_error: true,
        })
    }

    /// Returns the set of tool call IDs the user denied. This is the single
    /// interactive-approval funnel: destructive `run_command` calls, destructive
    /// shell-skill calls (resolved through the same skill_command path
    /// run_shell_skill uses), and filesystem calls whose path would escape
    /// work_dir all route through here before execute_tools ever runs.
    async fn run_approval_checks(
        &mut self,
        calls: &[ToolCall],
        ui_tx: &mpsc::Sender<UiUpdate>,
        action_rx: &mut mpsc::Receiver<Action>,
    ) -> HashSet<String> {
        // `--dangerously-skip-permissions` / `/permissions skip`. The
        // `run_command` arm below checks `is_destructive_cmd` directly rather
        // than through `preflight::check` (which already short-circuits on
        // `skip_permissions`), so it needs its own bypass here too.
        if self.cfg.skip_permissions {
            return HashSet::new();
        }
        let mut denied = HashSet::new();
        for tc in calls {
            // `Verdict` (not just an approval-reason string) so a path-based `Deny`
            // is handled the same way `save_editor_file` handles it — auto-blocked
            // and reported, not silently treated as allowed — rather than the two
            // callers of `preflight::check` disagreeing on what Deny means.
            let verdict = match tc.name.as_str() {
                "run_command" => {
                    let cmd = extract_cmd_str(&tc.input);
                    preflight::is_destructive_cmd(&cmd).then(|| {
                        preflight::Verdict::NeedApproval(format!("destructive command: {cmd}"))
                    })
                }
                // Only pre-check here when skills run inline (cfg.skill_subagents off).
                // When delegated to a subagent, its own tool-call loop (run_one_tool)
                // does this same preflight+approval per call as it actually runs
                // commands — checking here too would just double-prompt the user.
                "run_skill" if !self.cfg.skill_subagents => {
                    let input_map: std::collections::HashMap<String, String> =
                        serde_json::from_str(&tc.input).unwrap_or_default();
                    let skill_name = input_map.get("name").cloned().unwrap_or_default();
                    let query = input_map.get("query").cloned().unwrap_or_default();
                    self.skills
                        .iter()
                        .find(|s| s.name == skill_name && s.is_shell())
                        .and_then(|skill| skills::executor::resolve_chunks(skill, &query).ok())
                        .and_then(|cmds| {
                            cmds.into_iter()
                                .find(|cmd| preflight::is_destructive_cmd(cmd))
                        })
                        .map(|cmd| {
                            preflight::Verdict::NeedApproval(format!(
                                "destructive skill command: {cmd}"
                            ))
                        })
                }
                "read_file" | "write_file" | "edit_file" | "notebook_edit" | "multi_edit"
                | "create_directory" => extract_path_field(&tc.input).map(|path| {
                    let resolved = executor::resolve_path(&path, &self.work_dir);
                    let inv = preflight::Invocation::paths(tc.name.clone(), vec![resolved]);
                    preflight::check(&inv, &self.cfg, &self.allowed_commands)
                }),
                _ => None,
            };

            match verdict {
                None | Some(preflight::Verdict::Allow) => {}
                Some(preflight::Verdict::Deny(reason)) => {
                    let _ = ui_tx
                        .send(UiUpdate::ErrorMsg(format!("Denied: {reason}")))
                        .await;
                    denied.insert(tc.id.clone());
                }
                Some(preflight::Verdict::NeedApproval(reason)) => {
                    if !self.await_approval(ui_tx, action_rx, reason).await {
                        denied.insert(tc.id.clone());
                    }
                }
            }
        }
        denied
    }

    /// Send an approval prompt and block until the user responds. Shared by
    /// `run_approval_checks` (LLM tool-call path) and the interactive
    /// `/skill run` and user `/command` paths, which have no upstream
    /// approval gate of their own.
    async fn await_approval(
        &mut self,
        ui_tx: &mpsc::Sender<UiUpdate>,
        action_rx: &mut mpsc::Receiver<Action>,
        reason: String,
    ) -> bool {
        let _ = ui_tx.send(UiUpdate::AwaitingApproval { cmd: reason }).await;
        loop {
            match action_rx.recv().await {
                Some(Action::Approve) => break true,
                Some(Action::Deny) => break false,
                Some(Action::CancelStream) => {
                    self.cancel_flag
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    break false;
                }
                None => break false,
                _ => {} // ignore other actions while modal is open
            }
        }
    }

    /// Ask the user a question (from the model's ask_user tool call) and block
    /// until they type an answer. Returns `Some(answer)` on a reply, or `None`
    /// if the user cancelled / the channel closed.
    async fn await_ask_user(
        &mut self,
        ui_tx: &mpsc::Sender<UiUpdate>,
        action_rx: &mut mpsc::Receiver<Action>,
        question: String,
    ) -> Option<String> {
        let _ = ui_tx.send(UiUpdate::AskUser { question }).await;
        loop {
            match action_rx.recv().await {
                Some(Action::UserAnswer(answer)) => break Some(answer),
                Some(Action::CancelStream) => {
                    self.cancel_flag
                        .store(true, std::sync::atomic::Ordering::SeqCst);
                    break None;
                }
                None => break None,
                _ => {} // ignore other actions while the question modal is open
            }
        }
    }

    /// Run the configured verify_command after a file edit. Returns a Message to inject if
    /// the command fails (or None if passing / not configured).
    async fn run_verify_command(&self, ui_tx: &mpsc::Sender<UiUpdate>) -> Option<Message> {
        let cmd = self.cfg.verify_command.as_deref()?.to_string();
        let work_dir = self.work_dir.clone();
        let clean_env = self.cfg.clean_env;

        let _ = ui_tx
            .send(UiUpdate::SystemMsg(format!(
                "[Marlin Verify] Running: {cmd}"
            )))
            .await;

        let result = match tokio::task::spawn_blocking(move || {
            let mut command = std::process::Command::new("sh");
            command.arg("-c").arg(&cmd).current_dir(&work_dir);
            if clean_env {
                command.env_clear();
                for var in executor::CLEAN_ENV_VARS {
                    if let Ok(val) = std::env::var(var) {
                        command.env(var, val);
                    }
                }
            }
            command.output()
        })
        .await
        {
            Ok(Ok(out)) => out,
            _ => return None,
        };

        let stdout = String::from_utf8_lossy(&result.stdout).to_string();
        let stderr = String::from_utf8_lossy(&result.stderr).to_string();
        let combined = format!("{stdout}{stderr}").trim().to_string();

        if result.status.success() {
            let _ = ui_tx
                .send(UiUpdate::SystemMsg(
                    "[Marlin Verify] ✓ Tests passed.".into(),
                ))
                .await;
            None
        } else {
            let snippet: String = combined
                .lines()
                .rev()
                .take(60)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join("\n");
            let msg = format!(
                "[Marlin Verify] Tests failed (exit {}). Fix the errors before continuing.\n\n{}",
                result.status.code().unwrap_or(-1),
                snippet
            );
            let _ = ui_tx
                .send(UiUpdate::SystemMsg(
                    "[Marlin Verify] ✗ Tests failed — injecting error into context.".into(),
                ))
                .await;
            Some(Message::new_user(msg))
        }
    }

    /// Create a git checkpoint before an agentic turn (opt-in via
    /// `/checkpoints on`). Best-effort and non-blocking: any failure (not a
    /// git repo, git not installed, commit error) just logs a message and
    /// the turn proceeds normally. Uses the engine's blocking pool so the
    /// git subprocess never blocks the async loop.
    async fn maybe_checkpoint(&self, ui_tx: &mpsc::Sender<UiUpdate>) {
        if !self.cfg.checkpoints {
            return;
        }
        if !checkpoint::available(&self.work_dir) {
            let _ = ui_tx.send(UiUpdate::SystemMsg(
                "Checkpoints are on but this is not a git repository — skipping. (git required for /checkpoints)".into()
            )).await;
            return;
        }
        let work_dir = self.work_dir.clone();
        let hash = tokio::task::spawn_blocking(move || checkpoint::create(&work_dir)).await;
        match hash {
            Ok(Ok(h)) if !h.is_empty() => {
                let _ = ui_tx
                    .send(UiUpdate::SystemMsg(format!(
                        "Checkpoint created ({h}) — /undo will roll this turn back."
                    )))
                    .await;
            }
            Ok(Ok(_)) => {
                // No changes to commit — nothing to checkpoint.
            }
            Ok(Err(e)) => {
                let _ = ui_tx
                    .send(UiUpdate::SystemMsg(format!("Checkpoint skipped: {e}")))
                    .await;
            }
            Err(e) => {
                let _ = ui_tx
                    .send(UiUpdate::SystemMsg(format!("Checkpoint skipped: {e}")))
                    .await;
            }
        }
    }

    /// LLM-based context compaction: summarize old turns when approaching token budget.
    async fn maybe_compact_history(&mut self, ui_tx: &mpsc::Sender<UiUpdate>) {
        const KEEP_RECENT: usize = 8;

        // Compact when the conversation reaches ~70% of the configured context
        // budget (default 100k → 70k, matching the old hardcoded threshold).
        // Raising /budget raises the point at which compaction kicks in.
        let compact_above = (self.token_budget as f64 * 0.70) as usize;

        let cur_tokens = estimate_tokens(&self.history, "");
        if cur_tokens < compact_above {
            return;
        }
        if self.history.len() <= KEEP_RECENT {
            return;
        }
        // Don't re-compact immediately after a previous compaction; wait for 5k more tokens
        if self.compact_guard_tokens > 0 && cur_tokens < self.compact_guard_tokens + 5_000 {
            return;
        }

        let split = self.history.len() - KEEP_RECENT;
        let old: Vec<Message> = self.history[..split].to_vec();

        // Prefer the cheapest/fastest model so compaction doesn't waste quota
        let (compact_provider, compact_model) = self.cheapest_model();
        let provider = match self.registry.get(&compact_provider) {
            Ok(p) => p,
            Err(_) => return,
        };

        let ctx = compact_serialize(&old);

        let summary_req = StreamRequest {
            model: compact_model,
            messages: vec![Message::new_user(format!(
                "Produce a dense technical summary of this coding session fragment for an AI \
                coding assistant. Include: files created/modified (with key changes), commands \
                run and their outcomes, errors encountered and how they were resolved, decisions \
                made, and current task state. Be precise and comprehensive — this summary \
                replaces the original turns in context.\n\n{ctx}"
            ))],
            system_prompt: String::new(),
            max_tokens: 1500,
            tools: vec![],
            thinking: false,
        };

        let _ = ui_tx
            .send(UiUpdate::SystemMsg(format!(
                "Compacting context (~{cur_tokens} tokens) — summarizing {split} older turns…"
            )))
            .await;
        let mut stream = match provider.stream(summary_req).await {
            Ok(s) => s,
            Err(_) => return,
        };

        let mut summary = String::new();
        while let Some(chunk) = stream.recv().await {
            summary.push_str(&chunk.content);
            if chunk.done {
                break;
            }
        }

        if summary.trim().is_empty() {
            return;
        }

        let recent = self.history.split_off(split);
        self.history.clear();
        self.history.push(Message::new_user(format!(
            "[Marlin Context Summary — {split} turns condensed]\n{}",
            summary.trim()
        )));
        self.history.extend(recent);

        let new_tokens = estimate_tokens(&self.history, "");
        self.compact_guard_tokens = new_tokens;

        let _ = ui_tx
            .send(UiUpdate::SystemMsg(format!(
                "Context compacted: {split} turns → 1 summary (~{new_tokens} tokens now)."
            )))
            .await;
    }

    /// Returns (provider, model) for cheap compaction calls, preferring haiku > sonnet > active.
    fn cheapest_model(&self) -> (String, String) {
        let p = &self.cfg.active_provider;
        if let Ok(prov) = self.registry.get(p) {
            let models = prov.models();
            if let Some(m) = models.iter().find(|m| m.contains("haiku")) {
                return (p.clone(), m.clone());
            }
            if let Some(m) = models.iter().find(|m| m.contains("sonnet")) {
                return (p.clone(), m.clone());
            }
            // Fall back to the first model in the provider's list (typically the
            // smallest/cheapest), not the active model which could be expensive.
            if let Some(m) = models.first() {
                return (p.clone(), m.clone());
            }
        }
        (
            self.cfg.active_provider.clone(),
            self.cfg.active_model.clone(),
        )
    }

    // ── Model tier routing ────────────────────────────────────────────────────

    /// Select provider/model for this request based on difficulty score.
    async fn rate_and_route(&mut self, message: &str, ui_tx: &mpsc::Sender<UiUpdate>) {
        let Some(tiers) = self.cfg.model_tiers.clone() else {
            self.req_provider = self.cfg.active_provider.clone();
            self.req_model = self.cfg.active_model.clone();
            self.req_backup_provider.clear();
            self.req_backup_model.clear();
            return;
        };
        if !tiers.enabled {
            self.req_provider = self.cfg.active_provider.clone();
            self.req_model = self.cfg.active_model.clone();
            self.req_backup_provider.clear();
            self.req_backup_model.clear();
            return;
        }

        let score = self.rate_difficulty(message, &tiers).await;
        let tier_label = if score <= tiers.default_max_difficulty {
            "default"
        } else {
            "complex"
        };
        let _ = ui_tx
            .send(UiUpdate::TierSelected {
                score,
                tier: tier_label.into(),
            })
            .await;

        let selected: &ModelTier = if score <= tiers.default_max_difficulty {
            &tiers.default
        } else {
            &tiers.complex
        };

        self.req_provider = selected.provider.clone();
        self.req_model = selected.model.clone();
        self.req_backup_provider = selected.backup_provider.clone();
        self.req_backup_model = selected.backup_model.clone();
    }

    /// Ask the rater model to score a task 1–100.
    async fn rate_difficulty(&self, message: &str, tiers: &marlin_config::ModelTiers) -> u8 {
        let Ok(rater) = self.registry.get(&tiers.rater.provider) else {
            return 50;
        };
        let req = StreamRequest {
            model: tiers.rater.model.clone(),
            messages: vec![Message::new_user(format!(
                "Rate the difficulty of this coding task from 1 to 100 where 1 is trivial \
                and 100 is extremely complex architecture work. Reply with ONLY the number.\n\nTask: {message}"
            ))],
            system_prompt: String::new(),
            max_tokens: 8,
            tools: vec![],
            thinking: false,
        };
        let mut text = String::new();
        let stream_fut = async {
            if let Ok(mut stream) = rater.stream(req).await {
                while let Some(chunk) = stream.recv().await {
                    text.push_str(&chunk.content);
                    if chunk.done {
                        break;
                    }
                }
            }
        };
        // 10-second timeout — if the rater model hangs, fall back to default score.
        if tokio::time::timeout(Duration::from_secs(10), stream_fut)
            .await
            .is_err()
        {
            return 50;
        }
        text.trim().parse::<u8>().unwrap_or(50).clamp(1, 100)
    }

    /// Ask the already-routed model (see `rate_and_route`) for a short upfront
    /// plan before the tool loop starts, so the sidebar can show intended steps
    /// rather than only a retrospective log. Best-effort: any failure to reach
    /// the provider or parse a usable plan just leaves `plan_steps` empty —
    /// the granular `task_steps` log works exactly as before regardless.
    async fn maybe_generate_plan(&mut self, message: &str, ui_tx: &mpsc::Sender<UiUpdate>) {
        self.plan_steps.clear();
        self.plan_cursor = 0;

        let Ok(provider) = self.registry.get(&self.req_provider) else {
            return;
        };
        let req = StreamRequest {
            model: self.req_model.clone(),
            messages: vec![Message::new_user(format!(
                "Break the following coding task into 2-6 short, concrete, ordered \
                steps (imperative mood, under 8 words each). Reply with ONLY the \
                steps, one per line, no numbering or bullets, no other text. If the \
                task genuinely needs just one step, reply with one line.\n\nTask: {message}"
            ))],
            system_prompt: String::new(),
            max_tokens: 200,
            tools: vec![],
            thinking: false,
        };

        let mut text = String::new();
        let stream_fut = async {
            if let Ok(mut stream) = provider.stream(req).await {
                while let Some(chunk) = stream.recv().await {
                    text.push_str(&chunk.content);
                    if chunk.done {
                        break;
                    }
                }
            }
        };
        // 10-second timeout — if the plan model hangs, proceed without a plan.
        if tokio::time::timeout(Duration::from_secs(10), stream_fut)
            .await
            .is_err()
        {
            return;
        }

        let steps: Vec<TaskStep> = text
            .lines()
            .map(|l| l.trim().trim_start_matches(['-', '*', '•']).trim())
            .map(|l| {
                l.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ')')
                    .trim()
            })
            .filter(|l| !l.is_empty())
            .take(8)
            .map(TaskStep::planned)
            .collect();

        if !steps.is_empty() {
            self.plan_steps = steps;
            let _ = ui_tx
                .send(UiUpdate::PlanUpdate(self.plan_steps.clone()))
                .await;
        }
    }

    // ── Nightly daemon ────────────────────────────────────────────────────────

    fn maybe_spawn_daemon(&self, ui_tx: mpsc::Sender<UiUpdate>) {
        let Some(tiers) = &self.cfg.model_tiers else {
            return;
        };
        let Ok(provider) = self.registry.get(&tiers.rater.provider) else {
            return;
        };
        let model = tiers.rater.model.clone();
        skills::daemon::spawn(self.marlin_dir.clone(), provider, model, ui_tx);
    }

    /// Spawn every configured MCP server and fetch its tool list. Best-effort
    /// per server (a server that fails to spawn or handshake just gets
    /// reported and skipped, matching skill-validation's fail-soft style) —
    /// used both at startup and by `/mcp reload`.
    async fn connect_mcp_servers(&mut self, ui_tx: &mpsc::Sender<UiUpdate>) {
        self.mcp_clients.clear();
        self.mcp_tools.clear();
        for cfg in self.mcp_server_configs.clone() {
            match mcp::client::McpClient::spawn(&cfg).await {
                Ok(client) => match client.list_tools().await {
                    Ok(tools) => {
                        let count = tools.len();
                        for tool in tools {
                            self.mcp_tools.push((cfg.name.clone(), tool));
                        }
                        self.mcp_clients.insert(cfg.name.clone(), Arc::new(client));
                        let _ = ui_tx
                            .send(UiUpdate::SystemMsg(format!(
                                "mcp: connected '{}' ({count} tool(s))",
                                cfg.name
                            )))
                            .await;
                    }
                    Err(e) => {
                        let _ = ui_tx
                            .send(UiUpdate::ErrorMsg(format!(
                                "mcp: '{}' connected but tools/list failed: {e}",
                                cfg.name
                            )))
                            .await;
                    }
                },
                Err(e) => {
                    let _ = ui_tx
                        .send(UiUpdate::ErrorMsg(format!(
                            "mcp: '{}' failed to start: {e}",
                            cfg.name
                        )))
                        .await;
                }
            }
        }
    }

    /// Re-scan the working directory for files whose mtime/size changed since
    /// the last refresh and re-index just those (or drop files that were
    /// deleted). Runs periodically from the main action loop so the index stays
    /// fresh without a full rebuild on every edit. Returns the number of files
    /// re-indexed this round (0 when nothing changed).
    fn maybe_refresh_index(&mut self) -> usize {
        let Some(_idx) = &self.code_index else {
            return 0;
        };
        let work_dir = self.work_dir.clone();
        let (changed, next_state) = index::diff_against(&self.index_refresh, &work_dir);
        self.index_refresh = next_state;
        if changed.is_empty() {
            return 0;
        }
        let mut reindexed = 0;
        for rel in &changed {
            let abs = std::path::Path::new(&work_dir).join(rel);
            // Deleted files are dropped from the index.
            if !abs.exists() {
                if let Some(idx) = &mut self.code_index {
                    index::remove_file(idx, &abs.to_string_lossy());
                }
                continue;
            }
            // Skip binary/junk extensions the walker itself skips.
            if let Some(ext) = std::path::Path::new(rel)
                .extension()
                .and_then(|e| e.to_str())
            {
                let ext = ext.to_lowercase();
                let skips = [
                    "exe", "dll", "so", "dylib", "png", "jpg", "jpeg", "gif", "webp", "ico", "pdf",
                    "zip", "tar", "gz", "wasm", "bin", "lock",
                ];
                if skips.contains(&ext.as_str()) {
                    continue;
                }
            }
            if let Some(idx) = &mut self.code_index {
                index::update_file(idx, &abs.to_string_lossy());
            }
            reindexed += 1;
        }
        if reindexed > 0 {
            if let Some(idx) = &self.code_index {
                index::save(&self.marlin_dir, idx);
            }
        }
        reindexed
    }

    /// Skill names/descriptions to advertise in the `run_skill` tool description
    /// for this turn. Bounded by trigger-matching against `query` instead of
    /// listing every installed skill (which grows unbounded with skill count) —
    /// falls back to names-only (no descriptions) when nothing matches, so the
    /// model still knows what's available without paying for every description.
    fn skill_tool_list(&self, query: &str) -> Vec<(String, String)> {
        let skill_defs: Vec<SkillDef> = self.skills.iter().map(SkillDef::from).collect();
        let matched = skills::suggest::match_skills(query, &skill_defs);
        if matched.is_empty() {
            self.skills
                .iter()
                .map(|s| (s.name.clone(), String::new()))
                .collect()
        } else {
            matched
                .into_iter()
                .map(|m| (m.name, m.description))
                .collect()
        }
    }

    fn effective_system_prompt(&self) -> String {
        let mut s = String::new();
        s.push_str("You are Marlin, an AI coding assistant running in a terminal.\n");
        s.push_str("You help the user write, debug, and understand code.\n\n");

        // The tool list itself is duplication: every tool name and description
        // below is already sent as structured tool defs (see tools::all_tools) —
        // restating it here cost ~150 tokens on every request for nothing.
        s.push_str("When asked to create a file, write code, edit something, or run a command — DO IT with the appropriate tool. ");
        s.push_str("Do not explain how the user could do it themselves. Do not ask for confirmation before using tools. Just act.\n\n");

        s.push_str(&format!("Working directory: {}\n\n", self.work_dir));

        if !self.active_goal.is_empty() {
            s.push_str("## Active Goal\n");
            s.push_str(&self.active_goal);
            s.push('\n');
            s.push_str("\nWork toward this goal using tools. Keep calling tools until the task is fully complete.\n");
            s.push_str("When — and only when — every part of the goal is actually done, call the mark_complete tool \
                (alone, with no other tool calls in that turn) with a short summary. Do not just say you're done or \
                describe what you're about to do in plain text and stop; if you catch yourself writing \"let me now...\" \
                or \"I'll...\", make that tool call instead. Plain text with no tool call is only for asking the user \
                a question or reporting you're permanently blocked.\n");
            s.push_str(&format!(
                "Progress so far: {} tool calls made.\n",
                self.tool_iterations
            ));
        }

        if !self.cfg.system_prompt.is_empty() {
            s.push_str("\nAdditional instructions:\n");
            s.push_str(&self.cfg.system_prompt);
        }

        // Per-provider system prompt override takes precedence over global
        if let Some(override_prompt) = self
            .cfg
            .provider_system_prompts
            .get(&self.cfg.active_provider)
        {
            if !override_prompt.is_empty() {
                s.push_str("\nProvider-specific instructions:\n");
                s.push_str(override_prompt);
            }
        }

        match &self.ast_mode {
            AstMode::Off => {}
            AstMode::SExpr => {
                s.push_str("\n## AST Context Mode: SEXPR\n");
                s.push_str("File reads are delivered as compact S-expression AST representations produced by `ast-compiler decompile --format sexpr`, not raw source text.\n");
                s.push_str("The root token is `(meta ...)` followed by the recursive node tree.\n");
                s.push_str("Parse the tree structurally when reasoning about code. When you need to write changes, use write_file or edit_file with reconstructed source text.\n");
            }
            AstMode::Harness => {
                s.push_str("\n## AST Context Mode: HARNESS\n");
                s.push_str("You have three specialized AST tools available. Prefer them over read_file/edit_file for all code understanding and mutation:\n");
                s.push_str("  ast_skeleton  <file>                  — API surface map (signatures, no bodies). Always start here.\n");
                s.push_str("  ast_get_node  <file> <node_id>        — Full JSON for one node. Use after skeleton to inspect a target.\n");
                s.push_str("  ast_mutate    <file> <node_id> <op>   — Structural edit + automatic recompile + optimize.\n\n");
                s.push_str("CRITICAL RULES:\n");
                s.push_str(
                    "  1. Do NOT use edit_file for code mutations — use ast_mutate instead.\n",
                );
                s.push_str("  2. ast_mutate operations are: str-replace (old_json/new_json), append-stmt (statement_json), insert-before (index/statement_json).\n");
                s.push_str("  3. Always supply lang and source_file to ast_mutate so the source is regenerated deterministically.\n");
                s.push_str("  4. JSON values in node directives must be valid JSON, not source-code strings.\n");
            }
        }

        s
    }

    fn build_message_content(&self, text: &str) -> String {
        if self.attachments.is_empty() {
            return text.to_string();
        }
        let mut s = String::new();
        for (filename, content) in &self.attachments {
            let ext = Path::new(filename)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            s.push_str(&format!("File: {filename}\n```{ext}\n{content}\n```\n\n"));
        }
        s.push_str(text);
        s
    }

    fn save_session(&mut self) {
        let Some(session) = &mut self.session else {
            return;
        };
        session.messages = self.history.iter().map(to_session_message).collect();
        history::save_session(&self.marlin_dir, session);
    }

    /// Convert the current `self.history` into `HistoryEntry` values so the TUI
    /// can repopulate its chat display after a `/resume` / `/history <n>` /
    /// `--resume-last`. The engine keeps history as provider `Message`s; the TUI
    /// only needs a light rendering shape, so we flatten tool calls into their
    /// own entries (matching how `build_lines` in the TUI expects them).
    fn history_entries(&self) -> Vec<marlin_core::ui::HistoryEntry> {
        let mut out = Vec::new();
        for m in &self.history {
            match m.role.as_str() {
                "assistant" if !m.tool_calls.is_empty() => {
                    // Emit the assistant text (if any), then one entry per tool call.
                    if !m.content.trim().is_empty() {
                        out.push(marlin_core::ui::HistoryEntry {
                            role: "assistant".into(),
                            content: m.content.clone(),
                            tool_name: String::new(),
                            tool_input: String::new(),
                            is_error: false,
                        });
                    }
                    for tc in &m.tool_calls {
                        out.push(marlin_core::ui::HistoryEntry {
                            role: "assistant".into(),
                            content: String::new(),
                            tool_name: tc.name.clone(),
                            tool_input: tc.input.clone(),
                            is_error: false,
                        });
                    }
                }
                "assistant" => {
                    out.push(marlin_core::ui::HistoryEntry {
                        role: "assistant".into(),
                        content: m.content.clone(),
                        tool_name: String::new(),
                        tool_input: String::new(),
                        is_error: false,
                    });
                }
                "tool" => {
                    out.push(marlin_core::ui::HistoryEntry {
                        role: "tool".into(),
                        content: m.content.clone(),
                        tool_name: String::new(),
                        tool_input: String::new(),
                        is_error: m.is_error,
                    });
                }
                _ => {
                    out.push(marlin_core::ui::HistoryEntry {
                        role: "user".into(),
                        content: m.content.clone(),
                        tool_name: String::new(),
                        tool_input: String::new(),
                        is_error: false,
                    });
                }
            }
        }
        out
    }

    // ── Slash command handler ─────────────────────────────────────────────────

    /// Handle a slash command. Returns `Some(prompt)` if a prompt-type user command
    /// expanded a template that should be injected into the agentic loop.
    /// Handle a steering input sent while the model is working (or idle).
    /// Slash commands are executed instantly and their output is shown as a
    /// text field in the model's output area; plain text is shown verbatim as
    /// a steering note. Either way the in-flight stream is never interrupted.
    async fn handle_steer(
        &mut self,
        text: &str,
        ui_tx: &mpsc::Sender<UiUpdate>,
        action_rx: &mut mpsc::Receiver<Action>,
    ) {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return;
        }

        // Slash command → execute instantly, capture output as a text field.
        if trimmed.starts_with('/') {
            // /compact is handled specially (no agentic loop, no prompt injection).
            let cmd_name = trimmed
                .split_whitespace()
                .next()
                .unwrap_or("")
                .to_lowercase();
            if cmd_name == "/compact" {
                self.manual_compact(ui_tx).await;
                return;
            }
            if let Some(prompt) = self.handle_slash_command(trimmed, ui_tx, action_rx).await {
                // A prompt-type user command while steering — just show the
                // expanded prompt as a text field; don't start a new agentic loop.
                let _ = ui_tx
                    .send(UiUpdate::SteerResult(format!(
                        "[steer] /{} expanded prompt:\n{}",
                        cmd_name.trim_start_matches('/'),
                        prompt
                    )))
                    .await;
            }
            return;
        }

        // Plain text → show as a steering note in the model's output area.
        let _ = ui_tx.send(UiUpdate::SteerResult(trimmed.to_string())).await;
    }

    /// Manual `/compact` — force an LLM compaction of the older turns now,
    /// regardless of the token threshold. Mirrors `maybe_compact_history` but
    /// always runs (when there's enough history to compact) and reports the
    /// result as a text field in the model's output area.
    async fn manual_compact(&mut self, ui_tx: &mpsc::Sender<UiUpdate>) {
        const KEEP_RECENT: usize = 8;
        if self.history.len() <= KEEP_RECENT {
            let _ = ui_tx
                .send(UiUpdate::SteerResult(format!(
                    "[compact] nothing to compact — only {} turn(s) in context.",
                    self.history.len()
                )))
                .await;
            return;
        }

        let split = self.history.len() - KEEP_RECENT;
        let old: Vec<Message> = self.history[..split].to_vec();

        let (compact_provider, compact_model) = self.cheapest_model();
        let provider = match self.registry.get(&compact_provider) {
            Ok(p) => p,
            Err(_) => {
                let _ = ui_tx
                    .send(UiUpdate::SteerResult(format!(
                        "[compact] failed: provider '{compact_provider}' unavailable."
                    )))
                    .await;
                return;
            }
        };

        let ctx = compact_serialize(&old);
        let summary_req = StreamRequest {
            model: compact_model,
            messages: vec![Message::new_user(format!(
                "Produce a dense technical summary of this coding session fragment for an AI \
                coding assistant. Include: files created/modified (with key changes), commands \
                run and their outcomes, errors encountered and how they were resolved, decisions \
                made, and current task state. Be precise and comprehensive — this summary \
                replaces the original turns in context.\n\n{ctx}"
            ))],
            system_prompt: String::new(),
            max_tokens: 1500,
            tools: vec![],
            thinking: false,
        };

        let _ = ui_tx
            .send(UiUpdate::SteerResult(format!(
                "[compact] summarizing {split} older turns…"
            )))
            .await;

        let mut stream = match provider.stream(summary_req).await {
            Ok(s) => s,
            Err(e) => {
                let _ = ui_tx
                    .send(UiUpdate::SteerResult(format!("[compact] failed: {e}")))
                    .await;
                return;
            }
        };

        let mut summary = String::new();
        while let Some(chunk) = stream.recv().await {
            summary.push_str(&chunk.content);
            if chunk.done {
                break;
            }
        }

        if summary.trim().is_empty() {
            let _ = ui_tx
                .send(UiUpdate::SteerResult(
                    "[compact] failed: model returned an empty summary.".to_string(),
                ))
                .await;
            return;
        }

        let recent = self.history.split_off(split);
        self.history.clear();
        self.history.push(Message::new_user(format!(
            "[Marlin Context Summary — {split} turns condensed]\n{}",
            summary.trim()
        )));
        self.history.extend(recent);

        let new_tokens = estimate_tokens(&self.history, "");
        self.compact_guard_tokens = new_tokens;

        let _ = ui_tx
            .send(UiUpdate::SteerResult(format!(
                "[compact] done: {split} turns → 1 summary (~{new_tokens} tokens now)."
            )))
            .await;
    }

    async fn handle_slash_command(
        &mut self,
        raw: &str,
        ui_tx: &mpsc::Sender<UiUpdate>,
        action_rx: &mut mpsc::Receiver<Action>,
    ) -> Option<String> {
        let parts: Vec<&str> = raw.trim().splitn(2, ' ').collect();
        let cmd = parts[0].to_lowercase();
        let args: Vec<&str> = if parts.len() > 1 {
            parts[1].split_whitespace().collect()
        } else {
            vec![]
        };
        let rest = parts.get(1).copied().unwrap_or("").trim();

        macro_rules! sys {
            ($msg:expr) => {{
                ui_tx.send(UiUpdate::SystemMsg($msg.into())).await.ok();
            }};
        }
        macro_rules! err {
            ($msg:expr) => {{
                ui_tx.send(UiUpdate::ErrorMsg($msg.into())).await.ok();
            }};
        }
        macro_rules! save_cfg {
            () => {{
                if let Err(e) = self.cfg.save() {
                    err!(format!(
                        "Failed to save ~/.marlin/config.json: {e} — this change will be lost on restart"
                    ));
                }
            }};
        }

        match cmd.as_str() {
            "/help" => {
                sys!(help_text());
            }

            "/clear" => {
                self.history.clear();
                self.attachments.clear();
                sys!("Chat cleared.");
            }

            "/compact" => {
                self.manual_compact(ui_tx).await;
            }

            "/provider" | "/p" => {
                if args.is_empty() {
                    sys!(format!(
                        "Usage: /provider <name|list|new <name>>  — available: {}",
                        self.registry.names().join(", ")
                    ));
                    return None;
                }
                let subcmd = args[0].to_lowercase();
                match subcmd.as_str() {
                    "list" | "ls" => {
                        let names: Vec<String> = self.registry.names();
                        let user: Vec<marlin_providers::user_providers::UserProvider> =
                            marlin_providers::user_providers::load_all(&self.marlin_dir);
                        let mut lines: Vec<String> = names
                            .iter()
                            .map(|n| {
                                let marker = if *n == self.cfg.active_provider {
                                    " *"
                                } else {
                                    ""
                                };
                                format!("  {n}{marker}")
                            })
                            .collect();
                        for up in &user {
                            if !names.contains(&up.name) {
                                lines.push(format!("  {} (user, restart to activate)", up.name));
                            }
                        }
                        sys!(format!("Providers:\n{}", lines.join("\n")));
                    }

                    "new" | "create" => {
                        let name = args.get(1).copied().unwrap_or("my_provider");
                        match marlin_providers::user_providers::save_template(
                            &self.marlin_dir,
                            name,
                        ) {
                            Ok(path) => {
                                sys!(format!(
                                    "Provider template created:\n  {}\n\nEdit it and restart Marlin to activate.",
                                    path.display()
                                ));
                            }
                            Err(e) => err!(format!("Failed to create provider: {e}")),
                        }
                    }

                    name => {
                        if self.registry.get(name).is_err() {
                            err!(format!("Unknown provider: {name}"));
                            return None;
                        }
                        self.cfg.active_provider = name.to_string();
                        let model = self
                            .cfg
                            .providers
                            .get(name)
                            .and_then(|p| {
                                if p.model.is_empty() {
                                    None
                                } else {
                                    Some(p.model.clone())
                                }
                            })
                            .unwrap_or_default();
                        self.cfg.active_model = model.clone();
                        save_cfg!();
                        sys!(format!("Switched to provider: {name}  model: {model}"));
                        let _ = ui_tx.send(UiUpdate::StatusUpdate(self.status_info())).await;
                    }
                }
            }

            "/model" | "/m" => {
                if args.is_empty() {
                    if let Ok(p) = self.registry.get(&self.cfg.active_provider) {
                        sys!(format!("Available models: {}", p.models().join(", ")));
                    }
                    return None;
                }
                let model = args[0].to_string();
                self.cfg.active_model = model.clone();
                if let Some(pcfg) = self.cfg.providers.get_mut(&self.cfg.active_provider) {
                    pcfg.model = model.clone();
                }
                self.cfg
                    .remember_model(&self.cfg.active_provider.clone(), &model);
                save_cfg!();
                sys!(format!("Model set to: {model}"));
                let _ = ui_tx.send(UiUpdate::StatusUpdate(self.status_info())).await;
            }

            "/key" => {
                if args.is_empty() {
                    sys!("Usage: /key <provider> <api-key>");
                    return None;
                }
                if args.len() < 2 {
                    sys!("Usage: /key <provider> <api-key>");
                    return None;
                }
                let provider = args[0].to_lowercase();
                let key = args[1];
                match marlin_providers::user_providers::set_api_key(
                    &self.marlin_dir,
                    &provider,
                    key,
                ) {
                    Ok(true) => {}
                    Ok(false) => {
                        self.cfg.set_key(&provider, key);
                        save_cfg!();
                    }
                    Err(e) => {
                        err!(format!("Failed to save API key: {e}"));
                        return None;
                    }
                }
                self.registry = Registry::new(&self.cfg, Some(&self.marlin_dir));
                sys!(format!("API key saved for {provider}."));
            }

            "/endpoint" => {
                if args.len() < 2 {
                    sys!("Usage: /endpoint <provider> <url>");
                    return None;
                }
                let provider = args[0].to_lowercase();
                let new_endpoint = args[1];

                if let Err(e) = marlin_providers::user_providers::validate_endpoint(new_endpoint) {
                    err!(format!("Invalid endpoint: {e}"));
                    return None;
                }

                // Whether this provider already has a saved key, checked
                // *before* the switch — used below to warn if we're about to
                // start sending it to a new (non-local) host.
                let has_existing_key = self
                    .cfg
                    .providers
                    .get(&provider)
                    .map(|p| !p.api_key.is_empty())
                    .unwrap_or(false)
                    || marlin_providers::user_providers::load_all(&self.marlin_dir)
                        .iter()
                        .any(|p| p.name.eq_ignore_ascii_case(&provider) && !p.api_key.is_empty());

                match marlin_providers::user_providers::set_endpoint(
                    &self.marlin_dir,
                    &provider,
                    new_endpoint,
                ) {
                    Ok(true) => {}
                    Ok(false) => {
                        self.cfg.set_endpoint(&provider, new_endpoint);
                        save_cfg!();
                    }
                    Err(e) => {
                        err!(format!("Failed to save endpoint: {e}"));
                        return None;
                    }
                }
                self.registry = Registry::new(&self.cfg, Some(&self.marlin_dir));
                sys!(format!(
                    "Endpoint updated for {}: {}",
                    provider, new_endpoint
                ));

                if has_existing_key
                    && !marlin_providers::user_providers::endpoint_is_private_host(new_endpoint)
                {
                    sys!(format!(
                        "⚠ {provider} has a saved API key — it will now be sent to {new_endpoint} on every request. Only proceed if you trust this endpoint."
                    ));
                }
            }

            "/system" | "/sys" => {
                if rest.is_empty() {
                    if self.cfg.system_prompt.is_empty() {
                        sys!("No custom system prompt. Use /system <text> to set one.");
                    } else {
                        sys!(format!("Custom system prompt: {}", self.cfg.system_prompt));
                    }
                    return None;
                }
                self.cfg.system_prompt = rest.to_string();
                save_cfg!();
                sys!("System prompt updated.");
            }

            "/tokens" => {
                if args.is_empty() {
                    let system_prompt = self.effective_system_prompt();
                    let tools = all_tools(
                        &self.ast_mode,
                        &self.skill_tool_list(&self.active_goal),
                        &self.external_tools,
                        self.cfg.skill_subagents,
                        &self.mcp_tools,
                    );
                    let report = budget::compute(&system_prompt, &tools);
                    sys!(format!(
                        "Max output tokens: {}  (use /tokens <n> to change)\n\n\
                         Base prompt injection (target ~{}t, warning only):\n{}",
                        self.cfg.max_tokens,
                        budget::WARN_THRESHOLD,
                        report.format()
                    ));
                    return None;
                }
                if let Ok(n) = args[0].parse::<usize>() {
                    if n > 0 {
                        self.cfg.max_tokens = n;
                        save_cfg!();
                        sys!(format!("Max tokens: {n}"));
                    }
                }
            }

            "/budget" => {
                if args.is_empty() {
                    sys!(format!(
                        "Context budget: {} tokens (sidebar meter ceiling; use /budget <n> to change)",
                        self.token_budget
                    ));
                    return None;
                }
                if let Ok(n) = args[0].parse::<usize>() {
                    if n > 0 {
                        self.token_budget = n;
                        self.cfg.token_budget = n;
                        save_cfg!();
                        let _ = ui_tx
                            .send(UiUpdate::TokenUsage {
                                used: estimate_tokens(
                                    &self.history,
                                    &self.effective_system_prompt(),
                                ),
                                budget: self.token_budget,
                            })
                            .await;
                        sys!(format!("Context budget: {n} tokens"));
                    }
                } else {
                    err!("Usage: /budget <n>");
                }
            }

            "/providers" => {
                let names = self.registry.names();
                let mut lines: Vec<String> = Vec::new();
                for n in &names {
                    let mark = if n == &self.cfg.active_provider {
                        "▶ "
                    } else {
                        "  "
                    };
                    if let Ok(p) = self.registry.get(n) {
                        let models = p.models();
                        let preview: Vec<String> = models.iter().take(2).cloned().collect();
                        lines.push(format!("{mark}{n}  [{}...]", preview.join(", ")));
                    }
                }
                sys!(format!("Providers:\n{}", lines.join("\n")));
            }

            "/models" => match self.registry.get(&self.cfg.active_provider) {
                Ok(p) => sys!(format!(
                    "Models for {}:\n{}",
                    self.cfg.active_provider,
                    p.models().join("\n")
                )),
                Err(e) => err!(e.to_string()),
            },

            "/attach" | "/a" => {
                if args.is_empty() {
                    if self.attachments.is_empty() && self.image_attachments.is_empty() {
                        sys!("No files attached. Usage: /attach <file>");
                    } else {
                        let mut names: Vec<String> = self
                            .attachments
                            .iter()
                            .map(|(f, c)| format!("{} ({} lines)", f, c.lines().count()))
                            .collect();
                        names.extend(
                            self.image_attachments
                                .iter()
                                .map(|(f, _, _)| format!("{} (image)", f)),
                        );
                        sys!(format!("Attached:\n{}", names.join("\n")));
                    }
                    return None;
                }
                let path = self.resolve_path(args[0]);
                // Image files are attached as multimodal content, not inlined text.
                if let Some(mime) = image_mime(&path) {
                    match std::fs::read(&path) {
                        Ok(bytes) => {
                            let b64 = B64.encode(bytes);
                            self.image_attachments.retain(|(f, _, _)| f != &path);
                            self.image_attachments.push((path.clone(), mime, b64));
                            sys!(format!(
                                "Attached image: {} — send your next message to include it",
                                Path::new(&path)
                                    .file_name()
                                    .unwrap_or_default()
                                    .to_string_lossy()
                            ));
                        }
                        Err(e) => err!(format!("attach error: {e}")),
                    }
                    return None;
                }
                match std::fs::read_to_string(&path) {
                    Ok(content) => {
                        let lines = content.lines().count();
                        self.attachments.retain(|(f, _)| f != &path);
                        self.attachments.push((path.clone(), content));
                        sys!(format!(
                            "Attached: {} ({lines} lines) — send your next message to include it",
                            Path::new(&path)
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                        ));
                    }
                    Err(e) => err!(format!("attach error: {e}")),
                }
            }

            "/detach" => {
                if args.is_empty() {
                    self.attachments.clear();
                    self.image_attachments.clear();
                    sys!("All attachments cleared.");
                } else {
                    let name = args[0];
                    let before = self.attachments.len() + self.image_attachments.len();
                    self.attachments.retain(|(f, _)| {
                        Path::new(f).file_name().and_then(|n| n.to_str()) != Some(name) && f != name
                    });
                    self.image_attachments.retain(|(f, _, _)| {
                        Path::new(f).file_name().and_then(|n| n.to_str()) != Some(name) && f != name
                    });
                    if self.attachments.len() + self.image_attachments.len() < before {
                        sys!(format!("Detached: {name}"));
                    } else {
                        err!(format!("No attachment named {name:?}"));
                    }
                }
            }

            "/exec" => {
                if rest.is_empty() {
                    sys!("Usage: /exec <shell command>");
                    return None;
                }
                if self.cfg.sandbox_mode == SandboxMode::Off
                    && !self.cfg.skip_permissions
                    && !self.is_allowed(rest)
                {
                    let first = rest.split_whitespace().next().unwrap_or(rest);
                    err!(format!("Command not allowed: {rest:?}\nUse /allow {first} or /sandbox [permissive|docker|gvisor]."));
                    return None;
                }
                sys!(format!("Running: {rest}"));
                let cmd = rest.to_string();
                let wd = self.work_dir.clone();
                let out = tokio::task::spawn_blocking(move || {
                    std::process::Command::new("sh")
                        .arg("-c")
                        .arg(&cmd)
                        .current_dir(&wd)
                        .output()
                })
                .await;
                match out {
                    Ok(Ok(o)) => {
                        let text = format!(
                            "{}{}",
                            String::from_utf8_lossy(&o.stdout),
                            String::from_utf8_lossy(&o.stderr)
                        );
                        sys!(format!("[exec]\n{}", text.trim()));
                    }
                    _ => err!("exec failed"),
                }
            }

            "/allow" => {
                if args.is_empty() {
                    if self.allowed_commands.is_empty() {
                        sys!("No commands allowed. Use /allow <prefix> to permit.");
                    } else {
                        sys!(format!(
                            "Allowed prefixes: {}",
                            self.allowed_commands.join(", ")
                        ));
                    }
                    return None;
                }
                let pattern = rest.to_string();
                self.allowed_commands.push(pattern.clone());
                self.cfg.allowed_commands = self.allowed_commands.clone();
                save_cfg!();
                sys!(format!("Allowed: {pattern:?}"));
            }

            "/sandbox" => match args.first().copied() {
                Some("off") => {
                    self.cfg.sandbox_mode = SandboxMode::Off;
                    save_cfg!();
                    sys!("Sandbox off — shell commands require /allow.");
                }
                Some("on") | Some("permissive") => {
                    self.cfg.sandbox_mode = SandboxMode::Permissive;
                    save_cfg!();
                    sys!("Sandbox permissive — all commands allowed, running directly on host.");
                }
                Some("mxc") => {
                    if !executor::detect_mxc() {
                        err!(format!(
                            "MXC binary ({}) not found in PATH. \
                                Install from https://github.com/microsoft/mxc and retry.",
                            executor::mxc_binary_name()
                        ));
                    } else {
                        self.cfg.sandbox_mode = SandboxMode::Mxc;
                        save_cfg!();
                        sys!(format!(
                            "Sandbox mxc — AI commands run via Microsoft eXecution Containers \
                                ({}, network blocked, only workdir mounted rw).",
                            executor::mxc_binary_name()
                        ));
                    }
                }
                Some("docker") => {
                    if !executor::detect_docker() {
                        err!("docker CLI not found in PATH. Install Docker and retry.");
                    } else {
                        self.cfg.sandbox_mode = SandboxMode::Docker;
                        save_cfg!();
                        sys!("Sandbox docker — AI commands run inside a Docker container (network blocked, only workdir mounted rw).");
                    }
                }
                _ => {
                    let mode = self.cfg.sandbox_mode.label();
                    let mxc = if executor::detect_mxc() {
                        format!("available ({})", executor::mxc_binary_name())
                    } else {
                        format!("not found ({})", executor::mxc_binary_name())
                    };
                    let docker = if executor::detect_docker() {
                        "available (docker)"
                    } else {
                        "not found (docker)"
                    };
                    sys!(format!(
                        "Sandbox: {mode}  |  mxc: {mxc}  |  docker: {docker}\n\
                             /sandbox [off|permissive|mxc|docker]"
                    ));
                }
            },

            "/permissions" => match args.first().copied() {
                Some("skip") => {
                    self.cfg.skip_permissions = true;
                    save_cfg!();
                    sys!("Permissions skipped — all operations proceed without checks.");
                }
                Some("require") => {
                    self.cfg.skip_permissions = false;
                    save_cfg!();
                    sys!("Permissions required — file and command checks enabled.");
                }
                _ => {
                    let state = if self.cfg.skip_permissions {
                        "skip"
                    } else {
                        "require"
                    };
                    sys!(format!(
                        "Permissions: {state}  (use /permissions skip|require)"
                    ));
                }
            },

            "/verify" => {
                if rest.is_empty() {
                    match &self.cfg.verify_command {
                        Some(cmd) => {
                            sys!(format!("Verify command: {cmd}  (use /verify off to clear)"))
                        }
                        None => sys!("No verify command set.  Usage: /verify <shell-command>"),
                    }
                } else if rest == "off" || rest == "none" {
                    self.cfg.verify_command = None;
                    save_cfg!();
                    sys!("Verify command cleared.");
                } else {
                    self.cfg.verify_command = Some(rest.to_string());
                    save_cfg!();
                    sys!(format!("Verify command set: {rest}"));
                }
            }

            "/ast" => {
                let new_mode = match args.first().copied() {
                    Some("off") => Some(AstMode::Off),
                    Some("sexpr") => Some(AstMode::SExpr),
                    Some("harness") => Some(AstMode::Harness),
                    Some(other) => {
                        err!(format!(
                            "Unknown AST mode {other:?} — use: off, sexpr, harness"
                        ));
                        return None;
                    }
                    None => None,
                };
                if let Some(mode) = new_mode {
                    let label = mode.label();
                    self.ast_mode = mode.clone();
                    self.cfg.ast_mode = mode.clone();
                    save_cfg!();
                    let _ = ui_tx.send(UiUpdate::AstMode(mode)).await;
                    match label {
                        "off"     => sys!("AST mode off — file reads use raw text."),
                        "sexpr"   => sys!("AST mode: SEXPR — file reads deliver compact S-expression ASTs via ast-compiler."),
                        "harness" => sys!("AST mode: HARNESS — ast_skeleton / ast_get_node / ast_mutate tools now active."),
                        _         => {}
                    }
                } else {
                    sys!(format!(
                        "AST mode: {}  (use /ast off|sexpr|harness)",
                        self.ast_mode.label()
                    ));
                }
            }

            "/clean-env" => match args.first().copied() {
                Some("on") => {
                    self.cfg.clean_env = true;
                    save_cfg!();
                    sys!("Clean-env sandboxing ON — subprocesses get a stripped environment.");
                }
                Some("off") => {
                    self.cfg.clean_env = false;
                    save_cfg!();
                    sys!("Clean-env sandboxing OFF.");
                }
                _ => {
                    let state = if self.cfg.clean_env { "on" } else { "off" };
                    sys!(format!("Clean-env: {state}  (use /clean-env on|off)"));
                }
            },

            "/thinking" => match args.first().copied() {
                Some("on") => {
                    self.cfg.thinking = true;
                    save_cfg!();
                    sys!("Extended thinking ON — the model will reason before answering (Claude extended thinking / OpenAI reasoning models).");
                }
                Some("off") => {
                    self.cfg.thinking = false;
                    save_cfg!();
                    sys!("Extended thinking OFF.");
                }
                _ => {
                    let state = if self.cfg.thinking { "on" } else { "off" };
                    sys!(format!(
                        "Extended thinking: {state}  (use /thinking on|off)"
                    ));
                }
            },

            "/checkpoints" => match args.first().copied() {
                Some("on") => {
                    if !checkpoint::available(&self.work_dir) {
                        err!("This directory isn't a git repository — checkpoints require git.");
                    } else {
                        self.cfg.checkpoints = true;
                        save_cfg!();
                        sys!("Git checkpoints ON — a checkpoint commit is created before each turn; /undo rolls it back.");
                    }
                }
                Some("off") => {
                    self.cfg.checkpoints = false;
                    save_cfg!();
                    sys!("Git checkpoints OFF.");
                }
                _ => {
                    let state = if self.cfg.checkpoints { "on" } else { "off" };
                    let git = if checkpoint::available(&self.work_dir) {
                        "git repo detected"
                    } else {
                        "not a git repo"
                    };
                    sys!(format!(
                        "Checkpoints: {state}  ({git})  — use /checkpoints on|off"
                    ));
                }
            },

            "/undo" => {
                let work_dir = self.work_dir.clone();
                match tokio::task::spawn_blocking(move || checkpoint::undo(&work_dir)).await {
                    Ok(Ok(msg)) => sys!(format!("Undo complete: {msg}")),
                    Ok(Err(e)) => err!(format!("Undo failed: {e}")),
                    Err(e) => err!(format!("Undo failed: {e}")),
                }
            }

            "/color" => {
                if args.is_empty() {
                    let current = self
                        .cfg
                        .status_colors
                        .get(&self.work_dir)
                        .map(|c| format!("#{:02x}{:02x}{:02x}", c[0], c[1], c[2]))
                        .unwrap_or_else(|| "default".into());
                    sys!(format!(
                        "Status bar color for this directory: {current}\n\
                         Usage: /color <#rrggbb>  —  or /color off to clear.\n\
                         The color is saved per working directory so you can tell sessions apart."
                    ));
                    return None;
                }
                let arg = args[0];
                if arg == "off" || arg == "none" || arg == "default" {
                    self.cfg.status_colors.remove(&self.work_dir);
                    save_cfg!();
                    let _ = ui_tx.send(UiUpdate::StatusUpdate(self.status_info())).await;
                    sys!("Status bar color cleared (default).");
                    return None;
                }
                match marlin_config::parse_hex_color(arg) {
                    Some(rgb) => {
                        self.cfg.status_colors.insert(self.work_dir.clone(), rgb);
                        save_cfg!();
                        let _ = ui_tx.send(UiUpdate::StatusUpdate(self.status_info())).await;
                        sys!(format!(
                            "Status bar color set to #{:02x}{:02x}{:02x} for this directory.",
                            rgb[0], rgb[1], rgb[2]
                        ));
                    }
                    None => err!(format!(
                        "Invalid color: {arg:?} — use a hex value like #ff8800, or /color off to clear."
                    )),
                }
            }

            "/theme" => {
                match args.first().copied() {
                    Some("light") => {
                        self.cfg.theme = "light".into();
                        styles::set_light_theme(true);
                        save_cfg!();
                        sys!("Theme set to light.");
                    }
                    Some("dark") => {
                        self.cfg.theme = "dark".into();
                        styles::set_light_theme(false);
                        save_cfg!();
                        sys!("Theme set to dark.");
                    }
                    Some(name) => {
                        // Try to load a named theme from ~/.marlin/themes/<name>.toml
                        if let Some(palette) =
                            marlin_config::load_named_theme(&self.marlin_dir, name)
                        {
                            styles::load_palette(palette);
                            // Persist the named theme so it survives a restart.
                            self.cfg.theme = name.to_string();
                            save_cfg!();
                            sys!(format!("Theme '{}' applied (persisted).", name));
                        } else {
                            let named = marlin_config::list_themes(&self.marlin_dir);
                            if named.is_empty() {
                                err!(format!("Theme '{name}' not found. Add ~/.marlin/themes/{name}.toml to create it."));
                            } else {
                                let list: Vec<String> = named
                                    .iter()
                                    .map(|(n, d)| format!("  {n}  —  {d}"))
                                    .collect();
                                err!(format!(
                                    "Theme '{name}' not found. Available named themes:\n{}",
                                    list.join("\n")
                                ));
                            }
                        }
                    }
                    None => {
                        let named = marlin_config::list_themes(&self.marlin_dir);
                        let named_list = if named.is_empty() {
                            "  (none — add .toml files to ~/.marlin/themes/)".into()
                        } else {
                            named
                                .iter()
                                .map(|(n, d)| format!("  {n}  —  {d}"))
                                .collect::<Vec<_>>()
                                .join("\n")
                        };
                        sys!(format!(
                            "Theme: {}  (use /theme dark|light|<name>)\n\nNamed themes:\n{}",
                            self.cfg.theme, named_list
                        ));
                    }
                }
            }

            "/command" | "/commands" => {
                let subcmd = args.first().copied().unwrap_or("list");
                let subargs: Vec<&str> = args.get(1..).map(|a| a.to_vec()).unwrap_or_default();

                match subcmd {
                    "list" | "ls" => {
                        if self.user_commands.is_empty() {
                            sys!("No user commands. Add TOML files to ~/.marlin/commands/");
                        } else {
                            let lines: Vec<String> = self
                                .user_commands
                                .iter()
                                .map(|c| {
                                    let args_hint = if c.args.is_empty() {
                                        String::new()
                                    } else {
                                        format!(" {}", c.args)
                                    };
                                    format!("  /{}{:<20} — {}", c.name, args_hint, c.description)
                                })
                                .collect();
                            sys!(format!(
                                "User commands ({}):\n{}",
                                self.user_commands.len(),
                                lines.join("\n")
                            ));
                        }
                    }

                    "new" | "create" => {
                        let name = if subargs.is_empty() {
                            "my_command"
                        } else {
                            subargs[0]
                        };
                        let cmd = commands::UserCommand {
                            name: name.to_string(),
                            description: "Describe what this command does".into(),
                            args: "[optional-args]".into(),
                            run: commands::CommandRun {
                                kind: commands::CommandKind::Shell,
                                command: "echo {args}".into(),
                                template: String::new(),
                            },
                        };
                        match commands::save_command(&self.marlin_dir, &cmd) {
                            Ok(path) => {
                                sys!(format!(
                                    "Command template created:\n  {}\n\nEdit it, then /command reload to activate.",
                                    path.display()
                                ));
                                self.user_commands = commands::load_all(&self.marlin_dir);
                            }
                            Err(e) => err!(format!("Failed to create command: {e}")),
                        }
                    }

                    "reload" => {
                        self.user_commands = commands::load_all(&self.marlin_dir);
                        let defs: Vec<commands::UserCommandDef> = self
                            .user_commands
                            .iter()
                            .map(commands::UserCommandDef::from)
                            .collect();
                        let _ = ui_tx.send(UiUpdate::UserCommandsLoaded(defs)).await;
                        sys!(format!(
                            "Reloaded {} user command(s).",
                            self.user_commands.len()
                        ));
                    }

                    _ => {
                        sys!("Usage: /command [list|new <name>|reload]");
                    }
                }
            }

            "/tool" | "/tools" => {
                let subcmd = args.first().copied().unwrap_or("list");
                let subargs: Vec<&str> = args.get(1..).map(|a| a.to_vec()).unwrap_or_default();

                match subcmd {
                    "list" | "ls" => {
                        if self.external_tools.is_empty() {
                            sys!("No user tools. Add TOML files to ~/.marlin/tools/");
                        } else {
                            let lines: Vec<String> = self
                                .external_tools
                                .iter()
                                .map(|t| format!("  {:<24} — {}", t.name, t.description))
                                .collect();
                            sys!(format!(
                                "User tools ({}):\n{}",
                                self.external_tools.len(),
                                lines.join("\n")
                            ));
                        }
                    }

                    "new" | "create" => {
                        let name = if subargs.is_empty() {
                            "my_tool"
                        } else {
                            subargs[0]
                        };
                        match marlin_tools::external::save_template(&self.marlin_dir, name) {
                            Ok(path) => {
                                sys!(format!(
                                    "Tool template created:\n  {}\n\nEdit it, then /tool reload to activate.",
                                    path.display()
                                ));
                                self.external_tools =
                                    marlin_tools::external::load_all(&self.marlin_dir);
                            }
                            Err(e) => err!(format!("Failed to create tool: {e}")),
                        }
                    }

                    "reload" => {
                        self.external_tools = marlin_tools::external::load_all(&self.marlin_dir);
                        sys!(format!(
                            "Reloaded {} user tool(s).",
                            self.external_tools.len()
                        ));
                    }

                    _ => {
                        sys!("Usage: /tool [list|new <name>|reload]");
                    }
                }
            }

            "/mcp" | "/mcps" => {
                let subcmd = args.first().copied().unwrap_or("list");
                let subargs: Vec<&str> = args.get(1..).map(|a| a.to_vec()).unwrap_or_default();

                match subcmd {
                    "list" | "ls" => {
                        if self.mcp_server_configs.is_empty() {
                            sys!("No MCP servers configured. Add JSON files to ~/.marlin/mcp/, or /mcp new <name> <command> [args...].");
                        } else {
                            let lines: Vec<String> = self
                                .mcp_server_configs
                                .iter()
                                .map(|cfg| {
                                    if self.mcp_clients.contains_key(&cfg.name) {
                                        let tools: Vec<&str> = self
                                            .mcp_tools
                                            .iter()
                                            .filter(|(s, _)| s == &cfg.name)
                                            .map(|(_, t)| t.name.as_str())
                                            .collect();
                                        format!(
                                            "  {:<20} connected — {}",
                                            cfg.name,
                                            tools.join(", ")
                                        )
                                    } else {
                                        format!("  {:<20} not connected", cfg.name)
                                    }
                                })
                                .collect();
                            sys!(format!(
                                "MCP servers ({}):\n{}",
                                self.mcp_server_configs.len(),
                                lines.join("\n")
                            ));
                        }
                    }

                    "new" | "create" => {
                        if subargs.len() < 2 {
                            sys!("Usage: /mcp new <name> <command> [args...]");
                            return None;
                        }
                        let name = subargs[0];
                        let command = subargs[1];
                        let cmd_args: Vec<String> =
                            subargs[2..].iter().map(|s| s.to_string()).collect();
                        match mcp::save_template(&self.marlin_dir, name, command, cmd_args) {
                            Ok(path) => {
                                sys!(format!(
                                    "MCP server config created:\n  {}\n\n/mcp reload to connect.",
                                    path.display()
                                ));
                                self.mcp_server_configs = mcp::load_all(&self.marlin_dir);
                            }
                            Err(e) => err!(format!("Failed to create MCP server config: {e}")),
                        }
                    }

                    "reload" => {
                        self.mcp_server_configs = mcp::load_all(&self.marlin_dir);
                        self.connect_mcp_servers(ui_tx).await;
                        sys!(format!(
                            "Reconnected: {}/{} server(s), {} tool(s) total.",
                            self.mcp_clients.len(),
                            self.mcp_server_configs.len(),
                            self.mcp_tools.len()
                        ));
                    }

                    _ => sys!("Usage: /mcp [list|new <name> <command> [args...]|reload]"),
                }
            }

            "/index" => {
                if args.first().copied() == Some("status") {
                    if let Some(idx) = &self.code_index {
                        let symbol_count: usize = idx.files.iter().map(|f| f.symbols.len()).sum();
                        sys!(format!(
                            "Index: {} files, {} terms, {} symbols, built {}.",
                            idx.file_count,
                            idx.term_count,
                            symbol_count,
                            idx.built_at.format("%b %d %H:%M")
                        ));
                    } else {
                        sys!("No index built. Run /index to build one.");
                    }
                    return None;
                }
                let wd = self.work_dir.clone();
                sys!(format!("Building index for {wd}…"));
                let result = tokio::task::spawn_blocking(move || index::build(&wd, None)).await;
                match result {
                    Ok(Ok((idx, stats))) => {
                        // Re-seed the refresh baseline to match the freshly built
                        // index so the next periodic refresh isn't a no-op storm.
                        self.index_refresh = index::RefreshState::default();
                        let (_, next) = index::diff_against(&self.index_refresh, &self.work_dir);
                        self.index_refresh = next;
                        let _ = ui_tx.send(UiUpdate::IndexBuilt).await;
                        index::save(&self.marlin_dir, &idx);
                        sys!(format!("Index built: {} files, {} terms, {} symbols in {:?}. Use /search <query> or the AI will use it automatically.",
                            stats.files, stats.terms, stats.symbols, stats.elapsed));
                        self.code_index = Some(idx);
                    }
                    _ => err!("Index build failed"),
                }
            }

            "/search" => {
                if rest.is_empty() {
                    sys!("Usage: /search <query>");
                    return None;
                }
                let Some(idx) = &self.code_index else {
                    err!("No index. Run /index first.");
                    return None;
                };
                let results = index::search(idx, rest, 8);
                sys!(index::format_results(&results, rest));
            }

            "/revert" => {
                if args.is_empty() {
                    sys!("Usage: /revert <file> [n]  —  list snapshots or restore one");
                    return None;
                }
                let abs_path = self.resolve_path(args[0]);
                let snaps = snapshots::list(&self.marlin_dir, &self.work_dir, &abs_path);
                if snaps.is_empty() {
                    sys!(format!(
                        "No snapshots for {} — Marlin snapshots files before every AI edit.",
                        args[0]
                    ));
                    return None;
                }
                if args.len() < 2 {
                    let lines: Vec<String> = snaps
                        .iter()
                        .enumerate()
                        .map(|(i, s)| {
                            format!(
                                "  {:2}.  {}  [{}]  {}",
                                i + 1,
                                s.timestamp.format("%b %d %H:%M:%S"),
                                s.tool,
                                snapshots::human_size(s.size)
                            )
                        })
                        .collect();
                    sys!(format!(
                        "Snapshots for {} (newest first):\n{}\n\nUse /revert {} <n> to restore.",
                        args[0],
                        lines.join("\n"),
                        args[0]
                    ));
                    return None;
                }
                let n: usize = args[1].parse().unwrap_or(0);
                if n < 1 || n > snaps.len() {
                    err!(format!("Invalid snapshot number (1–{}).", snaps.len()));
                    return None;
                }
                let snap = &snaps[n - 1];
                match snapshots::restore(&self.marlin_dir, &self.work_dir, &abs_path, &snap.id) {
                    Ok(_) => sys!(format!(
                        "Restored {} → snapshot from {} ({}, {}).",
                        args[0],
                        snap.timestamp.format("%b %d %H:%M:%S"),
                        snap.tool,
                        snapshots::human_size(snap.size)
                    )),
                    Err(e) => err!(format!("Restore failed: {e}")),
                }
            }

            "/resume" => match history::list_sessions(&self.marlin_dir) {
                Ok(sessions) if !sessions.is_empty() => {
                    let s = &sessions[0];
                    self.history = s.messages.iter().map(from_session_message).collect();
                    let _ = ui_tx
                        .send(UiUpdate::HistoryLoaded(self.history_entries()))
                        .await;
                    sys!(format!("Resumed: {}", s.summary()));
                }
                _ => sys!("No saved sessions to resume."),
            },

            "/history" => {
                if args.first().copied() == Some("clear") {
                    match history::clear_sessions(&self.marlin_dir) {
                        Ok(_) => sys!("Session history cleared."),
                        Err(e) => err!(format!("Failed to clear sessions: {e}")),
                    }
                    return None;
                }
                let sessions = history::list_sessions(&self.marlin_dir).unwrap_or_default();
                if sessions.is_empty() {
                    sys!("No saved sessions.");
                    return None;
                }
                if let Some(n_str) = args.first() {
                    if let Ok(n) = n_str.parse::<usize>() {
                        if n >= 1 && n <= sessions.len() {
                            let s = &sessions[n - 1];
                            self.history = s.messages.iter().map(from_session_message).collect();
                            let _ = ui_tx
                                .send(UiUpdate::HistoryLoaded(self.history_entries()))
                                .await;
                            sys!(format!("Loaded: {}", s.summary()));
                            return None;
                        }
                        err!(format!("Invalid session number (1–{}).", sessions.len()));
                        return None;
                    }
                }
                let limit = sessions.len().min(20);
                let lines: Vec<String> = sessions[..limit]
                    .iter()
                    .enumerate()
                    .map(|(i, s)| format!("  {:2}.  {}  [{}]", i + 1, s.summary(), s.project))
                    .collect();
                sys!(format!("Saved sessions (newest first):\n{}\n\nUse /history <n> to load, /history clear to delete all.",
                    lines.join("\n")));
            }

            "/cat" => {
                if args.is_empty() {
                    sys!("Usage: /cat <file>");
                    return None;
                }
                let path = self.resolve_path(args[0]);
                match std::fs::read_to_string(&path) {
                    Ok(content) => sys!(format!("[{path}]\n{content}")),
                    Err(e) => err!(e.to_string()),
                }
            }

            // /open is an alias for /view — both just open the read-only file
            // pane; there's no separate file-browser behind /open (yet).
            "/view" | "/open" => {
                if args.is_empty() {
                    sys!(format!("Usage: {cmd} <file>"));
                    return None;
                }
                let path = self.resolve_path(args[0]);
                let result = std::fs::read_to_string(&path)
                    .map(|content| (path, content))
                    .map_err(|e| e.to_string());
                let _ = ui_tx.send(UiUpdate::OpenViewer(result)).await;
            }

            "/edit" => {
                if args.is_empty() {
                    sys!("Usage: /edit <file>");
                    return None;
                }
                let path = self.resolve_path(args[0]);
                // A missing file opens an empty buffer (Ctrl+S creates it) — anything
                // else (permission denied, etc.) is a real error to report.
                let result = match std::fs::read_to_string(&path) {
                    Ok(content) => Ok((path, content)),
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok((path, String::new())),
                    Err(e) => Err(e.to_string()),
                };
                let _ = ui_tx.send(UiUpdate::OpenEditor(result)).await;
            }

            "/diff-mode" => {
                if args.is_empty() {
                    sys!("Usage: /diff-mode <file>");
                    return None;
                }
                let path = self.resolve_path(args[0]);
                let snaps = snapshots::list(&self.marlin_dir, &self.work_dir, &path);
                let Some(latest) = snaps.first() else {
                    sys!(format!(
                        "No snapshots for {path} yet — snapshots are taken automatically \
                        on write_file/edit_file, so there's nothing to diff against until \
                        it's been edited at least once."
                    ));
                    return None;
                };

                let marlin_dir = self.marlin_dir.clone();
                let work_dir = self.work_dir.clone();
                let path_for_task = path.clone();
                let snap_id = latest.id.clone();

                // Snapshot/file reads plus the O(n·m) LCS diff are real work for
                // large files — route through the blocking pool like every other
                // tool's disk/CPU work (see spawn_tool), instead of running inline
                // on this async task and stalling it for the duration.
                let outcome = tokio::task::spawn_blocking(
                    move || -> Result<Vec<snapshots::DiffLine>, String> {
                        let old_content =
                            snapshots::read(&marlin_dir, &work_dir, &path_for_task, &snap_id)
                                .map_err(|e| format!("Failed to read snapshot: {e}"))?;
                        let new_content =
                            std::fs::read_to_string(&path_for_task).map_err(|e| e.to_string())?;
                        snapshots::diff_lines(&old_content, &new_content)
                            .ok_or_else(|| "File too large to diff.".to_string())
                    },
                )
                .await;

                match outcome {
                    Ok(Ok(diff)) => {
                        let _ = ui_tx.send(UiUpdate::OpenDiff { path, diff }).await;
                    }
                    Ok(Err(e)) => err!(e),
                    Err(e) => err!(format!("diff task failed: {e}")),
                }
            }

            "/ls" => {
                let dir = if args.is_empty() {
                    self.work_dir.clone()
                } else {
                    self.resolve_path(args[0])
                };
                match std::fs::read_dir(&dir) {
                    Err(e) => err!(e.to_string()),
                    Ok(entries) => {
                        let mut names: Vec<String> = entries
                            .filter_map(|e| e.ok())
                            .map(|e| {
                                let n = e.file_name().to_string_lossy().to_string();
                                if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                                    n + "/"
                                } else {
                                    n
                                }
                            })
                            .collect();
                        names.sort();
                        sys!(format!("[{dir}]\n{}", names.join("\n")));
                    }
                }
            }

            "/cd" => {
                if args.is_empty() {
                    sys!(format!("Current directory: {}", self.work_dir));
                    return None;
                }
                let new_dir = self.resolve_path(args[0]);
                match std::fs::metadata(&new_dir) {
                    Ok(m) if m.is_dir() => {
                        self.work_dir = new_dir.clone();
                        self.cfg.work_dir = new_dir.clone();
                        // Update the session so save_session writes with the
                        // correct project name and work_dir after a /cd.
                        if let Some(session) = &mut self.session {
                            let project_name = Path::new(&new_dir)
                                .file_name()
                                .unwrap_or_default()
                                .to_string_lossy()
                                .to_string();
                            session.project = project_name;
                            session.work_dir = new_dir.clone();
                        }
                        // Load per-project .marlonrc.toml if present
                        self.apply_project_config();
                        save_cfg!();
                        let _ = ui_tx.send(UiUpdate::StatusUpdate(self.status_info())).await;
                        sys!(format!("Directory: {new_dir}"));
                    }
                    _ => err!(format!("Not a directory: {}", args[0])),
                }
            }

            "/pwd" => {
                sys!(format!("Directory: {}", self.work_dir));
            }

            "/skill" | "/skills" => {
                let subcmd = args.first().copied().unwrap_or("list");
                let _subrest = args.get(1..).map(|a| a.join(" ")).unwrap_or_default();
                let subargs: Vec<&str> = args.get(1..).map(|a| a.to_vec()).unwrap_or_default();

                match subcmd {
                    "list" | "ls" => {
                        if self.skills.is_empty() {
                            sys!("No skills installed. Add .qmd files to ~/.marlin/skills/");
                        } else {
                            let lines: Vec<String> = self
                                .skills
                                .iter()
                                .map(|s| {
                                    let tag = if s.format == skills::SkillFormat::Toml {
                                        " [.toml, deprecated — /skill migrate]"
                                    } else {
                                        ""
                                    };
                                    format!("  {:20} — {}{tag}", s.name, s.description)
                                })
                                .collect();
                            sys!(format!(
                                "Skills ({}):\n{}",
                                self.skills.len(),
                                lines.join("\n")
                            ));
                        }
                    }

                    "run" | "r" => {
                        if subargs.is_empty() {
                            sys!("Usage: /skill run <name> [query]");
                            return None;
                        }
                        let skill_name = subargs[0];
                        let query = if subargs.len() > 1 {
                            subargs[1..].join(" ")
                        } else {
                            self.active_goal.clone()
                        };
                        if let Some(skill) =
                            self.skills.iter().find(|s| s.name == skill_name).cloned()
                        {
                            if self.cfg.skill_subagents {
                                sys!(format!(
                                    "Running skill '{}' with query: {query} (subagent)",
                                    skill.name
                                ));
                                let result = self
                                    .run_skill_as_subagent(&skill, &query, ui_tx, action_rx)
                                    .await;
                                if result.is_error {
                                    err!(format!("[skill: {skill_name}]\n{}", result.output));
                                } else {
                                    sys!(format!("[skill: {skill_name}]\n{}", result.output));
                                }
                            } else if skill.is_shell() {
                                match skills::executor::resolve_chunks(&skill, &query) {
                                    Err(e) => err!(format!("Skill error: {e}")),
                                    Ok(cmds) => {
                                        sys!(format!(
                                            "Running skill '{}' with query: {query}",
                                            skill.name
                                        ));
                                        let mut outputs = Vec::with_capacity(cmds.len());
                                        let mut failed = false;
                                        for cmd in cmds {
                                            let verdict = match self.preflight_shell(&cmd) {
                                                Err(result) => {
                                                    err!(format!(
                                                        "[skill: {skill_name}]\n{}",
                                                        result.output
                                                    ));
                                                    failed = true;
                                                    break;
                                                }
                                                Ok(v) => v,
                                            };
                                            let proceed = match verdict {
                                                preflight::Verdict::NeedApproval(reason) => {
                                                    self.await_approval(ui_tx, action_rx, reason)
                                                        .await
                                                }
                                                _ => true,
                                            };
                                            if !proceed {
                                                sys!(format!("[skill: {skill_name}] Denied."));
                                                failed = true;
                                                break;
                                            }
                                            let result = self.run_shell(cmd).await;
                                            if result.is_error {
                                                err!(format!(
                                                    "[skill: {skill_name}]\n{}",
                                                    result.output
                                                ));
                                                failed = true;
                                                break;
                                            }
                                            outputs.push(result.output);
                                        }
                                        if !failed {
                                            let prose = if skill.is_prompt() {
                                                skills::executor::expand_prompt(&skill, &query)
                                                    .unwrap_or_default()
                                            } else {
                                                String::new()
                                            };
                                            let body = outputs.join("\n\n");
                                            let out = if prose.is_empty() {
                                                body
                                            } else {
                                                format!("{prose}\n\n{body}")
                                            };
                                            sys!(format!("[skill: {skill_name}]\n{out}"));
                                        }
                                    }
                                }
                            } else if skill.is_prompt() {
                                match skills::executor::expand_prompt(&skill, &query) {
                                    Ok(prompt) => {
                                        sys!(format!("[skill: {}] Expanded prompt — copy and send to run:\n\n{prompt}", skill.name));
                                    }
                                    Err(e) => err!(format!("Skill error: {e}")),
                                }
                            } else {
                                err!(format!("skill '{skill_name}' has neither a shell chunk nor a prompt body"));
                            }
                        } else {
                            err!(format!("Unknown skill '{skill_name}'.  Use /skill list."));
                        }
                    }

                    "migrate" => match skills::migrate_all(&self.marlin_dir) {
                        Ok(0) => sys!("No .toml skills to migrate."),
                        Ok(n) => {
                            sys!(format!("Migrated {n} skill(s) to .qmd."));
                            let (loaded, diagnostics) = skills::load_all(&self.marlin_dir);
                            self.skills = loaded;
                            for d in &diagnostics {
                                err!(d.clone());
                            }
                        }
                        Err(e) => err!(format!("Migration failed: {e}")),
                    },

                    "new" | "create" => {
                        let name = if subargs.is_empty() {
                            "my_skill"
                        } else {
                            subargs[0]
                        };
                        let skill = skills::Skill {
                            name: name.to_string(),
                            description: "Describe what this skill does".into(),
                            triggers: vec!["keyword1".into(), "keyword2".into()],
                            body: String::new(),
                            chunks: vec![skills::Chunk {
                                lang: "sh".into(),
                                source: "echo {query}".into(),
                            }],
                            format: skills::SkillFormat::Qmd,
                        };
                        match skills::save_skill(&self.marlin_dir, &skill) {
                            Ok(path) => {
                                sys!(format!("Skill template created:\n  {}\n\nEdit the file to customise it, then /skill reload.", path.display()));
                                let (loaded, diagnostics) = skills::load_all(&self.marlin_dir);
                                self.skills = loaded;
                                for d in &diagnostics {
                                    err!(d.clone());
                                }
                            }
                            Err(e) => err!(format!("Failed to create skill: {e}")),
                        }
                    }

                    "suggest" => {
                        let suggestions_path = self.marlin_dir.join("skill_suggestions.md");
                        if suggestions_path.exists() {
                            match std::fs::read_to_string(&suggestions_path) {
                                Ok(content) => sys!(content),
                                Err(e) => err!(format!("Error reading suggestions: {e}")),
                            }
                        } else {
                            let context = self
                                .history
                                .iter()
                                .rev()
                                .find(|m| m.role == "user")
                                .map(|m| m.content.as_str())
                                .unwrap_or("");
                            let skill_defs: Vec<SkillDef> =
                                self.skills.iter().map(SkillDef::from).collect();
                            let hits = skills::suggest::match_skills(context, &skill_defs);
                            if hits.is_empty() {
                                sys!("No skill suggestions yet. Nightly analysis runs after 20h of activity (requires model_tiers config).");
                            } else {
                                let lines: Vec<String> = hits
                                    .iter()
                                    .map(|m| format!("  {:20} — {}", m.name, m.description))
                                    .collect();
                                sys!(format!("Suggested skills:\n{}", lines.join("\n")));
                            }
                        }
                    }

                    "reload" => {
                        let (loaded, diagnostics) = skills::load_all(&self.marlin_dir);
                        self.skills = loaded;
                        let skill_defs: Vec<SkillDef> =
                            self.skills.iter().map(SkillDef::from).collect();
                        let _ = ui_tx.send(UiUpdate::SkillsLoaded(skill_defs)).await;
                        for d in &diagnostics {
                            err!(d.clone());
                        }
                        sys!(format!("Reloaded {} skill(s).", self.skills.len()));
                    }

                    _ => {
                        sys!("Usage: /skill [list|run <name> [query]|new <name>|suggest|reload|migrate]");
                    }
                }
            }

            "/tiers" => match args.first().copied() {
                Some("on") => {
                    if self.cfg.model_tiers.is_none() {
                        self.cfg.model_tiers = Some(marlin_config::ModelTiers::default());
                    }
                    self.cfg.model_tiers.as_mut().unwrap().enabled = true;
                    save_cfg!();
                    sys!("Model tier routing enabled. Edit ~/.marlin/config.json (model_tiers) to configure.");
                }
                Some("off") => {
                    if let Some(t) = self.cfg.model_tiers.as_mut() {
                        t.enabled = false;
                    }
                    save_cfg!();
                    sys!("Model tier routing disabled — using active_provider/active_model.");
                }
                _ => {
                    let state = self.cfg.model_tiers.as_ref()
                            .map(|t| if t.enabled {
                                format!(
                                    "enabled\n  default (≤{}): {} / {}\n  complex (>{}): {} / {}\n  rater: {} / {}",
                                    t.default_max_difficulty,
                                    t.default.provider, t.default.model,
                                    t.default_max_difficulty,
                                    t.complex.provider, t.complex.model,
                                    t.rater.provider, t.rater.model,
                                )
                            } else {
                                "disabled".into()
                            })
                            .unwrap_or_else(|| "not configured (use /tiers on to enable)".into());
                    sys!(format!("Model tiers: {state}\n\nUse /tiers on|off"));
                }
            },

            "/subagents" => match args.first().copied() {
                Some("on") => {
                    self.cfg.skill_subagents = true;
                    save_cfg!();
                    sys!("Skill subagents ON — running a skill delegates to a nested agent loop.");
                }
                Some("off") => {
                    self.cfg.skill_subagents = false;
                    save_cfg!();
                    sys!("Skill subagents OFF — skills run inline again (old direct-execution behavior).");
                }
                _ => {
                    let state = if self.cfg.skill_subagents {
                        "on"
                    } else {
                        "off"
                    };
                    sys!(format!("Skill subagents: {state}  (use /subagents on|off)"));
                }
            },

            "/config" | "/settings" => {
                let _ = ui_tx
                    .send(UiUpdate::ConfigState {
                        state: self.config_state(),
                        open: true,
                    })
                    .await;
            }

            "/preflight" => {
                let scope = args.first().copied().unwrap_or("all");
                let mut lines = Vec::new();

                if scope == "startup" || scope == "all" {
                    let startup_lines = preflight::startup(
                        &self.cfg,
                        &self.marlin_dir,
                        &self.work_dir,
                        self.code_index.as_ref(),
                    );
                    lines.push(format!("startup: {} note(s)", startup_lines.len()));
                    lines.extend(startup_lines.into_iter().map(|l| format!("  {l}")));
                }

                if scope == "skills" || scope == "all" {
                    let (_loaded, diagnostics) = skills::load_all(&self.marlin_dir);
                    lines.push(format!("skills: {} note(s)", diagnostics.len()));
                    lines.extend(diagnostics.into_iter().map(|l| format!("  {l}")));
                }

                if lines.is_empty() {
                    sys!("preflight: no issues found.");
                } else {
                    sys!(format!("preflight [{scope}]:\n{}", lines.join("\n")));
                }
            }

            "/export" => {
                let format = args.first().copied().unwrap_or("html");
                let path = args.get(1).copied().unwrap_or("marlin_export");
                match format {
                    "html" => {
                        let html = self.export_html();
                        let out_path = if path.ends_with(".html") {
                            path.to_string()
                        } else {
                            format!("{path}.html")
                        };
                        match std::fs::write(&out_path, &html) {
                            Ok(_) => sys!(format!("Exported session to {out_path}")),
                            Err(e) => err!(format!("Failed to write {out_path}: {e}")),
                        }
                    }
                    "json" => {
                        let json = self.export_json();
                        let out_path = if path.ends_with(".json") {
                            path.to_string()
                        } else {
                            format!("{path}.json")
                        };
                        match std::fs::write(&out_path, &json) {
                            Ok(_) => sys!(format!("Exported session to {out_path}")),
                            Err(e) => err!(format!("Failed to write {out_path}: {e}")),
                        }
                    }
                    _ => sys!("Usage: /export [html|json] [filename]"),
                }
            }

            _ => {
                // Check user-defined commands before reporting unknown.
                let cmd_name = cmd.trim_start_matches('/');
                if let Some(ucmd) = self
                    .user_commands
                    .iter()
                    .find(|c| c.name == cmd_name)
                    .cloned()
                {
                    let args_str = rest.to_string();
                    match ucmd.run.kind {
                        commands::CommandKind::Shell => {
                            let command = ucmd
                                .run
                                .command
                                .replace("{args}", &executor::shell_quote(&args_str));
                            match self.preflight_shell(&command) {
                                Err(result) => err!(format!("[/{}]\n{}", ucmd.name, result.output)),
                                Ok(verdict) => {
                                    let proceed = match verdict {
                                        preflight::Verdict::NeedApproval(reason) => {
                                            self.await_approval(ui_tx, action_rx, reason).await
                                        }
                                        _ => true,
                                    };
                                    if proceed {
                                        sys!(format!("Running /{}: {command}", ucmd.name));
                                        let result = self.run_shell(command).await;
                                        if result.is_error {
                                            err!(format!("[/{}]\n{}", ucmd.name, result.output));
                                        } else {
                                            sys!(format!("[/{}]\n{}", ucmd.name, result.output));
                                        }
                                    } else {
                                        sys!(format!("[/{}] Denied.", ucmd.name));
                                    }
                                }
                            }
                        }
                        commands::CommandKind::Prompt => {
                            let prompt = ucmd.run.template.replace("{input}", &args_str);
                            sys!(format!(
                                "/{}: injecting prompt into conversation…",
                                ucmd.name
                            ));
                            return Some(prompt);
                        }
                    }
                } else {
                    err!(format!("Unknown command: {cmd}  (type /help for list)"));
                }
            }
        }
        None
    }

    /// Raw API key for the active provider — checks built-in/cfg-backed
    /// storage first, then falls back to a user-defined provider's toml.
    fn provider_api_key(&self) -> String {
        if let Some(pc) = self.cfg.providers.get(&self.cfg.active_provider) {
            if !pc.api_key.is_empty() {
                return pc.api_key.clone();
            }
        }
        marlin_providers::user_providers::load_all(&self.marlin_dir)
            .into_iter()
            .find(|p| p.name.eq_ignore_ascii_case(&self.cfg.active_provider))
            .map(|p| p.api_key)
            .unwrap_or_default()
    }

    fn config_state(&self) -> ConfigState {
        let mut models = self
            .registry
            .get(&self.cfg.active_provider)
            .map(|p| p.models())
            .unwrap_or_default();
        if let Some(pc) = self.cfg.providers.get(&self.cfg.active_provider) {
            for m in &pc.extra_models {
                if !models.contains(m) {
                    models.push(m.clone());
                }
            }
        }
        let named_themes: Vec<String> = marlin_config::list_themes(&self.marlin_dir)
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        ConfigState {
            provider: self.cfg.active_provider.clone(),
            providers: self.registry.names(),
            api_key: self.provider_api_key(),
            model: self.cfg.active_model.clone(),
            models,
            theme: self.cfg.theme.clone(),
            named_themes,
            sandbox_mode: self.cfg.sandbox_mode.label().into(),
            skip_permissions: self.cfg.skip_permissions,
            clean_env: self.cfg.clean_env,
            ast_mode: self.ast_mode.label().into(),
            skill_subagents: self.cfg.skill_subagents,
            max_tokens: self.cfg.max_tokens,
            tool_call_limit: self.cfg.tool_call_limit,
        }
    }

    /// Apply one setting change from the /config menu, then echo the refreshed
    /// snapshot so the menu stays in sync (e.g. the model list after a provider
    /// switch, or the unchanged sandbox mode when MXC isn't installed).
    async fn apply_config_set(&mut self, key: &str, value: &str, ui_tx: &mpsc::Sender<UiUpdate>) {
        match key {
            "provider" => {
                if self.registry.get(value).is_err() {
                    let _ = ui_tx
                        .send(UiUpdate::ErrorMsg(format!("Unknown provider: {value}")))
                        .await;
                } else {
                    self.cfg.active_provider = value.to_string();
                    let model = self
                        .cfg
                        .providers
                        .get(value)
                        .and_then(|p| {
                            if p.model.is_empty() {
                                None
                            } else {
                                Some(p.model.clone())
                            }
                        })
                        .unwrap_or_default();
                    self.cfg.active_model = model.clone();
                    let _ = self.cfg.save();
                    let _ = ui_tx.send(UiUpdate::StatusUpdate(self.status_info())).await;
                }
            }
            "new_provider" => {
                // Packed by the config menu's name/URL/model/key wizard as
                // newline-separated fields (see ConfigMenu::on_key_new_provider).
                let mut parts = value.splitn(4, '\n');
                let name = parts.next().unwrap_or("").trim();
                let endpoint = parts.next().unwrap_or("").trim();
                let model = parts.next().unwrap_or("").trim();
                let api_key = parts.next().unwrap_or("").trim();
                if name.is_empty() {
                    // Submitted with nothing typed — silently ignore.
                } else if self.registry.get(name).is_ok() {
                    let _ = ui_tx
                        .send(UiUpdate::ErrorMsg(format!(
                            "Provider already exists: {name}"
                        )))
                        .await;
                } else {
                    match marlin_providers::user_providers::save_new(
                        &self.marlin_dir,
                        name,
                        endpoint,
                        model,
                        api_key,
                    ) {
                        Ok(_) => {
                            self.registry = Registry::new(&self.cfg, Some(&self.marlin_dir));
                            self.cfg.active_provider = name.to_string();
                            self.cfg.active_model = self
                                .registry
                                .get(name)
                                .map(|p| p.models().first().cloned().unwrap_or_default())
                                .unwrap_or_default();
                            let _ = self.cfg.save();
                            let _ = ui_tx.send(UiUpdate::StatusUpdate(self.status_info())).await;
                            let _ = ui_tx
                                .send(UiUpdate::SystemMsg(format!(
                                    "Provider '{name}' created and selected."
                                )))
                                .await;
                        }
                        Err(e) => {
                            let _ = ui_tx
                                .send(UiUpdate::ErrorMsg(format!(
                                    "Failed to create provider: {e}"
                                )))
                                .await;
                        }
                    }
                }
            }
            "model" => {
                self.cfg.active_model = value.to_string();
                if let Some(pcfg) = self.cfg.providers.get_mut(&self.cfg.active_provider) {
                    pcfg.model = value.to_string();
                }
                self.cfg
                    .remember_model(&self.cfg.active_provider.clone(), value);
                let _ = self.cfg.save();
                let _ = ui_tx.send(UiUpdate::StatusUpdate(self.status_info())).await;
            }
            "api_key" => {
                let provider = self.cfg.active_provider.clone();
                match marlin_providers::user_providers::set_api_key(
                    &self.marlin_dir,
                    &provider,
                    value,
                ) {
                    Ok(true) => {}
                    Ok(false) => {
                        self.cfg.set_key(&provider, value);
                        let _ = self.cfg.save();
                    }
                    Err(e) => {
                        let _ = ui_tx
                            .send(UiUpdate::ErrorMsg(format!("Failed to save API key: {e}")))
                            .await;
                    }
                }
                self.registry = Registry::new(&self.cfg, Some(&self.marlin_dir));
            }
            "theme" => {
                if value == "dark" || value == "light" {
                    self.cfg.theme = value.to_string();
                    styles::set_light_theme(value == "light");
                    let _ = self.cfg.save();
                } else if let Some(palette) =
                    marlin_config::load_named_theme(&self.marlin_dir, value)
                {
                    // Named theme — apply and persist it.
                    styles::load_palette(palette);
                    self.cfg.theme = value.to_string();
                    let _ = self.cfg.save();
                }
            }
            "sandbox" => {
                let mode = match value {
                    "off" => Some(SandboxMode::Off),
                    "permissive" => Some(SandboxMode::Permissive),
                    "mxc" if executor::detect_mxc() => Some(SandboxMode::Mxc),
                    "mxc" => {
                        let _ = ui_tx
                            .send(UiUpdate::ErrorMsg(format!(
                                "MXC binary ({}) not found in PATH. \
                            Install from https://github.com/microsoft/mxc and retry.",
                                executor::mxc_binary_name()
                            )))
                            .await;
                        None
                    }
                    "docker" if executor::detect_docker() => Some(SandboxMode::Docker),
                    "docker" => {
                        let _ = ui_tx
                            .send(UiUpdate::ErrorMsg(
                                "docker CLI not found in PATH. Install Docker and retry.".into(),
                            ))
                            .await;
                        None
                    }
                    _ => None,
                };
                if let Some(mode) = mode {
                    self.cfg.sandbox_mode = mode;
                    let _ = self.cfg.save();
                }
            }
            "permissions" => {
                self.cfg.skip_permissions = value == "skip";
                let _ = self.cfg.save();
            }
            "clean_env" => {
                self.cfg.clean_env = value == "on";
                let _ = self.cfg.save();
            }
            "ast" => {
                let mode = match value {
                    "off" => Some(AstMode::Off),
                    "sexpr" => Some(AstMode::SExpr),
                    "harness" => Some(AstMode::Harness),
                    _ => None,
                };
                if let Some(mode) = mode {
                    self.ast_mode = mode.clone();
                    self.cfg.ast_mode = mode.clone();
                    let _ = self.cfg.save();
                    let _ = ui_tx.send(UiUpdate::AstMode(mode)).await;
                }
            }
            "subagents" => {
                self.cfg.skill_subagents = value == "on";
                let _ = self.cfg.save();
            }
            "max_tokens" => {
                if let Ok(n) = value.parse::<usize>() {
                    if n > 0 {
                        self.cfg.max_tokens = n;
                        let _ = self.cfg.save();
                    }
                }
            }
            "tool_call_limit" => {
                if let Ok(n) = value.parse::<usize>() {
                    if n > 0 {
                        self.cfg.tool_call_limit = n;
                        let _ = self.cfg.save();
                    }
                }
            }
            _ => {}
        }
        let _ = ui_tx
            .send(UiUpdate::ConfigState {
                state: self.config_state(),
                open: false,
            })
            .await;
    }

    /// Ctrl+S in the /edit pane. Goes through the exact same funnel a
    /// write_file tool call would: preflight path-escape check (approval
    /// modal if it needs one — `--dangerously-skip-permissions` still
    /// short-circuits it via `preflight::check`), then the real executor
    /// (snapshotting, etc.) via `spawn_tool`, not a raw `std::fs::write`.
    async fn save_editor_file(
        &mut self,
        path: String,
        content: String,
        ui_tx: &mpsc::Sender<UiUpdate>,
        action_rx: &mut mpsc::Receiver<Action>,
    ) {
        let resolved = executor::resolve_path(&path, &self.work_dir);
        let inv = preflight::Invocation::paths("write_file", vec![resolved.clone()]);

        let approved = match preflight::check(&inv, &self.cfg, &self.allowed_commands) {
            preflight::Verdict::Allow => true,
            preflight::Verdict::NeedApproval(reason) => {
                self.await_approval(ui_tx, action_rx, reason).await
            }
            preflight::Verdict::Deny(reason) => {
                let _ = ui_tx
                    .send(UiUpdate::ErrorMsg(format!("Save denied: {reason}")))
                    .await;
                false
            }
        };
        if !approved {
            let _ = ui_tx
                .send(UiUpdate::SystemMsg(format!("Save cancelled: {resolved}")))
                .await;
            return;
        }

        let input = serde_json::json!({ "path": path, "content": content }).to_string();
        let call = ToolCall {
            id: "editor-save".into(),
            name: "write_file".into(),
            input,
        };
        let result = self
            .spawn_tool(&call, ui_tx)
            .await
            .unwrap_or_else(|e| executor::ToolResult {
                output: e.to_string(),
                is_error: true,
            });

        if result.is_error {
            let _ = ui_tx.send(UiUpdate::ErrorMsg(result.output)).await;
        } else {
            if let Some(idx) = &mut self.code_index {
                index::update_file(idx, &resolved);
            }
            let _ = ui_tx
                .send(UiUpdate::SystemMsg(format!("Saved {resolved}")))
                .await;
            let _ = ui_tx.send(UiUpdate::EditorSaved { path: resolved }).await;
        }
    }

    fn is_allowed(&self, cmd: &str) -> bool {
        policy::is_command_allowed(cmd, &self.allowed_commands)
    }

    /// Delegates to `executor::resolve_path` (the same resolver every tool call
    /// and snapshot uses) rather than re-deriving the join itself — a separate
    /// `format!("{work_dir}/{p}")` here previously diverged from it whenever
    /// `work_dir` had a trailing slash (e.g. after `/cd src/`), producing a
    /// doubled-slash path that hashed to a different snapshot directory than
    /// the one snapshots were actually stored under.
    fn resolve_path(&self, p: &str) -> String {
        executor::resolve_path(p, &self.work_dir)
    }

    /// Consume the queued text + image attachments for the next message into a
    /// `Message`, clearing both queues.
    fn take_attachments(&mut self, text: &str) -> Message {
        let content = self.build_message_content(text);
        let images: Vec<(String, String)> = std::mem::take(&mut self.image_attachments)
            .into_iter()
            .map(|(_, mime, b64)| (mime, b64))
            .collect();
        self.attachments.clear();
        Message {
            role: "user".into(),
            content,
            tool_calls: vec![],
            tool_use_id: String::new(),
            tool_call_id: String::new(),
            images,
            is_error: false,
        }
    }

    /// Send a desktop notification (best-effort, non-blocking).
    fn send_notification(&self, title: &str, body: &str) {
        let title = title.to_string();
        let body = body.chars().take(120).collect::<String>();
        std::thread::spawn(move || {
            // Try terminal-notifier (macOS) first, then notify-send (Linux)
            let _ = std::process::Command::new("terminal-notifier")
                .args(["-title", &title, "-message", &body, "-group", "marlin"])
                .output();
            let _ = std::process::Command::new("notify-send")
                .args([&title, &body, "--app-name=marlin"])
                .output();
        });
    }

    /// Export the current conversation as a self-contained HTML file.
    fn export_html(&self) -> String {
        let mut html = String::from(
            "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">\
            <title>Marlin Session</title>\
            <style>body{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;padding:2em;\
            background:#1a1b26;color:#c0caf5;}\
            .user{color:#7aa2f7;margin:1em 0;}\
            .assistant{color:#9ece6a;margin:1em 0;}\
            .tool{color:#e0af68;margin:0.5em 0;font-size:0.9em;}\
            .error{color:#f7768e;margin:0.5em 0;}\
            .system{color:#565f89;margin:0.5em 0;font-size:0.85em;}\
            pre{background:#24283b;padding:0.8em;border-radius:6px;overflow-x:auto;}\
            code{font-family:monospace;}\
            .summary{color:#9ece6a;opacity:0.7;margin:1em 0;font-style:italic;}\
            </style></head><body>\n<h1>Marlin Session</h1>\n",
        );
        for msg in &self.history {
            let role = &msg.role;
            let content = html_escape(&msg.content);
            match role.as_str() {
                "user" => html.push_str(&format!(
                    "<div class=\"user\"><strong>You</strong><pre>{content}</pre></div>\n"
                )),
                "assistant" => {
                    if !msg.tool_calls.is_empty() {
                        for tc in &msg.tool_calls {
                            html.push_str(&format!(
                                "<div class=\"tool\"><strong>🔧 {}</strong><pre>{}</pre></div>\n",
                                html_escape(&tc.name),
                                html_escape(&tc.input)
                            ));
                        }
                    }
                    if !content.is_empty() {
                        html.push_str(&format!("<div class=\"assistant\"><strong>Marlin</strong><pre>{content}</pre></div>\n"));
                    }
                }
                "tool" => {
                    let class = if msg.is_error { "error" } else { "tool" };
                    html.push_str(&format!("<div class=\"{class}\"><strong>↳ result</strong><pre>{content}</pre></div>\n"));
                }
                _ => {}
            }
        }
        html.push_str("</body></html>\n");
        html
    }

    /// Export the current conversation as JSON (same format as session files).
    fn export_json(&self) -> String {
        let session_msgs: Vec<history::SessionMessage> =
            self.history.iter().map(to_session_message).collect();
        serde_json::to_string_pretty(&session_msgs).unwrap_or_else(|_| "[]".into())
    }
}

/// Escape HTML special characters in a string.
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Return a short hint for common tool errors, so the user gets a quick
/// suggestion without waiting for the model to see and respond to the error.
fn error_hint(tool_name: &str, output: &str) -> String {
    let out = output.to_lowercase();
    match tool_name {
        "run_command" => {
            if out.contains("command not found") {
                "The command wasn't found. Check the spelling or install it first.".into()
            } else if out.contains("permission denied") {
                "Permission denied. Try running with sudo or check file permissions.".into()
            } else if out.contains("no such file") {
                "File or directory not found. Check the path and try again.".into()
            } else if out.contains("not permitted") {
                "This command needs approval. Use /allow <command> to permit it.".into()
            } else {
                String::new()
            }
        }
        "read_file" => {
            if out.contains("no such file") {
                "File not found. Check the path — it may need to be created first.".into()
            } else if out.contains("permission denied") {
                "Can't read this file due to permissions.".into()
            } else {
                String::new()
            }
        }
        "write_file" | "edit_file" => {
            if out.contains("permission denied") {
                "Can't write to this location. Check directory permissions.".into()
            } else if out.contains("no such file") && tool_name == "edit_file" {
                "File doesn't exist yet. Use write_file to create it first.".into()
            } else if out.contains("old_string not found") {
                "The text to replace wasn't found. The file may have changed — re-read it and try again.".into()
            } else {
                String::new()
            }
        }
        "multi_edit" => {
            if out.contains("old_string not found") {
                "One of the edits' old_string wasn't found (possibly after earlier edits in the batch already ran). Re-read the file and adjust the edits.".into()
            } else if out.contains("edits") && (out.contains("empty") || out.contains("required")) {
                "multi_edit needs a non-empty 'edits' array of {old_string, new_string} pairs."
                    .into()
            } else {
                String::new()
            }
        }
        "search_codebase" => {
            if out.contains("index not built") {
                "No search index yet. Run /index to build one for this project.".into()
            } else {
                String::new()
            }
        }
        _ => String::new(),
    }
}

fn extract_file_path(input_json: &str, work_dir: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(input_json).ok()?;
    let p = v["path"].as_str()?;
    if p.is_empty() {
        return None;
    }
    if Path::new(p).is_absolute() {
        Some(p.to_string())
    } else {
        // Use resolve_path so trailing-slash work_dirs (e.g. after /cd src/)
        // don't produce doubled slashes that break the loop-guard file-hash
        // check (which also uses resolve_path).
        Some(marlin_tools::executor::resolve_path(p, work_dir))
    }
}

fn extract_cmd_str(input_json: &str) -> String {
    serde_json::from_str::<serde_json::Value>(input_json)
        .ok()
        .and_then(|v| v["command"].as_str().map(String::from))
        .unwrap_or_default()
}

fn extract_path_field(input_json: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(input_json)
        .ok()
        .and_then(|v| v["path"].as_str().map(String::from))
}

fn extract_summary_field(input_json: &str) -> String {
    serde_json::from_str::<serde_json::Value>(input_json)
        .ok()
        .and_then(|v| v["summary"].as_str().map(String::from))
        .unwrap_or_default()
}

/// Serialize a slice of messages into a compact text block for the compaction LLM call.
/// Handles tool-call messages (content is often empty) by including the call list.
fn compact_serialize(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        match m.role.as_str() {
            "assistant" if !m.tool_calls.is_empty() => {
                if !m.content.is_empty() {
                    let snip: String = m.content.chars().take(300).collect();
                    out.push_str(&format!("[assistant]: {snip}\n"));
                }
                for tc in &m.tool_calls {
                    let input_snip: String = tc.input.chars().take(200).collect();
                    out.push_str(&format!("  [tool_call] {}({})\n", tc.name, input_snip));
                }
                out.push('\n');
            }
            "tool" => {
                let snip: String = m.content.chars().take(400).collect();
                out.push_str(&format!("  [tool_result]: {snip}\n\n"));
            }
            _ => {
                let snip: String = m.content.chars().take(600).collect();
                out.push_str(&format!("[{}]: {snip}\n\n", m.role));
            }
        }
    }
    out
}

/// Catches the common failure mode where the model announces an action
/// ("Let me add docstrings to those functions.") but ends its turn with no
/// tool calls at all, so the announced work never happens and the turn is
/// reported as complete anyway. Heuristic: the trailing sentence of a
/// tool-call-free response reads as a promise of imminent action rather than
/// a completed/blocked report. Deliberately conservative (checked only
/// against the last sentence) to avoid flagging retrospective phrasing like
/// "I let the test run" or a plan summary that happens to contain "I'll".
/// Number of times to retry a provider call that fails with a transient
/// network error (connection refused, DNS, timeout, mid-stream disconnect —
/// anything that isn't an API-level error). The message history is intact so
/// a retry is safe; the request itself is deterministic given the history.
const NETWORK_RETRIES: usize = 3;
/// Backoff (ms) between network retries, doubling each attempt: 800, 1600, 3200.
const NETWORK_RETRY_BASE_MS: u64 = 800;

/// True if a provider error is a transient network failure worth retrying
/// rather than an API/auth/HTTP-level error. reqwest wraps connection-level
/// failures in `error sending request for url (...)`, so we match on that
/// rather than on the inner `reqwest::Error`.
fn is_transient_network_error(e: &anyhow::Error) -> bool {
    let s = e.to_string().to_lowercase();
    s.contains("error sending request")
        || s.contains("connect timeout")
        || s.contains("connection refused")
        || s.contains("connection reset")
        || s.contains("connection closed")
        || s.contains("dns")
        || s.contains("timed out")
        || s.contains("tls")
        || s.contains("unexpected eof")
        || s.contains("stream ended")
}

fn looks_like_unfinished_stall(text: &str) -> bool {
    const STALL_PHRASES: &[&str] = &[
        "let me ",
        "let's ",
        "i'll ",
        "i will ",
        "i'm going to ",
        "i am going to ",
        "now i'll ",
        "now i will ",
        "next i'll ",
        "next, i'll ",
        "next i will ",
        "going to now ",
        "i'll now ",
        "i will now ",
    ];

    // Scan all sentences, not just the last one. The model may say
    // "I'll now read the file. The file contains 200 lines of..."
    // where the stall signal is in the penultimate sentence.
    for sentence in text.split(['.', '!', '?', '\n']) {
        let s = sentence.trim().to_lowercase();
        if s.is_empty() {
            continue;
        }
        if STALL_PHRASES.iter().any(|p| s.starts_with(p)) {
            return true;
        }
    }
    false
}

/// Tools with no side effects and no ordering dependency on each other or on
/// prior tool calls in the same turn — safe to run concurrently when the
/// model requests several in one turn. Everything else (writes, commands,
/// skills, AST mutation, external tools) stays strictly sequential.
fn is_parallel_safe(name: &str) -> bool {
    matches!(
        name,
        "read_file"
            | "list_directory"
            | "search_codebase"
            | "search_symbols"
            | "grep"
            | "glob"
            | "ast_skeleton"
            | "ast_get_node"
    )
}

/// Assigns a shared group id (the run's starting index) to every call in a
/// run of 2+ consecutive parallel-safe calls, for the sidebar's grouped
/// display; solo calls (parallel-safe or not) get `None`. Name-based only —
/// doesn't know about `denied`, so it's an approximation of the batching
/// `execute_tools` actually performs (which also excludes denied calls from
/// concurrent execution), good enough for a cosmetic "these ran together" hint.
fn parallel_group_ids(calls: &[ToolCall]) -> Vec<Option<usize>> {
    let mut ids = vec![None; calls.len()];
    let mut i = 0;
    while i < calls.len() {
        if is_parallel_safe(&calls[i].name) {
            let start = i;
            while i < calls.len() && is_parallel_safe(&calls[i].name) {
                i += 1;
            }
            if i - start > 1 {
                for id in ids[start..i].iter_mut() {
                    *id = Some(start);
                }
            }
        } else {
            i += 1;
        }
    }
    ids
}

fn tool_short_desc(name: &str, input_json: &str) -> String {
    let v = serde_json::from_str::<serde_json::Value>(input_json).unwrap_or_default();
    match name {
        "read_file" | "write_file" | "edit_file" | "multi_edit" | "notebook_edit" => {
            let path = v["path"].as_str().unwrap_or("?");
            let basename = Path::new(path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| path.to_string());
            format!("{name}: {basename}")
        }
        "run_command" => {
            let cmd = v["command"].as_str().unwrap_or("?");
            let short = shell_aware_truncate(cmd, 3);
            format!("run: {short}")
        }
        "search_codebase" => {
            let q = v["query"].as_str().unwrap_or("?");
            format!("search: {q}")
        }
        "grep" => {
            let p = v["pattern"].as_str().unwrap_or("?");
            format!("grep: {p}")
        }
        "glob" => {
            let p = v["pattern"].as_str().unwrap_or("?");
            format!("glob: {p}")
        }
        "ast_skeleton" => {
            let f = v["file"].as_str().unwrap_or("?");
            format!(
                "ast_skeleton: {}",
                Path::new(f)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| f.into())
            )
        }
        "ast_get_node" => {
            let id = v["node_id"].as_str().unwrap_or("?");
            format!("ast_get_node: {id}")
        }
        "ast_mutate" => {
            let op = v["operation"].as_str().unwrap_or("?");
            let id = v["node_id"].as_str().unwrap_or("?");
            format!("ast_mutate: {op} @ {id}")
        }
        _ => name.to_string(),
    }
}

/// Truncate a shell command to the first `n` "words", respecting single-quoted
/// and double-quoted strings so `sh -c 'some long command'` becomes
/// `sh -c 'some long command'` (3 tokens) rather than `sh -c 'some` (3
/// whitespace-split tokens that break the quote).
fn shell_aware_truncate(cmd: &str, n: usize) -> String {
    let mut tokens: Vec<&str> = Vec::new();
    let mut start = 0;
    let mut in_single = false;
    let mut in_double = false;
    let bytes = cmd.as_bytes();

    for (i, &b) in bytes.iter().enumerate() {
        if in_single {
            if b == b'\'' {
                in_single = false;
            }
        } else if in_double {
            if b == b'"' {
                in_double = false;
            }
        } else if b == b'\'' {
            in_single = true;
        } else if b == b'"' {
            in_double = true;
        } else if b == b' ' || b == b'\t' {
            if start < i {
                tokens.push(&cmd[start..i]);
                if tokens.len() >= n {
                    return tokens.join(" ");
                }
            }
            start = i + 1;
        }
    }
    if start < cmd.len() {
        tokens.push(&cmd[start..]);
    }
    tokens.join(" ")
}

fn help_text() -> String {
    let cmds = [
        ("/help", "show this help"),
        ("/config", "open the interactive settings menu"),
        ("/clear", "clear chat history and attachments"),
        ("/compact", "manually compact older turns into a summary now"),
        ("/provider <name>", "switch provider (claude/ollama/groq/fireworks/moonshot/custom)"),
        ("/model <name>", "switch model"),
        ("/providers", "list all providers and models"),
        ("/models", "list models for current provider"),
        ("/key <provider> <key>", "set API key"),
        ("/endpoint <provider> <url>", "set API endpoint for a provider"),
        ("/system <prompt>", "set additional system prompt"),
        ("/tokens [n]", "no args: show prompt injection budget breakdown; <n>: set max output tokens"),
        ("/budget [n]", "no args: show context budget; <n>: set sidebar meter ceiling (persists)"),
        ("/attach <file>", "attach a file to your next message"),
        ("/detach [file]", "remove attachment(s)"),
        ("/exec <cmd>", "run a shell command (must be /allow-ed first, or /sandbox on)"),
        ("/allow <prefix>", "allow a shell command prefix (e.g. /allow npm)"),
        ("/sandbox [off|permissive|mxc|docker]", "command isolation: off=require /allow, permissive=allow all, mxc=MS eXecution Containers, docker=Docker container"),
        ("/permissions [skip|require]", "skip or require permission checks (persists)"),
        ("/verify [cmd|off]", "set shell command to run after every file edit (Write-Test-Fix)"),
        ("/ast [off|sexpr|harness]", "AST context mode: off=raw, sexpr=S-expr reads, harness=JSON surgery (persists)"),
        ("/clean-env [on|off]", "strip subprocess environment for isolation (persists)"),
        ("/thinking [on|off]", "request extended thinking / chain-of-thought reasoning (persists)"),
        ("/checkpoints [on|off]", "git-checkpoint each turn so /undo can roll it back (persists)"),
        ("/undo", "roll the working tree back to the last checkpoint"),
        ("/theme [dark|light|<name>]", "switch theme (persists); named themes live in ~/.marlin/themes/"),
        ("/color [<#rrggbb>|off]", "set the status bar background color for this directory (persists per workdir)"),
        ("/command [list|new|reload]", "manage user-defined slash commands (~/.marlin/commands/)"),
        ("/tool [list|new|reload]", "manage user-defined LLM tools (~/.marlin/tools/)"),
        ("/mcp [list|new|reload]", "manage MCP server connections (~/.marlin/mcp/)"),
        ("/index [status]", "build (or check) the TF-IDF codebase search index"),
        ("/search <query>", "search the index and show ranked results with snippets"),
        ("/revert <file> [n]", "list file snapshots or restore one"),
        ("/resume", "resume the most recent saved session"),
        ("/history [n|clear]", "list saved sessions, load one by number, or clear all"),
        ("/cat <file>", "print file contents"),
        ("/view <file>", "open a scrollable read-only pane for a file"),
        ("/open <file>", "alias for /view"),
        ("/diff-mode <file>", "show current file vs. its most recent snapshot"),
        ("/edit <file>", "open an editable pane (Ctrl+S save, Esc close)"),
        ("/ls [dir]", "list directory"),
        ("/cd <dir>", "change working directory"),
        ("/pwd", "show working directory"),
        ("/skill list", "list installed skills"),
        ("/skill run <name> [query]", "run a skill"),
        ("/skill new <name>", "create a new skill template"),
        ("/skill suggest", "show skill suggestions from nightly analysis"),
        ("/skill reload", "reload skills from disk"),
        ("/skill migrate", "rewrite deprecated .toml skills to .qmd"),
        ("/tiers [on|off]", "model tier routing (easy→default, hard→complex with backups)"),
        ("/subagents [on|off]", "delegate skill runs to a nested subagent loop (on by default)"),
        ("/preflight [startup|skills|all]", "show startup + skill validation diagnostics"),
    ];

    let mut s = "Commands:\n".to_string();
    for (cmd, desc) in &cmds {
        let pad = 32usize.saturating_sub(cmd.len());
        s.push_str(&format!("  {}{}{}\n", cmd, " ".repeat(pad.max(1)), desc));
    }
    s
}

#[cfg(test)]
mod parallel_batching_tests {
    use super::*;

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: format!("id-{name}-{}", name.len()),
            name: name.to_string(),
            input: "{}".into(),
        }
    }

    #[test]
    fn is_parallel_safe_covers_only_read_only_tools() {
        assert!(is_parallel_safe("read_file"));
        assert!(is_parallel_safe("list_directory"));
        assert!(is_parallel_safe("search_codebase"));
        assert!(is_parallel_safe("grep"));
        assert!(is_parallel_safe("glob"));
        assert!(is_parallel_safe("ast_skeleton"));
        assert!(is_parallel_safe("ast_get_node"));

        assert!(!is_parallel_safe("write_file"));
        assert!(!is_parallel_safe("edit_file"));
        assert!(!is_parallel_safe("notebook_edit"));
        assert!(!is_parallel_safe("run_command"));
        assert!(!is_parallel_safe("create_directory"));
        assert!(!is_parallel_safe("ast_mutate"));
        assert!(!is_parallel_safe("run_skill"));
    }

    #[test]
    fn consecutive_safe_calls_share_a_group() {
        let calls = vec![
            call("read_file"),
            call("list_directory"),
            call("search_codebase"),
        ];
        let groups = parallel_group_ids(&calls);
        assert_eq!(groups, vec![Some(0), Some(0), Some(0)]);
    }

    #[test]
    fn solo_safe_call_gets_no_group() {
        let calls = vec![call("write_file"), call("read_file"), call("run_command")];
        let groups = parallel_group_ids(&calls);
        assert_eq!(groups, vec![None, None, None]);
    }

    #[test]
    fn mixed_batch_only_groups_the_consecutive_safe_run() {
        // write, [read, list], run_command, [search, ast_skeleton]
        let calls = vec![
            call("write_file"),
            call("read_file"),
            call("list_directory"),
            call("run_command"),
            call("search_codebase"),
            call("ast_skeleton"),
        ];
        let groups = parallel_group_ids(&calls);
        assert_eq!(groups, vec![None, Some(1), Some(1), None, Some(4), Some(4)]);
    }
}
