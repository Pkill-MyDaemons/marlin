use std::time::Instant;

use chrono::Local;
use tachyonfx::{fx, Effect, EffectTimer, Interpolation};
use tui_scrollview::ScrollViewState;
use tui_textarea::TextArea;

use crate::widgets::config_menu::ConfigMenu;
use crate::widgets::diff::DiffPane;
use crate::widgets::editor::EditorPane;
use crate::widgets::suggestions::{all_commands, CmdDef};
use crate::widgets::viewer::ViewerPane;
use marlin_engine::UiUpdate;

use super::entry::{ChatEntry, EntryRole};

// ── Chat state ───────────────────────────────────────────────────────────────

pub struct ChatView {
    pub entries: Vec<ChatEntry>,

    // Input
    pub textarea: TextArea<'static>,
    pub input_history: Vec<String>,
    pub history_idx: i32,
    pub history_draft: String,
    pub suggestions_defs: Vec<CmdDef>,
    pub suggestions: Vec<usize>, // indices into suggestions_defs

    // Streaming
    pub streaming: bool,
    pub stream_buf: String,
    pub tool_iterations: usize,
    pub active_goal: String,
    pub current_tool: String,
    /// Streaming output from a long-running run_command tool call. Rendered
    /// live inside the tool-call bubble rather than bleeding into the main
    /// chat stream buffer (which is model text).
    pub tool_stream_buf: String,

    // Rate-limit
    pub rate_limited: bool,
    pub rate_limit_secs: u32,
    pub rate_limit_total: u32,

    /// Set after a Ctrl+C/Esc that had nothing to cancel (model idle) — a
    /// second consecutive one quits, like Ctrl+Q. Any other key disarms it,
    /// so it only fires on two presses in a row.
    pub(super) quit_armed: bool,

    // Approval modal
    pub approval_pending: Option<String>,
    /// Set when the engine is waiting for the user to answer the model's
    /// ask_user tool call. Holds the question to display in the modal.
    pub ask_pending: Option<String>,

    // /config settings menu overlay
    pub config_menu: Option<ConfigMenu>,

    // /view, /open read-only file preview overlay
    pub viewer: Option<ViewerPane>,

    // /diff-mode overlay
    pub diff_pane: Option<DiffPane>,

    // /edit overlay
    pub editor: Option<EditorPane>,

    /// When a DiffPreview is shown, this holds the tool_id so the user's
    /// accept/reject decision can be sent back to the engine.
    pub(super) pending_diff_tool_id: Option<String>,

    // Scroll
    pub scroll_state: ScrollViewState,
    pub content_height: u16,
    pub viewport_height: u16,
    pub at_bottom: bool,
    /// Animated scroll position (in rows) used while following the bottom.
    /// Eases toward the target bottom so the viewport glides smoothly instead
    /// of jumping — faster when there's a lot of content to scroll through,
    /// slowing down as it approaches (e.g. when the model response stops).
    pub smooth_offset: f64,
    /// True when new content (a stream chunk, tool call, etc.) arrived while
    /// the user was scrolled up, so the renderer can show a "scroll to bottom"
    /// hint. Cleared once the user returns to the bottom.
    pub new_content_arrived: bool,
    /// Smoothed mouse-wheel velocity in notches per second, used to scroll
    /// the viewport proportionally to how fast the wheel is spinning. Fast
    /// spins move more lines per notch; slow spins move fewer. Reset after a
    /// pause so a fresh scroll starts clean instead of inheriting stale
    /// momentum.
    pub scroll_velocity: f64,
    /// Timestamp of the last mouse-wheel event, used to measure wheel speed.
    pub last_scroll_time: Option<Instant>,

    pub width: u16,
    pub height: u16,
    pub provider: String,
    pub model: String,
    /// Absolute working directory, shown in the session status bar.
    pub work_dir: String,
    /// Per-directory status bar background color (None = default theme color).
    pub status_color: Option<[u8; 3]>,
    pub frame: u64,
    pub(super) last_frame_time: Instant,
    pub(super) bubble_effect: Effect,

    // Skill suggestions
    pub skills: Vec<marlin_skills::SkillDef>,
    pub skill_hints: Vec<crate::widgets::suggestions::SkillHint>,
    /// Skill matches sent by the engine for the last message.
    pub last_skill_matches: Vec<(String, String)>,

    // Typewriter animation
    pub typewriter_enabled: bool,
    pub(super) typewriter_pos: usize,

    /// Number of built-in slash commands — used to safely append/remove user commands.
    builtin_cmd_count: usize,
}

impl ChatView {
    pub fn new(width: u16, height: u16) -> Self {
        let mut ta = TextArea::default();
        ta.set_placeholder_text("Message Marlin... (Enter to send, Shift+Enter for newline)");

        let builtin_cmds = all_commands();
        let builtin_count = builtin_cmds.len();
        Self {
            entries: vec![],
            textarea: ta,
            input_history: vec![],
            history_idx: -1,
            history_draft: String::new(),
            suggestions_defs: builtin_cmds,
            suggestions: vec![],
            streaming: false,
            stream_buf: String::new(),
            tool_iterations: 0,
            active_goal: String::new(),
            current_tool: String::new(),
            tool_stream_buf: String::new(),
            rate_limited: false,
            rate_limit_secs: 0,
            rate_limit_total: 0,
            quit_armed: false,
            approval_pending: None,
            ask_pending: None,
            config_menu: None,
            viewer: None,
            diff_pane: None,
            editor: None,
            pending_diff_tool_id: None,
            scroll_state: ScrollViewState::default(),
            content_height: 0,
            viewport_height: 1,
            at_bottom: true,
            smooth_offset: 0.0,
            new_content_arrived: false,
            scroll_velocity: 0.0,
            last_scroll_time: None,
            width,
            height,
            provider: String::new(),
            model: String::new(),
            work_dir: String::new(),
            status_color: None,
            frame: 0,
            last_frame_time: Instant::now(),
            bubble_effect: fx::repeating(fx::ping_pong(fx::hsl_shift_fg(
                [28.0, 0.0, 0.0],
                EffectTimer::from_ms(900, Interpolation::SineInOut),
            ))),
            skills: vec![],
            skill_hints: vec![],
            last_skill_matches: vec![],
            typewriter_enabled: false,
            typewriter_pos: 0,
            builtin_cmd_count: builtin_count,
        }
    }

    pub fn resize(&mut self, width: u16, height: u16) {
        self.width = width;
        self.height = height;
    }

    pub fn add_system(&mut self, text: &str) {
        self.entries.push(ChatEntry::system(text));
        self.maybe_scroll_to_bottom();
    }

    pub fn add_error(&mut self, text: &str) {
        self.entries.push(ChatEntry::error(text));
        self.maybe_scroll_to_bottom();
    }

    /// The viewer/diff/editor overlays are mutually exclusive — `on_key`
    /// intercepts them in a fixed order (viewer, then diff, then editor)
    /// regardless of which was opened most recently, so more than one being
    /// `Some` at once means whichever is checked first silently eats input
    /// meant for a different, visually-on-top pane. Call before opening any
    /// of them so only the newest stays open.
    pub(super) fn close_overlay_panes(&mut self) {
        self.viewer = None;
        self.diff_pane = None;
        self.editor = None;
    }

    pub fn apply_update(&mut self, update: UiUpdate) {
        match update {
            // A session was loaded via /resume, /history <n>, or --resume-last.
            // Repopulate the chat display so the resumed conversation is visible
            // instead of an empty screen (the model already holds the history in
            // its context window — this only restores what the TUI shows).
            UiUpdate::HistoryLoaded(entries) => {
                self.entries.clear();
                self.entries.extend(entries.into_iter().map(history_entry_to_chat));
                // Pin to the bottom so the most recent exchange is in view.
                self.at_bottom = true;
                self.smooth_offset = f64::MAX;
                self.stream_buf.clear();
                self.streaming = false;
                self.current_tool.clear();
                self.tool_iterations = 0;
                self.maybe_scroll_to_bottom();
            }
            UiUpdate::StreamChunk(chunk) => {
                if !self.streaming {
                    self.typewriter_pos = 0;
                }
                self.streaming = true;
                self.stream_buf.push_str(&chunk);
                self.maybe_scroll_to_bottom();
            }
            UiUpdate::ToolCall { name, input } => {
                // mark_complete is a pure completion signal — its summary is
                // rendered separately (see UiUpdate::Summary) as muted text, so
                // don't show the tool-call bubble or count it as a tool iteration.
                if name == "mark_complete" {
                    return;
                }
                self.current_tool = name.clone();
                self.tool_iterations += 1;
                // A new tool call starts — reset any leftover streaming output
                // from a previous tool so it doesn't bleed into this one.
                self.tool_stream_buf.clear();
                // Commit any partial streamed text as an Assistant entry *before*
                // the tool call. Otherwise the in-progress text stays in stream_buf
                // (which is always appended at the very bottom of the viewport) and
                // the tool bubble renders above it — even though the text was said
                // first. Committing here makes the text scroll up and the tool bubble
                // follow it chronologically, as it should.
                if !self.stream_buf.is_empty() {
                    let text = std::mem::take(&mut self.stream_buf);
                    self.entries.push(ChatEntry {
                        role: EntryRole::Assistant,
                        content: text,
                        tool_name: String::new(),
                        time: Local::now(),
                    });
                }
                self.entries.push(ChatEntry {
                    role: EntryRole::ToolCall,
                    content: input,
                    tool_name: name,
                    time: Local::now(),
                });
                self.maybe_scroll_to_bottom();
            }
            UiUpdate::ToolResult {
                name,
                output,
                is_error,
            } => {
                // The mark_complete result ("Acknowledged.") is a pure signal —
                // hide it along with its tool call.
                if name == "mark_complete" {
                    return;
                }
                // The tool finished — clear the live streaming buffer so the
                // committed result (below) is what shows in the bubble.
                self.tool_stream_buf.clear();
                self.entries.push(ChatEntry {
                    role: EntryRole::ToolResult { is_error },
                    content: output,
                    tool_name: name,
                    time: Local::now(),
                });
                self.maybe_scroll_to_bottom();
            }
            UiUpdate::Summary(summary) => {
                self.entries.push(ChatEntry {
                    role: EntryRole::Summary,
                    content: summary,
                    tool_name: String::new(),
                    time: Local::now(),
                });
                self.maybe_scroll_to_bottom();
            }
            UiUpdate::SteerResult(text) => {
                self.entries.push(ChatEntry {
                    role: EntryRole::Steer,
                    content: text,
                    tool_name: String::new(),
                    time: Local::now(),
                });
                self.maybe_scroll_to_bottom();
            }
            UiUpdate::SystemMsg(msg) => {
                self.add_system(&msg);
            }
            UiUpdate::ErrorMsg(msg) => {
                self.add_error(&msg);
            }
            UiUpdate::RateLimited { secs } => {
                self.rate_limited = true;
                self.rate_limit_secs = secs;
                self.rate_limit_total = secs;
                self.streaming = false;
                self.add_system(&format!(
                    "Rate limited. Resuming automatically in {secs}s..."
                ));
            }
            UiUpdate::GoalComplete { tool_count } => {
                self.streaming = false;
                self.current_tool = String::new();
                self.typewriter_pos = 0;
                if !self.stream_buf.is_empty() {
                    let text = std::mem::take(&mut self.stream_buf);
                    self.entries.push(ChatEntry {
                        role: EntryRole::Assistant,
                        content: text,
                        tool_name: String::new(),
                        time: Local::now(),
                    });
                }
                if tool_count > 0 {
                    self.add_system(&format!("Goal complete. ({tool_count} tool calls)"));
                }
                self.active_goal.clear();
                self.tool_iterations = 0;
                self.maybe_scroll_to_bottom();
            }
            UiUpdate::StatusUpdate(info) => {
                self.provider = info.provider;
                self.model = info.model;
                self.work_dir = info.work_dir;
                self.status_color = info.status_color;
            }
            UiUpdate::AwaitingApproval { cmd } => {
                self.approval_pending = Some(cmd);
            }
            UiUpdate::AskUser { question } => {
                self.ask_pending = Some(question);
            }
            UiUpdate::ConfigState { state, open } => {
                if let Some(menu) = &mut self.config_menu {
                    menu.sync(state);
                } else if open {
                    self.config_menu = Some(ConfigMenu::new(state, self.typewriter_enabled));
                }
            }
            UiUpdate::OpenViewer(Ok((path, content))) => {
                self.close_overlay_panes();
                self.viewer = Some(ViewerPane::new(path, content));
            }
            UiUpdate::OpenViewer(Err(msg)) => {
                self.add_error(&msg);
            }
            UiUpdate::OpenDiff { path, diff } => {
                self.close_overlay_panes();
                self.diff_pane = Some(DiffPane::new(path, diff));
            }
            UiUpdate::OpenEditor(Ok((path, content))) => {
                self.close_overlay_panes();
                self.editor = Some(EditorPane::new(path, content));
            }
            UiUpdate::OpenEditor(Err(msg)) => {
                self.add_error(&msg);
            }
            UiUpdate::EditorSaved { path } => {
                if let Some(editor) = &mut self.editor {
                    if editor.path == path {
                        editor.mark_saved();
                    }
                }
            }
            UiUpdate::SkillsLoaded(defs) => {
                self.skills = defs;
            }
            UiUpdate::SkillMatches(matches) => {
                self.last_skill_matches = matches;
            }
            UiUpdate::TierSelected { score, tier } => {
                self.add_system(&format!("difficulty {score}/100 → {tier} tier"));
            }
            UiUpdate::UserCommandsLoaded(defs) => {
                // Keep only built-in commands, then append loaded user commands.
                self.suggestions_defs.truncate(self.builtin_cmd_count);
                for d in defs {
                    self.suggestions_defs.push(CmdDef {
                        cmd: format!("/{}", d.name),
                        args: d.args,
                        desc: d.description,
                    });
                }
            }
            // TaskUpdate, TokenUsage, AstMode, PromptBudget, and Subagent* are
            // consumed by the runner/sidebar.
            UiUpdate::TaskUpdate(_)
            | UiUpdate::PlanUpdate(_)
            | UiUpdate::TokenUsage { .. }
            | UiUpdate::AstMode(_)
            | UiUpdate::PromptBudget(_)
            | UiUpdate::SubagentStarted { .. }
            | UiUpdate::SubagentToolCall { .. }
            | UiUpdate::SubagentFinished { .. } => {}
            UiUpdate::IndexBuilt => {}
            UiUpdate::ToolStreamChunk { chunk } => {
                // Stream tool output into the tool-call bubble buffer so it
                // stays inside the tool box instead of bleeding into the main
                // chat stream (which is model text).
                self.tool_stream_buf.push_str(&chunk);
                self.maybe_scroll_to_bottom();
            }
            UiUpdate::DiffPreview {
                tool_id,
                path,
                diff,
            } => {
                self.close_overlay_panes();
                self.diff_pane = Some(DiffPane::new(path, diff));
                // Store the tool_id so on_key can send AcceptDiff/RejectDiff
                self.pending_diff_tool_id = Some(tool_id);
            }
        }
    }

    pub fn tick_rate_limit(&mut self) -> bool {
        if !self.rate_limited || self.rate_limit_secs == 0 {
            return false;
        }
        self.rate_limit_secs -= 1;
        if self.rate_limit_secs == 0 {
            self.rate_limited = false;
        }
        true
    }

    pub(super) fn maybe_scroll_to_bottom(&mut self) {
        // The scroll_state is driven to the bottom inside render_viewport when at_bottom is true.
        // Calling this just keeps the at_bottom flag set; the position is applied at render time.
        // But if the user has scrolled up, mark that new content arrived so the
        // renderer can show a "scroll to bottom" hint.
        if !self.at_bottom {
            self.new_content_arrived = true;
        }
    }

    /// Tab-complete a file or directory path from the last word in the input.
    /// Returns the full input with the completed path, or None if no match.
    pub(super) fn tab_complete_path(&self, input: &str) -> Option<String> {
        // Find the last word boundary — the part we're trying to complete
        let last_space = input.rfind(' ').map(|i| i + 1).unwrap_or(0);
        let prefix = &input[..last_space];
        let partial = &input[last_space..];

        // Only complete if the partial looks like a path (contains / or is a filename)
        if partial.is_empty() {
            return None;
        }

        // Resolve the partial path relative to work_dir
        let base = std::path::Path::new(&self.work_dir);
        let (search_dir, file_prefix) = if partial.ends_with('/') {
            (base.join(partial), String::new())
        } else if let Some(slash_pos) = partial.rfind('/') {
            let dir_part = &partial[..=slash_pos];
            let file_part = &partial[slash_pos + 1..];
            (base.join(dir_part), file_part.to_string())
        } else {
            (base.to_path_buf(), partial.to_string())
        };

        let entries = match std::fs::read_dir(&search_dir) {
            Ok(iter) => iter,
            Err(_) => return None,
        };

        let mut matches: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().starts_with(&file_prefix))
            .map(|e| {
                let name = e.file_name().to_string_lossy().to_string();
                if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    name + "/"
                } else {
                    name
                }
            })
            .collect();

        if matches.is_empty() {
            return None;
        }

        matches.sort();

        if matches.len() == 1 {
            // Single match — complete it
            let rel = if partial.contains('/') {
                let dir_prefix =
                    partial[..partial.rfind('/').map(|i| i + 1).unwrap_or(0)].to_string();
                format!("{dir_prefix}{}", matches[0])
            } else {
                matches[0].clone()
            };
            Some(format!("{prefix}{rel}"))
        } else {
            // Multiple matches — find common prefix
            let mut common = matches[0].clone();
            for m in &matches[1..] {
                while !m.starts_with(&common) {
                    common.pop();
                    if common.is_empty() {
                        return None;
                    }
                }
            }
            if common.len() > file_prefix.len() {
                let rel = if partial.contains('/') {
                    let dir_prefix =
                        partial[..partial.rfind('/').map(|i| i + 1).unwrap_or(0)].to_string();
                    format!("{dir_prefix}{common}")
                } else {
                    common
                };
                Some(format!("{prefix}{rel}"))
            } else {
                None
            }
        }
    }
}

/// Convert a `HistoryEntry` (from a loaded/resumed session) into a `ChatEntry`
/// the renderer knows how to draw. Tool-call entries carry their tool name and
/// JSON input; tool-result entries carry an error flag.
fn history_entry_to_chat(e: marlin_engine::HistoryEntry) -> ChatEntry {
    let role = match e.role.as_str() {
        "assistant" if !e.tool_name.is_empty() => EntryRole::ToolCall,
        "tool" => EntryRole::ToolResult { is_error: e.is_error },
        "assistant" => EntryRole::Assistant,
        _ => EntryRole::User,
    };
    let content = match role {
        EntryRole::ToolCall => e.tool_input,
        _ => e.content,
    };
    ChatEntry {
        role,
        content,
        tool_name: e.tool_name,
        time: Local::now(),
    }
}
