//! Channel message types shared between the engine (producer), the TUI
//! (consumer), and the skills daemon (which emits `UiUpdate::SystemMsg`).
//! These live in `marlin-core` so the engine and TUI crates can both depend
//! on them without forming a cycle.

use marlin_commands::commands::UserCommandDef;
use marlin_config::config::AstMode;
use marlin_snapshots::snapshots::DiffLine;

use crate::skill::SkillDef;
use crate::tasks::TaskStep;

/// A single conversation entry from a loaded/resumed session, carried to the
/// TUI so it can repopulate its chat display after `/resume` or `--resume-last`.
/// Lives in `marlin-core` (rather than reusing the engine's provider `Message`)
/// so the TUI doesn't need a dependency on `marlin-providers`/`marlin-history`.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    /// "user" | "assistant" | "tool"
    pub role: String,
    pub content: String,
    /// Tool name for a tool-call entry (empty otherwise).
    pub tool_name: String,
    /// JSON-encoded input args for a tool-call entry (empty otherwise).
    pub tool_input: String,
    /// True for a failed tool result.
    pub is_error: bool,
}

#[derive(Debug)]
pub enum UiUpdate {
    StreamChunk(String),
    /// A session was loaded via `/resume` or `--resume-last` — the TUI should
    /// repopulate its chat display with these entries.
    HistoryLoaded(Vec<HistoryEntry>),
    ToolCall {
        name: String,
        input: String,
    },
    ToolResult {
        name: String,
        output: String,
        is_error: bool,
    },
    /// Final summary text from a `mark_complete` tool call. The TUI renders it
    /// as a muted/grayed-out assistant-style message instead of a tool bubble.
    Summary(String),
    SystemMsg(String),
    ErrorMsg(String),
    RateLimited {
        secs: u32,
    },
    GoalComplete {
        tool_count: usize,
    },
    StatusUpdate(StatusInfo),
    IndexBuilt,
    /// Engine is paused waiting for user approval of a destructive command
    AwaitingApproval {
        cmd: String,
    },
    /// Engine is paused waiting for the user to answer the model's ask_user
    /// tool call. The user's typed reply comes back as Action::UserAnswer.
    AskUser {
        question: String,
    },
    /// Updated task list for the sidebar
    TaskUpdate(Vec<TaskStep>),
    /// Upfront plan for the sidebar — coarse ordered checklist, separate from
    /// the granular `TaskUpdate` log.
    PlanUpdate(Vec<TaskStep>),
    /// Opens the /view or /open read-only file pane with `Ok((resolved_path,
    /// content))`, or reports why it couldn't (missing file, read error) as `Err`.
    OpenViewer(Result<(String, String), String>),
    /// Opens the /diff-mode pane: current file content vs. its most recent snapshot.
    OpenDiff {
        path: String,
        diff: Vec<DiffLine>,
    },
    /// Opens the /edit pane with `Ok((resolved_path, content))`, or reports why
    /// it couldn't (missing file, read error) as `Err`.
    OpenEditor(Result<(String, String), String>),
    /// A Ctrl+S save from the /edit pane completed — lets the pane clear its
    /// dirty flag and rebase its "original content" comparison.
    EditorSaved {
        path: String,
    },
    /// Token budget update for the sidebar meter
    TokenUsage {
        used: usize,
        budget: usize,
    },
    /// Base prompt injection (system prompt + tool defs) budget check — informational,
    /// never blocking. `Some(total_tokens)` when over budget::WARN_THRESHOLD, else `None`.
    PromptBudget(Option<usize>),
    /// AST mode changed — drives the status bar badge
    AstMode(AstMode),
    /// Skills loaded on startup — TUI uses these for typing suggestions.
    SkillsLoaded(Vec<SkillDef>),
    /// Skill keyword matches for the most recent user message.
    SkillMatches(Vec<(String, String)>),
    /// Difficulty score and selected tier for the current request.
    TierSelected {
        score: u8,
        tier: String,
    },
    /// User-defined commands loaded from ~/.marlin/commands/ — sent to TUI for autocomplete.
    UserCommandsLoaded(Vec<UserCommandDef>),
    /// A subagent (delegated skill run) started — shown in the sidebar below Tasks.
    SubagentStarted {
        id: String,
        label: String,
    },
    /// A subagent is about to run one tool call — updates its sidebar status line.
    SubagentToolCall {
        id: String,
        name: String,
    },
    /// A subagent finished (successfully or not).
    SubagentFinished {
        id: String,
        ok: bool,
    },
    /// Config snapshot for the interactive /config menu. `open` is true when
    /// the user ran /config; false for refreshes after a ConfigSet applied.
    ConfigState {
        state: ConfigState,
        open: bool,
    },
    /// Streaming chunk from a long-running run_command tool call.
    ToolStreamChunk {
        chunk: String,
    },
    /// Diff preview for a pending write_file/edit_file — the TUI shows a
    /// unified diff and the user accepts or rejects before the write happens.
    DiffPreview {
        tool_id: String,
        path: String,
        diff: Vec<DiffLine>,
    },
    /// Result of a steering command/note the user sent while the model was
    /// working. Rendered as a distinct text field in the model's output area
    /// without interrupting the in-flight stream.
    SteerResult(String),
}

#[derive(Debug, Clone)]
pub struct StatusInfo {
    pub provider: String,
    pub model: String,
    /// Current git branch of the work directory (None if not a git repo).
    pub git_branch: Option<String>,
    /// Absolute path of the working directory.
    pub work_dir: String,
    /// Per-directory status bar background color (None = default theme color).
    pub status_color: Option<[u8; 3]>,
    /// Number of currently-running background processes (see bg_start/bg_kill).
    pub bg_count: usize,
}

/// Snapshot of the editable settings shown in the /config menu.
#[derive(Debug, Clone, Default)]
pub struct ConfigState {
    pub provider: String,
    pub providers: Vec<String>,
    /// Raw API key for `provider` — always kept in sync so the /config menu's
    /// API key field tracks whichever provider is currently selected.
    pub api_key: String,
    pub model: String,
    pub models: Vec<String>,
    pub theme: String,
    /// Names of available named themes (~/.marlin/themes/*.toml) — the /config
    /// menu cycles through these alongside dark/light.
    pub named_themes: Vec<String>,
    pub sandbox_mode: String,
    pub skip_permissions: bool,
    pub clean_env: bool,
    pub ast_mode: String,
    pub skill_subagents: bool,
    pub max_tokens: usize,
    pub tool_call_limit: usize,
}

#[derive(Debug, Clone)]
pub enum Action {
    SendMessage(String),
    SlashCommand(String),
    CancelStream,
    /// User approved a destructive command in the modal
    Approve,
    /// User denied a destructive command in the modal
    Deny,
    /// User typed an answer to the model's ask_user tool call.
    UserAnswer(String),
    /// A setting was changed in the /config menu.
    ConfigSet {
        key: String,
        value: String,
    },
    /// Ctrl+S in the /edit pane — write `content` to `path`, through the same
    /// preflight funnel (path-escape approval, snapshotting) as the LLM's
    /// own write_file tool call.
    SaveEditorFile {
        path: String,
        content: String,
    },
    /// User accepted a diff preview for a pending write_file/edit_file.
    AcceptDiff {
        tool_id: String,
    },
    /// User rejected a diff preview for a pending write_file/edit_file.
    RejectDiff {
        tool_id: String,
    },
    /// A steering command/note the user typed while the model was working.
    /// If it's a slash command it's executed instantly; otherwise it's shown
    /// as a text field in the model's output without interrupting the stream.
    Steer(String),
    Quit,
}
