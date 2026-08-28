use std::io;
use std::time::Duration;

use crossterm::{
    event::{
        self, DisableBracketedPaste, DisableFocusChange, DisableMouseCapture,
        EnableBracketedPaste, EnableFocusChange, EnableMouseCapture, Event, KeyCode, KeyModifiers,
        MouseEventKind,
    },
    execute,
    terminal::{self, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    buffer::Buffer,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Clear, Paragraph, Widget},
    Terminal,
};
use tokio::sync::mpsc;

use crate::styles;
use crate::views::{chat::ChatView, splash::SplashView};
use crate::widgets::{sidebar::Sidebar, statusbar::StatusBar};
use marlin_engine::{Action, UiUpdate};

enum View {
    Splash(SplashView),
    Chat,
}

pub fn run(
    action_tx: mpsc::Sender<Action>,
    mut ui_rx: mpsc::Receiver<UiUpdate>,
    layout: marlin_config::LayoutConfig,
) -> io::Result<()> {
    // If a panic escapes the render loop, the terminal is left in raw mode on
    // the alternate screen — the screen becomes unreadable and the user has to
    // `reset` to recover. Install a panic hook that restores the terminal
    // before the process unwinds, so a crash leaves a usable shell behind.
    // The hook is installed *before* raw mode is entered so it's in place for
    // the whole session.
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Best-effort restore — ignore errors, we're already panicking.
        let _ = terminal::disable_raw_mode();
        let _ = execute!(
            std::io::stdout(),
            LeaveAlternateScreen,
            DisableMouseCapture,
            DisableBracketedPaste,
            DisableFocusChange
        );
        default_hook(info);
    }));

    terminal::enable_raw_mode()?;
    let mut stdout = io::stdout();
    // Without this, a scroll wheel event never reaches the app at all — the
    // terminal emulator just scrolls its own native scrollback buffer instead.
    // Focus-change events are enabled so the app can force a full redraw when
    // the tab regains focus (see the FocusGained handler below) — otherwise a
    // stale alternate-screen frame left behind by the terminal emulator shows
    // as garbled, misaligned text.
    execute!(
        stdout,
        EnterAlternateScreen,
        EnableMouseCapture,
        EnableBracketedPaste,
        EnableFocusChange
    )?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let size = terminal.size()?;
    let mut status_bar = StatusBar::new(size.width);
    let mut chat = ChatView::new(size.width, size.height.saturating_sub(1));
    let mut sidebar = Sidebar::new();
    let mut view = View::Splash(SplashView::new());

    let mut rate_tick = std::time::Instant::now();
    // Set when the terminal tab regains focus. The terminal emulator may leave
    // a stale alternate-screen frame behind (garbled, misaligned text) when the
    // tab is switched away; ratatui's diff-based rendering only repaints cells
    // that changed, so static cells stay corrupted. A full clear forces every
    // cell to be repainted on the next frame.
    let mut force_redraw = false;

    'outer: loop {
        // Process all pending engine updates
        loop {
            match ui_rx.try_recv() {
                Ok(update) => {
                    match &update {
                        UiUpdate::StatusUpdate(info) => {
                            status_bar.provider = info.provider.clone();
                            status_bar.model = info.model.clone();
                            status_bar.git_branch = info.git_branch.clone();
                            status_bar.bg_count = info.bg_count;
                        }
                        UiUpdate::AstMode(mode) => {
                            status_bar.ast_mode = mode.clone();
                        }
                        UiUpdate::ToolCall { name, .. } => {
                            status_bar.active_tool = name.clone();
                            status_bar.streaming = true;
                        }
                        UiUpdate::StreamChunk(_) => {
                            status_bar.streaming = true;
                        }
                        UiUpdate::GoalComplete { .. } => {
                            status_bar.active_tool.clear();
                            status_bar.streaming = false;
                        }
                        UiUpdate::TaskUpdate(steps) => {
                            sidebar.tasks = steps.clone();
                        }
                        UiUpdate::PlanUpdate(steps) => {
                            sidebar.plan = steps.clone();
                        }
                        UiUpdate::TokenUsage { used, budget } => {
                            sidebar.token_used = *used;
                            sidebar.token_budget = *budget;
                            sidebar.push_token_sample(*used);
                        }
                        UiUpdate::PromptBudget(over) => {
                            status_bar.prompt_budget_over = *over;
                        }
                        UiUpdate::SubagentStarted { id, label } => {
                            sidebar.subagent_started(id.clone(), label.clone());
                        }
                        UiUpdate::SubagentToolCall { id, name } => {
                            sidebar.subagent_tool_call(id, name.clone());
                        }
                        UiUpdate::SubagentFinished { id, ok } => {
                            sidebar.subagent_finished(id, *ok);
                        }
                        _ => {}
                    }
                    chat.apply_update(update);
                }
                Err(mpsc::error::TryRecvError::Empty) => break,
                Err(mpsc::error::TryRecvError::Disconnected) => break 'outer,
            }
        }

        // Splash auto-transition
        if let View::Splash(ref splash) = view {
            if splash.is_done() {
                view = View::Chat;
                status_bar.mode = "chat".into();
            }
        }

        // Rate-limit countdown (drives the progress bar display only)
        if chat.rate_limited && rate_tick.elapsed() >= Duration::from_secs(1) {
            rate_tick = std::time::Instant::now();
            chat.tick_rate_limit();
        }

        // Snapshot sidebar state for rendering
        let sidebar_snap = Sidebar {
            token_used: sidebar.token_used,
            token_budget: sidebar.token_budget,
            token_history: sidebar.token_history.clone(),
            tasks: sidebar.tasks.clone(),
            plan: sidebar.plan.clone(),
            subagents: sidebar.subagents.clone(),
            selected_category: sidebar.selected_category,
            expanded: sidebar.expanded,
            focused: sidebar.focused,
        };
        let approval_cmd = chat.approval_pending.clone();
        let ask_question = chat.ask_pending.clone();

        // Render
        if force_redraw {
            // The terminal emulator may have left a stale alternate-screen frame
            // behind while the tab was unfocused. Clear the whole screen so the
            // next frame repaints every cell from scratch instead of relying on
            // ratatui's diff (which only repaints changed cells).
            terminal.clear()?;
            force_redraw = false;
        }
        terminal.draw(|f| {
            let area = f.area();
            let buf = f.buffer_mut();

            // Fill background
            Block::default()
                .style(styles::style_app_bg())
                .render(area, buf);

            match &mut view {
                View::Splash(splash) => {
                    splash.render(area, buf);
                }
                View::Chat => {
                    let status_area = Rect {
                        y: area.y,
                        height: 1,
                        ..area
                    };
                    let body_area = Rect {
                        y: area.y + 1,
                        height: area.height.saturating_sub(1),
                        ..area
                    };
                    status_bar.streaming = chat.streaming;
                    status_bar.frame = status_bar.frame.wrapping_add(1);
                    status_bar.render(status_area, buf);

                    // Split body into chat + optional sidebar
                    let (chat_area, sidebar_area) = if area.width >= layout.min_sidebar_width {
                        let chunks = Layout::default()
                            .direction(Direction::Horizontal)
                            .constraints([
                                Constraint::Min(40),
                                Constraint::Length(layout.sidebar_width),
                            ])
                            .split(body_area);
                        (chunks[0], Some(chunks[1]))
                    } else {
                        (body_area, None)
                    };

                    chat.render(chat_area, buf);

                    if let Some(sa) = sidebar_area {
                        sidebar_snap.render(sa, buf);
                    }

                    // /config settings menu — rendered on top of chat
                    if let Some(menu) = &chat.config_menu {
                        menu.render(chat_area, buf);
                    }

                    // /view, /open file pane — rendered on top of chat
                    if let Some(viewer) = &mut chat.viewer {
                        viewer.render(chat_area, buf);
                    }

                    // /diff-mode pane — rendered on top of chat
                    if let Some(diff) = &mut chat.diff_pane {
                        diff.render(chat_area, buf);
                    }

                    // /edit pane — rendered on top of chat
                    if let Some(editor) = &mut chat.editor {
                        editor.render(chat_area, buf);
                    }

                    // Approval modal — rendered on top of chat
                    if let Some(cmd) = &approval_cmd {
                        render_approval_modal(cmd, chat_area, buf);
                    }

                    // ask_user modal — rendered on top of chat
                    if let Some(question) = &ask_question {
                        render_ask_modal(question, chat_area, buf);
                    }
                }
            }
        })?;

        // Poll for terminal events (16ms ≈ 60fps). Drain ALL pending events
        // in this frame rather than one-per-frame: a burst of scroll-wheel
        // notches (trackpad momentum) is consumed together, so keystrokes
        // queued behind a long scroll aren't starved until every notch is
        // handled one frame at a time.
        if event::poll(Duration::from_millis(16))? {
            loop {
                match event::read()? {
                    Event::Key(key) => {
                        if key.code == KeyCode::Char('q')
                            && key.modifiers.contains(KeyModifiers::CONTROL)
                        {
                            let _ = action_tx.blocking_send(Action::Quit);
                            break 'outer;
                        }

                        if let View::Splash(_) = &view {
                            view = View::Chat;
                            status_bar.mode = "chat".into();
                            continue;
                        }

                        // Shift+Right focuses the sidebar; Shift+Left (or Esc)
                        // returns to the text input. Plain Left/Right arrows are
                        // left alone so they move the cursor in the input box.
                        // Tab is left free for slash-command autocomplete.
                        if key.code == KeyCode::Right && key.modifiers.contains(KeyModifiers::SHIFT)
                        {
                            sidebar.focused = true;
                            if sidebar.selected_category.is_none() {
                                sidebar.selected_category = Some(0);
                            }
                            continue;
                        }

                        if sidebar.focused {
                            match key.code {
                                KeyCode::Left if key.modifiers.contains(KeyModifiers::SHIFT) => {
                                    sidebar.focused = false
                                }
                                KeyCode::Esc => sidebar.focused = false,
                                KeyCode::Up => sidebar.move_selection(-1),
                                KeyCode::Down => sidebar.move_selection(1),
                                KeyCode::Enter => sidebar.toggle_expand(),
                                _ => {}
                            }
                            continue;
                        }

                        if let Some(action) = chat.on_key(key) {
                            match &action {
                                Action::Quit => {
                                    let _ = action_tx.blocking_send(Action::Quit);
                                    break 'outer;
                                }
                                _ => {
                                    let _ = action_tx.blocking_send(action);
                                }
                            }
                        }
                    }
                    Event::Paste(text) => {
                        if let View::Splash(_) = &view {
                            continue;
                        }
                        chat.on_paste(&text);
                    }
                    Event::Resize(w, h) => {
                        status_bar.width = w;
                        chat.resize(w, h.saturating_sub(1));
                    }
                    Event::FocusGained => {
                        // Tab regained focus — the terminal emulator may have left
                        // a stale alternate-screen frame behind. Force a full
                        // redraw on the next frame.
                        force_redraw = true;
                    }
                    Event::Mouse(mouse) => {
                        if let View::Splash(_) = &view {
                            continue;
                        }
                        let scrolled = match mouse.kind {
                            MouseEventKind::ScrollUp => Some(true),
                            MouseEventKind::ScrollDown => Some(false),
                            _ => None,
                        };
                        if let Some(up) = scrolled {
                            if let Some(action) = chat.on_mouse_scroll(up) {
                                let _ = action_tx.blocking_send(action);
                            }
                        }
                    }
                    _ => {}
                }
                // Stop draining once the queue is empty; the outer loop
                // re-renders and polls again.
                if !event::poll(Duration::ZERO)? {
                    break;
                }
            }
        }
    }

    terminal::disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture,
        DisableBracketedPaste,
        DisableFocusChange
    )?;
    Ok(())
}

fn render_approval_modal(cmd: &str, area: Rect, buf: &mut Buffer) {
    let modal_w = area.width.clamp(40, 72);
    let modal_h = 7u16;
    let x = area.x + (area.width.saturating_sub(modal_w)) / 2;
    let y = area.y + (area.height.saturating_sub(modal_h)) / 2;
    let modal_area = Rect {
        x,
        y,
        width: modal_w,
        height: modal_h,
    };

    // Clear the background
    Clear.render(modal_area, buf);

    // Red-bordered block
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(
            Style::default()
                .fg(Color::Rgb(215, 50, 50))
                .add_modifier(Modifier::BOLD),
        )
        .title(Span::styled(
            " ⚠  Destructive Command ",
            Style::default()
                .fg(Color::Rgb(215, 50, 50))
                .add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(modal_area);
    block.render(modal_area, buf);

    let max_cmd = inner.width.saturating_sub(2) as usize;
    let cmd_display = if cmd.chars().count() > max_cmd {
        cmd.chars()
            .take(max_cmd.saturating_sub(1))
            .collect::<String>()
            + "…"
    } else {
        cmd.to_string()
    };

    let lines = vec![
        Line::from(Span::styled(
            &cmd_display,
            Style::default()
                .fg(Color::Rgb(240, 180, 60))
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Allow this command to run?",
            Style::default().fg(Color::Rgb(210, 220, 240)),
        )),
        Line::from(vec![
            Span::styled(
                "  [y] ",
                Style::default()
                    .fg(Color::Rgb(70, 195, 110))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                "Yes, run it",
                Style::default().fg(Color::Rgb(150, 220, 170)),
            ),
            Span::styled(
                "     [n] ",
                Style::default()
                    .fg(Color::Rgb(215, 70, 70))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("No, deny", Style::default().fg(Color::Rgb(220, 150, 150))),
        ]),
    ];

    Paragraph::new(lines)
        .style(Style::default().bg(Color::Rgb(12, 16, 30)))
        .render(inner, buf);
}

fn render_ask_modal(question: &str, area: Rect, buf: &mut Buffer) {
    let modal_w = area.width.clamp(40, 72);
    let modal_h = 8u16;
    let x = area.x + (area.width.saturating_sub(modal_w)) / 2;
    let y = area.y + (area.height.saturating_sub(modal_h)) / 2;
    let modal_area = Rect {
        x,
        y,
        width: modal_w,
        height: modal_h,
    };

    // Clear the background
    Clear.render(modal_area, buf);

    // Aqua-bordered block
    let block = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Double)
        .border_style(
            Style::default()
                .fg(Color::Rgb(0, 200, 200))
                .add_modifier(Modifier::BOLD),
        )
        .title(Span::styled(
            " ❓  Marlin asks ",
            Style::default()
                .fg(Color::Rgb(0, 200, 200))
                .add_modifier(Modifier::BOLD),
        ));

    let inner = block.inner(modal_area);
    block.render(modal_area, buf);

    let max_q = inner.width.saturating_sub(2) as usize;
    let q_display = if question.chars().count() > max_q {
        question
            .chars()
            .take(max_q.saturating_sub(1))
            .collect::<String>()
            + "…"
    } else {
        question.to_string()
    };

    let lines = vec![
        Line::from(Span::styled(
            &q_display,
            Style::default().fg(Color::Rgb(220, 230, 240)),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Type your answer below, then press Enter.",
            Style::default().fg(Color::Rgb(150, 180, 200)),
        )),
        Line::from(vec![
            Span::styled(
                "  [enter] ",
                Style::default()
                    .fg(Color::Rgb(70, 195, 110))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Submit", Style::default().fg(Color::Rgb(150, 220, 170))),
            Span::styled(
                "     [esc] ",
                Style::default()
                    .fg(Color::Rgb(215, 70, 70))
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("Cancel", Style::default().fg(Color::Rgb(220, 150, 150))),
        ]),
    ];

    Paragraph::new(lines)
        .style(Style::default().bg(Color::Rgb(12, 16, 30)))
        .render(inner, buf);
}
