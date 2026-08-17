//! Estimates the token cost of the *base prompt injection* — the system
//! prompt plus the structured tool defs sent with every request — as opposed
//! to `context::estimate_tokens`, which sizes the growing conversation
//! history. Target is ~4k tokens; enforcement is a warning, never a hard cap
//! (see engine::mod's `/tokens` command and the status bar badge).

use marlin_providers::ToolDef;

/// Warn (never block) once the base injection exceeds this many tokens.
pub const WARN_THRESHOLD: usize = 4_000;

pub struct Component {
    pub name: String,
    pub tokens: usize,
}

pub struct Report {
    pub components: Vec<Component>,
    pub total: usize,
}

impl Report {
    pub fn over_budget(&self) -> bool {
        self.total > WARN_THRESHOLD
    }

    pub fn format(&self) -> String {
        let mut lines: Vec<String> = self
            .components
            .iter()
            .map(|c| format!("  {:<20} ~{}t", c.name, c.tokens))
            .collect();
        lines.push(format!(
            "  {:<20} ~{}t{}",
            "total",
            self.total,
            if self.over_budget() {
                format!(" (over {WARN_THRESHOLD}t target)")
            } else {
                String::new()
            }
        ));
        lines.join("\n")
    }
}

fn count(s: &str) -> usize {
    // Same 4-chars/token heuristic as context::estimate_tokens — this module
    // exists to break that estimate down per-component, not to replace the
    // exact provider-side count used for the /tokens conversation total.
    s.len().saturating_add(3) / 4
}

/// Per-component estimate of the base prompt injection: system prompt, then
/// one entry per tool def (name + description + property schema overhead).
pub fn compute(system_prompt: &str, tools: &[ToolDef]) -> Report {
    let mut components = vec![Component {
        name: "system_prompt".into(),
        tokens: count(system_prompt),
    }];

    for t in tools {
        let prop_tokens: usize = t
            .properties
            .iter()
            .map(|p| count(&p.name) + count(&p.description) + 4) // +4: JSON schema overhead per property
            .sum();
        let tokens = count(&t.name) + count(&t.description) + prop_tokens + 8; // +8: per-tool JSON overhead
        components.push(Component {
            name: t.name.clone(),
            tokens,
        });
    }

    let total = components.iter().map(|c| c.tokens).sum();
    Report { components, total }
}

#[cfg(test)]
mod tests {
    use super::*;
    use marlin_config::AstMode;
    use marlin_tools::all_tools;

    #[test]
    fn default_config_stays_under_budget() {
        // No skills, no external tools, no AST harness — the minimal request
        // every fresh install sends. Regressions here mean someone added
        // string-literal bloat back to the base injection.
        let system_prompt = "You are Marlin, an AI coding assistant running in a terminal.\n\
            You help the user write, debug, and understand code.\n\n\
            When asked to create a file, write code, edit something, or run a command — DO IT with the appropriate tool. \
            Do not explain how the user could do it themselves. Do not ask for confirmation before using tools. Just act.\n\n\
            Working directory: /home/user/project\n\n";
        let tools = all_tools(&AstMode::Off, &[], &[], false, &[]);
        let report = compute(system_prompt, &tools);
        assert!(
            !report.over_budget(),
            "default config exceeds the {WARN_THRESHOLD}t budget: {} tokens\n{}",
            report.total,
            report.format()
        );
    }
}
