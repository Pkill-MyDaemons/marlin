use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileEntry {
    pub path: String,
    pub tf: HashMap<String, f64>,
    /// Symbol names (functions, types, classes, consts) defined in this file,
    /// extracted cheaply for common languages. Used for symbol-aware search.
    #[serde(default)]
    pub symbols: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Index {
    pub files: Vec<FileEntry>,
    pub df: HashMap<String, usize>, // document frequency per term
    pub file_count: usize,
    pub term_count: usize,
    pub built_at: DateTime<Utc>,
    #[serde(skip)]
    pub work_dir: String,
}

#[derive(Debug)]
pub struct BuildStats {
    pub files: usize,
    pub terms: usize,
    pub symbols: usize,
    pub elapsed: Duration,
}

pub struct SearchResult {
    pub path: String,
    pub score: f64,
    pub snippet: String,
    /// True when the match came from a symbol name rather than raw term tf-idf.
    pub is_symbol: bool,
}

const SKIP_DIRS: &[&str] = &[
    ".git",
    "node_modules",
    "target",
    "dist",
    ".next",
    "vendor",
    "__pycache__",
    ".venv",
    "venv",
    "build",
    "out",
    ".cargo",
    "bin",
    "obj",
];
const SKIP_EXTS: &[&str] = &[
    "exe", "dll", "so", "dylib", "png", "jpg", "jpeg", "gif", "webp", "ico", "pdf", "zip", "tar",
    "gz", "wasm", "bin", "lock",
];

pub fn build(work_dir: &str, _ignored: Option<()>) -> Result<(Index, BuildStats)> {
    let start = Instant::now();
    let mut files_vec: Vec<FileEntry> = Vec::new();
    let mut df: HashMap<String, usize> = HashMap::new();
    let mut symbol_total = 0usize;

    walk_dir(
        Path::new(work_dir),
        work_dir,
        &mut files_vec,
        &mut df,
        &mut symbol_total,
    )?;

    let file_count = files_vec.len();
    let term_count = df.len();

    let idx = Index {
        files: files_vec,
        df,
        file_count,
        term_count,
        built_at: Utc::now(),
        work_dir: work_dir.to_string(),
    };

    Ok((
        idx,
        BuildStats {
            files: file_count,
            terms: term_count,
            symbols: symbol_total,
            elapsed: start.elapsed(),
        },
    ))
}

fn walk_dir(
    dir: &Path,
    work_dir: &str,
    files: &mut Vec<FileEntry>,
    df: &mut HashMap<String, usize>,
    symbol_total: &mut usize,
) -> Result<()> {
    let entries = std::fs::read_dir(dir)?;
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");

        if path.is_dir() {
            if !SKIP_DIRS.contains(&name) {
                let _ = walk_dir(&path, work_dir, files, df, symbol_total);
            }
            continue;
        }

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if SKIP_EXTS.contains(&ext.as_str()) {
            continue;
        }

        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let rel_path = path
            .strip_prefix(work_dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();

        let terms = tokenize(&text);
        let mut tf: HashMap<String, f64> = HashMap::new();
        let total = terms.len() as f64;
        if total == 0.0 {
            continue;
        }

        for term in &terms {
            *tf.entry(term.clone()).or_insert(0.0) += 1.0;
        }
        for v in tf.values_mut() {
            *v /= total;
        }

        for term in tf.keys() {
            *df.entry(term.clone()).or_insert(0) += 1;
        }

        let symbols = extract_symbols(&ext, &text);
        *symbol_total += symbols.len();

        files.push(FileEntry {
            path: rel_path,
            tf,
            symbols,
        });
    }
    Ok(())
}

fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| t.len() >= 2 && t.len() <= 40)
        .map(|t| t.to_lowercase())
        .collect()
}

/// Cheap, regex-free line scan that pulls top-level symbol declarations out of
/// common languages. Not a full parser — good enough to make `search_symbols`
/// and symbol-boosted search useful without a heavy dependency.
pub fn extract_symbols(ext: &str, text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        // Skip comment/string-ish lines cheaply.
        if t.is_empty()
            || t.starts_with("//")
            || t.starts_with('#')
            || t.starts_with('*')
            || t.starts_with("/*")
            || t.starts_with("<!--")
        {
            continue;
        }
        let sym = match ext {
            "rs" => {
                // fn foo, fn foo<T>, struct Name, enum Name, trait Name, impl Name, type Name =, const FOO, static FOO
                if let Some(r) = strip_keyword(t, "pub fn").or_else(|| strip_keyword(t, "fn")) {
                    ident_after_parens(r)
                } else if let Some(r) =
                    strip_keyword(t, "pub struct").or_else(|| strip_keyword(t, "struct"))
                {
                    Some(symbol_name(r))
                } else if let Some(r) =
                    strip_keyword(t, "pub enum").or_else(|| strip_keyword(t, "enum"))
                {
                    Some(symbol_name(r))
                } else if let Some(r) =
                    strip_keyword(t, "pub trait").or_else(|| strip_keyword(t, "trait"))
                {
                    Some(symbol_name(r))
                } else if let Some(r) = strip_keyword(t, "impl") {
                    // impl Foo, impl<T> Foo
                    let r = r.trim_start_matches(['<', '(']);
                    Some(symbol_name(r))
                } else if let Some(r) =
                    strip_keyword(t, "pub type").or_else(|| strip_keyword(t, "type"))
                {
                    Some(symbol_name(r))
                } else if let Some(r) =
                    strip_keyword(t, "const").or_else(|| strip_keyword(t, "static"))
                {
                    Some(symbol_name(r))
                } else {
                    None
                }
            }
            "go" => {
                if let Some(r) = strip_keyword(t, "func") {
                    // func Name(...)  OR  func (r *T) Name(...)
                    let r = r.trim_start();
                    if r.starts_with('(') {
                        // Receiver form: name comes after the receiver's ')'
                        let after = r.splitn(2, ')').nth(1).unwrap_or("").trim();
                        Some(symbol_name(after))
                    } else {
                        // Plain form: name precedes the first '('
                        Some(r.split('(').next().unwrap_or(r).trim().to_string())
                    }
                } else if let Some(r) = strip_keyword(t, "type") {
                    Some(symbol_name(r))
                } else if let Some(r) =
                    strip_keyword(t, "const").or_else(|| strip_keyword(t, "var"))
                {
                    Some(symbol_name(r))
                } else {
                    None
                }
            }
            "py" => {
                if let Some(r) = strip_keyword(t, "async def").or_else(|| strip_keyword(t, "def")) {
                    Some(symbol_name(r))
                } else if let Some(r) = strip_keyword(t, "class") {
                    Some(symbol_name(r))
                } else {
                    None
                }
            }
            "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" => {
                if let Some(r) =
                    strip_keyword(t, "export function").or_else(|| strip_keyword(t, "function"))
                {
                    Some(symbol_name(r))
                } else if let Some(r) =
                    strip_keyword(t, "export class").or_else(|| strip_keyword(t, "class"))
                {
                    Some(symbol_name(r))
                } else if let Some(r) =
                    strip_keyword(t, "export interface").or_else(|| strip_keyword(t, "interface"))
                {
                    Some(symbol_name(r))
                } else if let Some(r) =
                    strip_keyword(t, "export const").or_else(|| strip_keyword(t, "const"))
                {
                    Some(symbol_name(r))
                } else {
                    None
                }
            }
            "c" | "h" | "cpp" | "hpp" | "cc" | "cxx" | "hh" => {
                if t.starts_with('#') {
                    None
                } else if let Some(r) = strip_keyword(t, "typedef") {
                    Some(symbol_name(r))
                } else if let Some(r) =
                    strip_keyword(t, "static inline").or_else(|| strip_keyword(t, "inline"))
                {
                    Some(symbol_name(r))
                } else {
                    // "type name(" pattern — conservative
                    let lower = t.to_lowercase();
                    if lower.contains("(")
                        && !lower.starts_with("if ")
                        && !lower.starts_with("for ")
                        && !lower.starts_with("while ")
                        && !lower.starts_with("switch ")
                    {
                        let before = t.split('(').next().unwrap_or("").trim();
                        let before = before.split("->").next().unwrap_or(before).trim();
                        let parts: Vec<&str> = before.split_whitespace().collect();
                        if parts.len() >= 2 {
                            Some(
                                parts
                                    .last()
                                    .unwrap_or(&"")
                                    .trim_end_matches('*')
                                    .trim_end_matches('&')
                                    .to_string(),
                            )
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                }
            }
            "rb" => {
                if let Some(r) = strip_keyword(t, "def") {
                    Some(symbol_name(r))
                } else if let Some(r) = strip_keyword(t, "class") {
                    Some(symbol_name(r))
                } else if let Some(r) = strip_keyword(t, "module") {
                    Some(symbol_name(r))
                } else {
                    None
                }
            }
            "sh" | "bash" | "zsh" => {
                if let Some(r) = strip_keyword(t, "function") {
                    Some(symbol_name(r))
                } else if let Some(r) = strip_keyword(t, "export") {
                    Some(symbol_name(r))
                } else {
                    // foo() {
                    if t.ends_with("()") || t.ends_with("() {") {
                        Some(symbol_name(t))
                    } else {
                        None
                    }
                }
            }
            _ => None,
        };
        if let Some(s) = sym {
            if !s.is_empty() {
                out.push(s);
            }
        }
    }
    out
}

/// After a keyword match, take the identifier that follows (up to whitespace,
/// `{`, `:`, `(`, `=`, `<`).
fn symbol_name(rest: &str) -> String {
    rest.trim()
        .split(|c: char| {
            c.is_whitespace()
                || c == '{'
                || c == '('
                || c == '='
                || c == '<'
                || c == ':'
                || c == ';'
        })
        .next()
        .unwrap_or("")
        .trim_matches(|c: char| c == '*' || c == '&' || c == '"' || c == '\'')
        .to_string()
}

/// For `fn foo(`, after stripping the keyword the arg list begins with `(` —
/// skip to the identifier before the parens.
fn ident_after_parens(rest: &str) -> Option<String> {
    let r = rest.trim_start();
    Some(r.splitn(2, '(').next().unwrap_or(r).trim().to_string())
}

/// Strip a leading keyword (e.g. "fn") and the whitespace that follows it,
/// returning the remainder. Returns None if the line doesn't start with it.
fn strip_keyword<'a>(line: &'a str, kw: &str) -> Option<&'a str> {
    let rest = line.strip_prefix(kw)?;
    // Must be followed by whitespace, `(`, `<` (generics), or end — not more
    // identifier chars (avoids matching "fnord" when looking for "fn").
    let next = rest.chars().next()?;
    if next.is_alphanumeric() && next != '(' && next != '<' {
        return None;
    }
    Some(rest)
}

pub fn search(idx: &Index, query: &str, limit: usize) -> Vec<SearchResult> {
    let n = idx.file_count as f64;
    let query_terms: Vec<String> = tokenize(query);
    if query_terms.is_empty() {
        return vec![];
    }

    let mut scores: Vec<(usize, f64, bool)> =
        idx.files
            .iter()
            .enumerate()
            .filter_map(|(i, f)| {
                let mut score = 0.0f64;
                let mut is_symbol = false;
                for term in &query_terms {
                    let tf = f.tf.get(term).copied().unwrap_or(0.0);
                    let df = idx.df.get(term).copied().unwrap_or(1) as f64;
                    let idf = (n / df + 1.0).ln();
                    score += tf * idf;

                    // Symbol hits get a strong boost — a definition match is far more
                    // relevant than incidental term co-occurrence.
                    let sym_hit = f.symbols.iter().any(|s| {
                        s.to_lowercase() == *term || s.to_lowercase().contains(term.as_str())
                    });
                    if sym_hit {
                        score += 3.0;
                        is_symbol = true;
                    }
                }
                if score > 0.0 {
                    Some((i, score, is_symbol))
                } else {
                    None
                }
            })
            .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(limit);

    scores
        .into_iter()
        .map(|(i, score, is_symbol)| {
            let f = &idx.files[i];
            let full_path = format!("{}/{}", idx.work_dir, f.path);
            let snippet = extract_snippet(&full_path, &query_terms);
            SearchResult {
                path: f.path.clone(),
                score,
                snippet,
                is_symbol,
            }
        })
        .collect()
}

/// Find files that *define* the given symbol name (function/type/const/class).
pub fn search_symbols(idx: &Index, symbol: &str, limit: usize) -> Vec<SearchResult> {
    let term = symbol.to_lowercase();
    let mut hits: Vec<(usize, f64)> = Vec::new();
    for (i, f) in idx.files.iter().enumerate() {
        for s in &f.symbols {
            if s.to_lowercase() == term {
                hits.push((i, 10.0));
                break;
            }
            if s.to_lowercase().contains(&term) {
                hits.push((i, 6.0));
                break;
            }
        }
    }
    hits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    hits.truncate(limit);
    hits.into_iter()
        .map(|(i, score)| {
            let f = &idx.files[i];
            let full_path = format!("{}/{}", idx.work_dir, f.path);
            let snippet = extract_snippet(&full_path, &[term.clone()]);
            SearchResult {
                path: f.path.clone(),
                score,
                snippet,
                is_symbol: true,
            }
        })
        .collect()
}

fn extract_snippet(path: &str, terms: &[String]) -> String {
    let Ok(text) = std::fs::read_to_string(path) else {
        return String::new();
    };
    for line in text.lines() {
        let lower = line.to_lowercase();
        if terms.iter().any(|t| lower.contains(t.as_str())) {
            let trimmed = line.trim();
            if trimmed.len() > 120 {
                let mut end = 120;
                while end > 0 && !trimmed.is_char_boundary(end) {
                    end -= 1;
                }
                return format!("{}…", &trimmed[..end]);
            }
            return trimmed.to_string();
        }
    }
    String::new()
}

pub fn format_results(results: &[SearchResult], query: &str) -> String {
    if results.is_empty() {
        return format!("No results for {:?}", query);
    }
    let mut out = format!("Search results for {:?}:\n", query);
    for r in results {
        let mark = if r.is_symbol { "⚡" } else { "  " };
        out.push_str(&format!("  {mark} [{:.3}] {}\n", r.score, r.path));
        if !r.snippet.is_empty() {
            out.push_str(&format!("          {}\n", r.snippet));
        }
    }
    out
}

pub fn format_symbol_results(results: &[SearchResult], symbol: &str) -> String {
    if results.is_empty() {
        return format!("No definitions for {:?}", symbol);
    }
    let mut out = format!("Definitions of {:?}:\n", symbol);
    for r in results {
        out.push_str(&format!("  [{:.3}] {}\n", r.score, r.path));
        if !r.snippet.is_empty() {
            out.push_str(&format!("          {}\n", r.snippet));
        }
    }
    out
}

/// Drop a file from the index (used when it's deleted on disk).
pub fn remove_file(idx: &mut Index, abs_path: &str) {
    let rel_path = abs_path
        .strip_prefix(&idx.work_dir)
        .map(|s| s.trim_start_matches('/').to_string())
        .unwrap_or_else(|| abs_path.to_string());
    if let Some(pos) = idx.files.iter().position(|f| f.path == rel_path) {
        let old = &idx.files[pos];
        for term in old.tf.keys() {
            if let Some(v) = idx.df.get_mut(term) {
                *v = v.saturating_sub(1);
            }
        }
        idx.files.remove(pos);
        idx.file_count = idx.files.len();
        idx.term_count = idx.df.len();
    }
}

pub fn update_file(idx: &mut Index, abs_path: &str) {
    let Ok(text) = std::fs::read_to_string(abs_path) else {
        return;
    };
    let rel_path = abs_path
        .strip_prefix(&idx.work_dir)
        .map(|s| s.trim_start_matches('/').to_string())
        .unwrap_or_else(|| abs_path.to_string());

    let ext = Path::new(abs_path)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let symbols = extract_symbols(&ext, &text);

    // Remove old entry
    if let Some(pos) = idx.files.iter().position(|f| f.path == rel_path) {
        let old = &idx.files[pos];
        for term in old.tf.keys() {
            if let Some(v) = idx.df.get_mut(term) {
                *v = v.saturating_sub(1);
            }
        }
        idx.files.remove(pos);
    }

    // Re-index
    let terms = tokenize(&text);
    let mut tf: HashMap<String, f64> = HashMap::new();
    let total = terms.len() as f64;
    if total == 0.0 {
        // Keep a stub entry so the file is still discoverable by symbol/name.
        idx.files.push(FileEntry {
            path: rel_path,
            tf,
            symbols,
        });
        idx.file_count = idx.files.len();
        idx.term_count = idx.df.len();
        return;
    }
    for term in &terms {
        *tf.entry(term.clone()).or_insert(0.0) += 1.0;
    }
    for v in tf.values_mut() {
        *v /= total;
    }
    for term in tf.keys() {
        *idx.df.entry(term.clone()).or_insert(0) += 1;
    }

    idx.files.push(FileEntry {
        path: rel_path,
        tf,
        symbols,
    });
    idx.file_count = idx.files.len();
    idx.term_count = idx.df.len();
}

// ── Background refresh ───────────────────────────────────────────────────────
//
// The index is normally kept fresh in-place on every edit (the engine calls
// `update_file` after a write). But files added/removed externally, or an
// index that's simply stale, need a periodic rescan. This module provides a
// lightweight mtime-scan the engine can run from a background task.

#[derive(Clone, Default)]
pub struct RefreshState {
    /// path (relative) -> (mtime_secs, size)
    mtimes: HashMap<String, (i64, u64)>,
}

/// Collect (path, mtime, size) for every indexable file under work_dir,
/// returning the set of files whose mtime/size differ from `state`. Also
/// returns a rebuilt RefreshState snapshot.
pub fn diff_against(state: &RefreshState, work_dir: &str) -> (Vec<String>, RefreshState) {
    let mut next = RefreshState {
        mtimes: HashMap::new(),
    };
    let mut changed: Vec<String> = Vec::new();
    collect_mtimes(
        Path::new(work_dir),
        work_dir,
        &mut next.mtimes,
        &mut changed,
        state,
    );
    (changed, next)
}

fn collect_mtimes(
    dir: &Path,
    work_dir: &str,
    out: &mut HashMap<String, (i64, u64)>,
    changed: &mut Vec<String>,
    prev: &RefreshState,
) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if !SKIP_DIRS.contains(&name) {
                collect_mtimes(&path, work_dir, out, changed, prev);
            }
            continue;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if SKIP_EXTS.contains(&ext.as_str()) {
            continue;
        }

        let Ok(meta) = entry.metadata() else { continue };
        let Ok(mtime) = meta.modified() else { continue };
        let mtime_secs = mtime
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);
        let size = meta.len();
        let rel = path
            .strip_prefix(work_dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .to_string();
        out.insert(rel.clone(), (mtime_secs, size));
        if prev.mtimes.get(&rel) != Some(&(mtime_secs, size)) {
            changed.push(rel);
        }
    }
}

fn index_path(marlin_dir: &Path, work_dir: &str) -> PathBuf {
    let hash = {
        let mut h: u64 = 0xcbf29ce484222325;
        for b in work_dir.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
        format!("{h:016x}")
    };
    let dir = marlin_dir.join("index");
    let _ = std::fs::create_dir_all(&dir);
    dir.join(format!("{hash}.json"))
}

pub fn save(marlin_dir: &Path, idx: &Index) {
    if let Ok(data) = serde_json::to_string(idx) {
        let _ = std::fs::write(index_path(marlin_dir, &idx.work_dir), data);
    }
}

pub fn load(marlin_dir: &Path, work_dir: &str) -> Result<Index> {
    let path = index_path(marlin_dir, work_dir);
    let data = std::fs::read_to_string(path)?;
    let mut idx: Index = serde_json::from_str(&data)?;
    idx.work_dir = work_dir.to_string();
    Ok(idx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_rust_symbols() {
        let src = "fn main() {}\n\
                    pub fn run(args: Vec<String>) -> Result<()> {}\n\
                    struct Config {}\n\
                    pub enum Mode { A, B }\n\
                    trait Builder {}\n\
                    impl Config {}\n\
                    const MAX: usize = 10;\n";
        let syms = extract_symbols("rs", src);
        assert!(syms.contains(&"main".to_string()));
        assert!(syms.contains(&"run".to_string()));
        assert!(syms.contains(&"Config".to_string()));
        assert!(syms.contains(&"Mode".to_string()));
        assert!(syms.contains(&"Builder".to_string()));
        assert!(syms.contains(&"MAX".to_string()));
    }

    #[test]
    fn does_not_match_keyword_inside_identifier() {
        let src = "fn handle() {}\nlet fnord = 1;\n";
        let syms = extract_symbols("rs", src);
        assert!(syms.contains(&"handle".to_string()));
        assert!(!syms.contains(&"fnord".to_string()));
    }

    #[test]
    fn extracts_python_and_go() {
        let py = "import os\ndef parse(data):\n    return 1\nclass Widget:\n    pass\n";
        let psyms = extract_symbols("py", py);
        assert!(psyms.contains(&"parse".to_string()));
        assert!(psyms.contains(&"Widget".to_string()));

        let go = "package main\nfunc Run() {}\nfunc (s *Server) Start() {}\ntype Config struct{}\n";
        let gsyms = extract_symbols("go", go);
        assert!(gsyms.contains(&"Run".to_string()));
        assert!(gsyms.contains(&"Start".to_string()));
        assert!(gsyms.contains(&"Config".to_string()));
    }

    #[test]
    fn search_symbols_finds_definitions() {
        let df: HashMap<String, usize> = HashMap::new();
        let files = vec![
            FileEntry {
                path: "a.rs".into(),
                tf: HashMap::new(),
                symbols: vec!["connect".into()],
            },
            FileEntry {
                path: "b.rs".into(),
                tf: HashMap::new(),
                symbols: vec!["other".into()],
            },
        ];
        let idx = Index {
            files,
            df,
            file_count: 2,
            term_count: 0,
            built_at: Utc::now(),
            work_dir: ".".into(),
        };
        let hits = search_symbols(&idx, "connect", 5);
        assert_eq!(hits.len(), 1);
        assert!(hits[0].is_symbol);
        assert_eq!(hits[0].path, "a.rs");
    }

    #[test]
    fn symbol_hits_boost_search_ranking() {
        let df: HashMap<String, usize> = HashMap::new();
        // b.rs defines "widget"; a.rs merely mentions it once in text.
        let mut a_tf = HashMap::new();
        a_tf.insert("widget".into(), 0.5);
        let files = vec![
            FileEntry {
                path: "a.rs".into(),
                tf: a_tf,
                symbols: vec![],
            },
            FileEntry {
                path: "b.rs".into(),
                tf: HashMap::new(),
                symbols: vec!["Widget".into()],
            },
        ];
        let idx = Index {
            files,
            df,
            file_count: 2,
            term_count: 0,
            built_at: Utc::now(),
            work_dir: ".".into(),
        };
        let results = search(&idx, "widget", 5);
        assert_eq!(results[0].path, "b.rs");
    }

    #[test]
    fn diff_detects_changed_files() {
        // Use a dedicated temp dir so no other process mutates it mid-test
        // (the shared system temp dir can change under us, breaking the
        // "nothing changed on second pass" assertion).
        let dir = std::env::temp_dir().join(format!("marlin_idx_test_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.txt"), "hello world").unwrap();

        let (changed, state) = diff_against(&RefreshState::default(), dir.to_str().unwrap());
        assert_eq!(changed.len(), 1, "initial scan should report a.txt");
        let (changed2, _) = diff_against(&state, dir.to_str().unwrap());
        assert!(changed2.is_empty(), "unchanged scan should report nothing");

        // Touching a file makes it appear changed again.
        std::fs::write(dir.join("a.txt"), "hello world again").unwrap();
        let (changed3, _) = diff_against(&state, dir.to_str().unwrap());
        assert_eq!(changed3.len(), 1, "modified file should be reported");

        std::fs::remove_dir_all(&dir).unwrap();
    }
}
