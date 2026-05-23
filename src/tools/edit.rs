use std::path::PathBuf;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::fs;

use super::{Tool, ToolCtx, write_file::is_outside_cwd};
use crate::confirm::confirm;

pub struct EditTool;

#[derive(Deserialize)]
struct Input {
    path: String,
    old_string: String,
    new_string: String,
    #[serde(default)]
    replace_all: bool,
}

#[async_trait]
impl Tool for EditTool {
    fn name(&self) -> &'static str {
        "edit"
    }

    fn description(&self) -> &'static str {
        "Replace exact byte-matching `old_string` with `new_string` in `path`. \
         Errors if `old_string` is not unique unless `replace_all` is set. \
         On a miss, returns up to 3 nearest line-number candidates so you can \
         retry without re-reading. Asks for confirmation if the path resolves \
         outside the agent's CWD."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path":        { "type": "string" },
                "old_string":  { "type": "string", "description": "Exact byte string to replace." },
                "new_string":  { "type": "string", "description": "Replacement string." },
                "replace_all": { "type": "boolean", "default": false, "description": "Replace every occurrence." }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn run(&self, ctx: ToolCtx, input: serde_json::Value) -> Result<String> {
        let inp: Input =
            serde_json::from_value(input).map_err(|e| anyhow!("edit: invalid input: {e}"))?;
        let path = PathBuf::from(&inp.path);

        if !ctx.yolo
            && is_outside_cwd(&path)
            && !confirm(&format!("edit outside CWD: {}", path.display())).await?
        {
            return Ok("Error: user denied edit".to_owned());
        }

        let bytes = match fs::read(&path).await {
            Ok(b) => b,
            Err(e) => return Ok(format!("Error: read {}: {e}", inp.path)),
        };
        let text = match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(_) => return Ok(format!("Error: {} is not valid UTF-8", inp.path)),
        };

        if inp.old_string.is_empty() {
            return Ok("Error: old_string must be non-empty".to_owned());
        }

        let count = text.matches(&inp.old_string).count();
        let new_text = match (count, inp.replace_all) {
            (0, _) => {
                return Ok(format!(
                    "Error: old_string not found in {}. Nearest lines:\n{}",
                    inp.path,
                    nearest_lines(&text, &inp.old_string, 3)
                ));
            }
            (1, _) => text.replacen(&inp.old_string, &inp.new_string, 1),
            (n, false) => {
                return Ok(format!(
                    "Error: old_string occurs {n} times in {}; pass replace_all=true or supply more context",
                    inp.path
                ));
            }
            (_, true) => text.replace(&inp.old_string, &inp.new_string),
        };

        match fs::write(&path, new_text.as_bytes()).await {
            Ok(()) => {
                let n = if inp.replace_all { count } else { 1 };
                Ok(format!("edited {} ({n} replacement(s))", inp.path))
            }
            Err(e) => Ok(format!("Error: write {}: {e}", inp.path)),
        }
    }
}

/// Best-effort hint when an exact match fails: score every line in `haystack`
/// against `needle` using edit-distance similarity, return the top `n`.
fn nearest_lines(haystack: &str, needle: &str, n: usize) -> String {
    let needle_trimmed = needle.trim();
    if needle_trimmed.is_empty() {
        return "  (no hints available)".to_owned();
    }
    // Score each line by similarity to the needle. Use the first non-blank
    // line of the needle as the query for single-line matching, but also
    // score multi-line windows for longer needles.
    let needle_first = needle
        .lines()
        .find(|l| !l.trim().is_empty())
        .unwrap_or("")
        .trim();

    let lines: Vec<&str> = haystack.lines().collect();
    let mut scored: Vec<(usize, f64, &str)> = lines
        .iter()
        .enumerate()
        .map(|(i, line)| {
            let score = line_similarity(line.trim(), needle_first);
            (i + 1, score, *line)
        })
        .filter(|(_, score, _)| *score > 0.1)
        .collect();

    if scored.is_empty() {
        return "  (no hints available)".to_owned();
    }

    // Sort by similarity descending, then by line number ascending.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(n);
    scored
        .iter()
        .map(|(line_num, _score, line)| format!("  {line_num:>6}\t{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

/// Compute similarity between two strings using the `similar` crate.
/// Returns a value between 0.0 (completely different) and 1.0 (identical).
fn line_similarity(a: &str, b: &str) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let diff = similar::TextDiff::from_chars(a, b);
    let matching: usize = diff
        .ops()
        .iter()
        .map(|op| match op {
            similar::DiffOp::Equal { len, .. } => *len,
            _ => 0,
        })
        .sum();
    let max_len = a.len().max(b.len());
    matching as f64 / max_len as f64
}
