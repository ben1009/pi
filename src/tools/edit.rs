use std::path::PathBuf;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::fs;

use super::write_file::is_outside_cwd;
use super::{Tool, ToolCtx};
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
        let inp: Input = serde_json::from_value(input)
            .map_err(|e| anyhow!("edit: invalid input: {e}"))?;
        let path = PathBuf::from(&inp.path);

        if !ctx.yolo && is_outside_cwd(&path) {
            if !confirm(&format!("edit outside CWD: {}", path.display())).await? {
                return Ok("Error: user denied edit".to_owned());
            }
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

/// Best-effort hint when an exact match fails: find lines whose first non-blank
/// content overlaps the first non-blank line of `needle`, return up to `n` of them.
fn nearest_lines(haystack: &str, needle: &str, n: usize) -> String {
    let needle_first = needle.lines().next().unwrap_or("").trim();
    if needle_first.is_empty() {
        return "  (no hints available)".to_owned();
    }
    let mut hits: Vec<(usize, &str)> = haystack
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains(needle_first) || needle_first.contains(l.trim()))
        .take(n)
        .map(|(i, l)| (i + 1, l))
        .collect();
    if hits.is_empty() {
        // Fall back to longest-common-substring-ish: any line sharing >= 8 chars.
        if needle_first.len() >= 8 {
            let frag = &needle_first[..needle_first.len().min(16)];
            hits = haystack
                .lines()
                .enumerate()
                .filter(|(_, l)| l.contains(frag))
                .take(n)
                .map(|(i, l)| (i + 1, l))
                .collect();
        }
    }
    if hits.is_empty() {
        return "  (no hints available)".to_owned();
    }
    hits.iter()
        .map(|(i, l)| format!("  {i:>6}\t{l}"))
        .collect::<Vec<_>>()
        .join("\n")
}
