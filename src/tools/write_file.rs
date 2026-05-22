use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::fs;

use super::{Tool, ToolCtx};
use crate::confirm::confirm;

pub struct WriteTool;

#[derive(Deserialize)]
struct Input {
    path: String,
    content: String,
}

#[async_trait]
impl Tool for WriteTool {
    fn name(&self) -> &'static str {
        "write"
    }

    fn description(&self) -> &'static str {
        "Write a UTF-8 file (overwriting if it exists). Auto-creates parent dirs. \
         Content is written byte-for-byte; no newline coercion. \
         Asks for confirmation if the target resolves outside the agent's CWD \
         (skipped under --yolo)."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path":    { "type": "string", "description": "File path. Relative paths resolve against the agent's CWD." },
                "content": { "type": "string", "description": "File contents, written as-is." }
            },
            "required": ["path", "content"]
        })
    }

    async fn run(&self, ctx: ToolCtx, input: serde_json::Value) -> Result<String> {
        let inp: Input =
            serde_json::from_value(input).map_err(|e| anyhow!("write: invalid input: {e}"))?;
        let path = PathBuf::from(&inp.path);

        if !ctx.yolo
            && is_outside_cwd(&path)
            && !confirm(&format!("write outside CWD: {}", path.display())).await?
        {
            return Ok("Error: user denied write".to_owned());
        }

        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
            && let Err(e) = fs::create_dir_all(parent).await
        {
            return Ok(format!(
                "Error: create parent dir {}: {e}",
                parent.display()
            ));
        }

        match fs::write(&path, inp.content.as_bytes()).await {
            Ok(()) => Ok(format!("wrote {} bytes to {}", inp.content.len(), inp.path)),
            Err(e) => Ok(format!("Error: write {}: {e}", inp.path)),
        }
    }
}

/// True if `path`, resolved through symlinks where it (or its closest existing
/// ancestor) actually exists, sits outside the process CWD.
pub(super) fn is_outside_cwd(path: &Path) -> bool {
    let cwd = match std::env::current_dir().and_then(|p| p.canonicalize()) {
        Ok(p) => p,
        Err(_) => return true, // be cautious if we can't even resolve CWD
    };
    let resolved = canonicalize_or_parent(path);
    !resolved.starts_with(&cwd)
}

/// canonicalize() requires the path to exist. For new files we walk up to the
/// nearest existing ancestor and resolve from there, then re-attach the tail.
fn canonicalize_or_parent(path: &Path) -> PathBuf {
    let abs = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().unwrap_or_default().join(path)
    };
    let mut cur = abs.as_path();
    let mut tail = PathBuf::new();
    loop {
        if let Ok(c) = cur.canonicalize() {
            return c.join(&tail);
        }
        match cur.file_name() {
            Some(name) => {
                let mut new_tail = PathBuf::from(name);
                new_tail.push(&tail);
                tail = new_tail;
                cur = cur.parent().unwrap_or(Path::new(""));
            }
            None => return abs,
        }
    }
}
