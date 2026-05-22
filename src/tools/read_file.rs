use std::path::PathBuf;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::fs;

use super::{Tool, ToolCtx, truncate, write_file::is_outside_cwd};
use crate::confirm::confirm;

const DEFAULT_LIMIT: usize = 2000;

pub struct ReadTool;

#[derive(Deserialize)]
struct Input {
    path: String,
    #[serde(default)]
    offset: Option<usize>,
    #[serde(default)]
    limit: Option<usize>,
}

#[async_trait]
impl Tool for ReadTool {
    fn name(&self) -> &'static str {
        "read"
    }

    fn description(&self) -> &'static str {
        "Read a UTF-8 text file with `cat -n` style line numbers. \
         Default window 2000 lines from offset 0. \
         Returns an error for non-UTF-8 files."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "path":   { "type": "string", "description": "File path. Relative paths resolve against the agent's CWD." },
                "offset": { "type": "integer", "minimum": 0, "description": "0-based line offset to start at." },
                "limit":  { "type": "integer", "minimum": 1, "description": "Max number of lines to return (default 2000)." }
            },
            "required": ["path"]
        })
    }

    async fn run(&self, ctx: ToolCtx, input: serde_json::Value) -> Result<String> {
        let inp: Input =
            serde_json::from_value(input).map_err(|e| anyhow!("read: invalid input: {e}"))?;
        let path = PathBuf::from(&inp.path);

        // Same CWD-boundary policy as write/edit: confirm if reading outside
        // CWD when --yolo is off, since `read` exfiltrates contents to the
        // model provider.
        if !ctx.yolo
            && is_outside_cwd(&path)
            && !confirm(&format!("read outside CWD: {}", path.display())).await?
        {
            return Ok("Error: user denied read".to_owned());
        }

        let bytes = match fs::read(&path).await {
            Ok(b) => b,
            Err(e) => return Ok(format!("Error: read {}: {e}", inp.path)),
        };
        let text = match std::str::from_utf8(&bytes) {
            Ok(s) => s,
            Err(_) => return Ok(format!("Error: {} is not valid UTF-8", inp.path)),
        };

        let offset = inp.offset.unwrap_or(0);
        let limit = inp.limit.unwrap_or(DEFAULT_LIMIT);

        let mut out = String::with_capacity(text.len().min(64 * 1024));
        let mut shown = 0usize;
        let total = text.lines().count();
        for (i, line) in text.lines().enumerate().skip(offset).take(limit) {
            out.push_str(&format!("{:>6}\t{}\n", i + 1, line));
            shown += 1;
        }
        if shown == 0 {
            out.push_str(&format!("(empty window: file has {total} line(s))\n"));
        } else if offset + shown < total {
            out.push_str(&format!(
                "... <{} more lines not shown; pass offset={} to continue>\n",
                total - (offset + shown),
                offset + shown
            ));
        }
        Ok(truncate(out, ctx.max_output))
    }
}
