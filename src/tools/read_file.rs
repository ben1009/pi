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

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn ctx() -> ToolCtx {
        ToolCtx {
            yolo: true,
            max_output: 1024 * 1024,
            stream_stderr: false,
        }
    }

    #[tokio::test]
    async fn read_basic_file() {
        let dir = std::env::temp_dir().join(format!("pi-rs-read-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "line one\nline two\nline three").unwrap();

        let input = serde_json::json!({"path": path.display().to_string()});
        let result = ReadTool.run(ctx(), input).await.unwrap();
        assert!(result.contains("line one"));
        assert!(result.contains("line two"));
        assert!(result.contains("line three"));
        // Should have line numbers
        assert!(result.contains("1\t"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn read_with_offset_and_limit() {
        let dir = std::env::temp_dir().join(format!("pi-rs-read-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        for i in 1..=10 {
            writeln!(f, "line {i}").unwrap();
        }

        let input =
            serde_json::json!({"path": path.display().to_string(), "offset": 2, "limit": 3});
        let result = ReadTool.run(ctx(), input).await.unwrap();
        assert!(result.contains("line 3"));
        assert!(result.contains("line 4"));
        assert!(result.contains("line 5"));
        assert!(!result.contains("line 1"));
        assert!(!result.contains("line 6"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn read_nonexistent_file() {
        let input = serde_json::json!({"path": "/tmp/nonexistent-file-abc123.txt"});
        let result = ReadTool.run(ctx(), input).await.unwrap();
        assert!(result.starts_with("Error:"));
    }

    #[tokio::test]
    async fn read_non_utf8_file() {
        let dir = std::env::temp_dir().join(format!("pi-rs-read-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("binary.bin");
        std::fs::write(&path, [0xFF, 0xFE, 0x00, 0x01]).unwrap();

        let input = serde_json::json!({"path": path.display().to_string()});
        let result = ReadTool.run(ctx(), input).await.unwrap();
        assert!(result.contains("not valid UTF-8"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn read_empty_window() {
        let dir = std::env::temp_dir().join(format!("pi-rs-read-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.txt");
        std::fs::write(&path, "").unwrap();

        let input = serde_json::json!({"path": path.display().to_string()});
        let result = ReadTool.run(ctx(), input).await.unwrap();
        assert!(result.contains("empty window"));

        std::fs::remove_dir_all(&dir).ok();
    }
}
