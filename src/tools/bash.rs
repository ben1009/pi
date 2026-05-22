use std::process::Stdio;
use std::time::Duration;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::process::Command;

use super::{Tool, ToolCtx, truncate};
use crate::confirm::confirm;

const DEFAULT_TIMEOUT_MS: u64 = 120_000;
const MAX_TIMEOUT_MS: u64 = 600_000;

pub struct BashTool;

#[derive(Deserialize)]
struct Input {
    command: String,
    #[serde(default)]
    timeout_ms: Option<u64>,
}

#[async_trait]
impl Tool for BashTool {
    fn name(&self) -> &'static str {
        "bash"
    }

    fn description(&self) -> &'static str {
        "Run a shell command via `bash -c`. Returns stdout and stderr merged. \
         Default timeout 120s, max 600s. The working directory is the agent's CWD; \
         do not `cd` between calls."
    }

    fn schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "command": { "type": "string", "description": "Shell command to run." },
                "timeout_ms": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_TIMEOUT_MS,
                    "description": "Optional timeout in milliseconds (default 120000, max 600000)."
                }
            },
            "required": ["command"]
        })
    }

    async fn run(&self, ctx: ToolCtx, input: serde_json::Value) -> Result<String> {
        let inp: Input = serde_json::from_value(input)
            .map_err(|e| anyhow!("bash: invalid input: {e}"))?;
        let timeout_ms = inp.timeout_ms.unwrap_or(DEFAULT_TIMEOUT_MS).min(MAX_TIMEOUT_MS);

        if !ctx.yolo && !confirm(&format!("run bash: {}", inp.command)).await? {
            return Ok("Error: user denied bash execution".to_owned());
        }

        let mut cmd = Command::new("bash");
        cmd.arg("-c")
            .arg(&inp.command)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let child = cmd
            .spawn()
            .map_err(|e| anyhow!("bash: spawn failed: {e}"))?;

        let output = match tokio::time::timeout(
            Duration::from_millis(timeout_ms),
            child.wait_with_output(),
        )
        .await
        {
            Ok(r) => r.map_err(|e| anyhow!("bash: wait failed: {e}"))?,
            Err(_) => return Ok(format!("Error: bash command timed out after {timeout_ms}ms")),
        };

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut combined = String::with_capacity(stdout.len() + stderr.len() + 32);
        combined.push_str(&stdout);
        if !stderr.is_empty() {
            if !combined.is_empty() && !combined.ends_with('\n') {
                combined.push('\n');
            }
            combined.push_str(&stderr);
        }
        let code = output.status.code().unwrap_or(-1);
        let header = if code == 0 {
            String::new()
        } else {
            format!("[exit {code}]\n")
        };
        Ok(truncate(format!("{header}{combined}"), ctx.max_output))
    }
}
