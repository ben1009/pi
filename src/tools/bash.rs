use std::process::Stdio;
use std::time::Duration;

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::io::{AsyncRead, AsyncReadExt};
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

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("bash: spawn failed: {e}"))?;

        let stdout = child.stdout.take();
        let stderr = child.stderr.take();

        // Drain pipes concurrently with wait(). If the child writes more than
        // ~64KB (the OS pipe buffer) it blocks on write() until someone reads,
        // so reading sequentially after wait() would deadlock.
        let cap = ctx.max_output * 2;
        let stdout_task = stdout.map(|s| tokio::spawn(read_capped(s, cap)));
        let stderr_task = stderr.map(|s| tokio::spawn(read_capped(s, cap)));

        let status = match tokio::time::timeout(
            Duration::from_millis(timeout_ms),
            child.wait(),
        )
        .await
        {
            Ok(r) => r.map_err(|e| anyhow!("bash: wait failed: {e}"))?,
            Err(_) => {
                // tokio::time::timeout only drops the future; the child keeps
                // running and can mutate state. Kill and reap before returning.
                let _ = child.start_kill();
                let _ = child.wait().await;
                if let Some(t) = stdout_task { let _ = t.await; }
                if let Some(t) = stderr_task { let _ = t.await; }
                return Ok(format!("Error: bash command timed out after {timeout_ms}ms (child killed)"));
            }
        };

        let stdout_bytes = match stdout_task {
            Some(t) => t.await.unwrap_or_default(),
            None => Vec::new(),
        };
        let stderr_bytes = match stderr_task {
            Some(t) => t.await.unwrap_or_default(),
            None => Vec::new(),
        };

        let stdout = String::from_utf8_lossy(&stdout_bytes);
        let stderr = String::from_utf8_lossy(&stderr_bytes);
        let mut combined = String::with_capacity(stdout.len() + stderr.len() + 32);
        combined.push_str(&stdout);
        if !stderr.is_empty() {
            if !combined.is_empty() && !combined.ends_with('\n') {
                combined.push('\n');
            }
            combined.push_str(&stderr);
        }
        let code = status.code().unwrap_or(-1);
        let header = if code == 0 {
            String::new()
        } else {
            format!("[exit {code}]\n")
        };
        Ok(truncate(format!("{header}{combined}"), ctx.max_output))
    }
}

/// Read up to `cap` bytes from `r`, draining anything beyond the cap so the
/// child's pipe buffer doesn't block its writes. We don't care about the
/// excess — `truncate()` would discard it anyway, and capping here bounds
/// memory for runaway commands like `cat /dev/urandom`.
async fn read_capped<R: AsyncRead + Unpin>(mut r: R, cap: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(cap.min(8192));
    let mut limited = (&mut r).take(cap as u64);
    let _ = limited.read_to_end(&mut buf).await;
    let mut sink = tokio::io::sink();
    let _ = tokio::io::copy(&mut r, &mut sink).await;
    buf
}
