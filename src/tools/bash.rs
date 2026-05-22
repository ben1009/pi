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
        // Wrap in `{ …; } 2>&1` so the OS gives us a single, time-ordered stream.
        // Buffering each pipe separately would reorder interleaved logs.
        cmd.arg("-c")
            .arg(format!("{{ {}; }} 2>&1", inp.command))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        let mut child = cmd
            .spawn()
            .map_err(|e| anyhow!("bash: spawn failed: {e}"))?;

        let stdout = child.stdout.take();

        // Drain the pipe concurrently with wait(). If the child writes more than
        // ~64KB (the OS pipe buffer) it blocks on write() until someone reads,
        // so reading sequentially after wait() would deadlock.
        let cap = ctx.max_output.saturating_mul(2);
        let stdout_task = stdout.map(|s| tokio::spawn(read_capped(s, cap)));

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
                if let Some(t) = stdout_task { t.abort(); }
                return Ok(format!("Error: bash command timed out after {timeout_ms}ms (child killed)"));
            }
        };

        // Child has exited but the pipe fd may have been leaked to a daemon
        // (e.g. `nohup foo &`). Bound the drain so we don't hang on the leaked
        // writer; on timeout, abort the reader and return what we have.
        let out_bytes = drain_or_abort(stdout_task).await;

        let out = String::from_utf8_lossy(&out_bytes);
        let code = status.code().unwrap_or(-1);
        let header = if code == 0 {
            String::new()
        } else {
            format!("[exit {code}]\n")
        };
        Ok(truncate(format!("{header}{out}"), ctx.max_output))
    }
}

/// Read up to `cap` bytes from `r` into a buffer, then keep draining (and
/// discarding) the rest of the stream so the child's pipe buffer never fills.
/// Without the post-cap drain, a child that writes more than `cap` bytes would
/// fill the ~64KB OS pipe buffer and block on `write()` until the bash timeout.
async fn read_capped<R: AsyncRead + Unpin>(mut r: R, cap: usize) -> Vec<u8> {
    let mut buf = Vec::with_capacity(cap.min(8192));
    let mut limited = (&mut r).take(cap as u64);
    let _ = limited.read_to_end(&mut buf).await;
    let mut sink = tokio::io::sink();
    let _ = tokio::io::copy(&mut r, &mut sink).await;
    buf
}

/// Wait briefly for the pipe-reader task to finish; if it doesn't (leaked pipe
/// fd holding the writer side open), abort the handle so the task doesn't
/// linger detached.
async fn drain_or_abort(task: Option<tokio::task::JoinHandle<Vec<u8>>>) -> Vec<u8> {
    let Some(mut handle) = task else { return Vec::new() };
    match tokio::time::timeout(Duration::from_secs(5), &mut handle).await {
        Ok(res) => res.unwrap_or_default(),
        Err(_) => {
            handle.abort();
            Vec::new()
        }
    }
}
