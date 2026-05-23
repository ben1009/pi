use std::{process::Stdio, time::Duration};

use anyhow::{Result, anyhow};
use async_trait::async_trait;
use serde::Deserialize;
use serde_json::json;
use tokio::{
    io::{AsyncRead, AsyncReadExt, AsyncWriteExt},
    process::Command,
    sync::mpsc,
};

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
        let inp: Input =
            serde_json::from_value(input).map_err(|e| anyhow!("bash: invalid input: {e}"))?;
        let timeout_ms = inp
            .timeout_ms
            .unwrap_or(DEFAULT_TIMEOUT_MS)
            .min(MAX_TIMEOUT_MS);

        if !ctx.yolo && !confirm(&format!("run bash: {}", inp.command)).await? {
            return Ok("Error: user denied bash execution".to_owned());
        }

        let mut cmd = Command::new("bash");
        // `exec 2>&1` redirects FD 2 to FD 1 in the running shell, then the
        // user command runs on the next line. Avoids the `{ …; }` wrapper —
        // which breaks if the model emits a trailing `#` comment or `;`.
        cmd.arg("-c")
            .arg(format!("exec 2>&1\n{}", inp.command))
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

        let stdout_tasks = if ctx.stream_stderr {
            // Stream output to stderr in real-time via a channel.
            let (tx, mut rx) = mpsc::channel::<Vec<u8>>(64);
            let reader = stdout.map(|s| tokio::spawn(read_capped_tee(s, cap, tx)));
            // Writer forwards chunks to stderr as they arrive.
            let writer = tokio::spawn(async move {
                let mut stderr = tokio::io::stderr();
                while let Some(chunk) = rx.recv().await {
                    let _ = stderr.write_all(&chunk).await;
                }
            });
            match reader {
                Some(r) => StdoutTasks::Tee(r, writer),
                None => StdoutTasks::None,
            }
        } else {
            match stdout.map(|s| tokio::spawn(read_capped(s, cap))) {
                Some(r) => StdoutTasks::Simple(r),
                None => StdoutTasks::None,
            }
        };

        let status =
            match tokio::time::timeout(Duration::from_millis(timeout_ms), child.wait()).await {
                Ok(r) => r.map_err(|e| anyhow!("bash: wait failed: {e}"))?,
                Err(_) => {
                    // tokio::time::timeout only drops the future; the child keeps
                    // running and can mutate state. Kill and reap before returning.
                    let _ = child.start_kill();
                    let _ = child.wait().await;
                    stdout_tasks.abort();
                    return Ok(format!(
                        "Error: bash command timed out after {timeout_ms}ms (child killed)"
                    ));
                }
            };

        // Child has exited but the pipe fd may have been leaked to a daemon
        // (e.g. `nohup foo &`). Bound the drain so we don't hang on the leaked
        // writer; on timeout, abort the reader and return what we have.
        let out_bytes = stdout_tasks.drain_or_abort().await;

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

/// Tracks the stdout reader task (and optional stderr writer task for tee mode).
enum StdoutTasks {
    None,
    Simple(tokio::task::JoinHandle<Vec<u8>>),
    Tee(
        tokio::task::JoinHandle<Vec<u8>>,
        tokio::task::JoinHandle<()>,
    ),
}

impl StdoutTasks {
    /// Wait briefly for tasks to finish; abort on timeout so tasks don't linger.
    async fn drain_or_abort(self) -> Vec<u8> {
        match self {
            StdoutTasks::None => Vec::new(),
            StdoutTasks::Simple(mut handle) => {
                match tokio::time::timeout(Duration::from_secs(5), &mut handle).await {
                    Ok(res) => res.unwrap_or_default(),
                    Err(_) => {
                        handle.abort();
                        Vec::new()
                    }
                }
            }
            StdoutTasks::Tee(mut reader, mut writer) => {
                let bytes = match tokio::time::timeout(Duration::from_secs(5), &mut reader).await {
                    Ok(res) => res.unwrap_or_default(),
                    Err(_) => {
                        reader.abort();
                        Vec::new()
                    }
                };
                // Writer exits when the channel sender is dropped (reader done).
                // Give it a moment to flush remaining chunks; abort if it lingers.
                if tokio::time::timeout(Duration::from_secs(1), &mut writer)
                    .await
                    .is_err()
                {
                    writer.abort();
                }
                bytes
            }
        }
    }

    /// Abort all tasks immediately (used on timeout).
    fn abort(self) {
        match self {
            StdoutTasks::None => {}
            StdoutTasks::Simple(h) => h.abort(),
            StdoutTasks::Tee(r, w) => {
                r.abort();
                w.abort();
            }
        }
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

/// Like `read_capped`, but also sends each chunk to a channel for real-time
/// streaming (e.g. to stderr). The `cap` only limits the *collected* buffer
/// returned to the model; streaming to stderr continues for the entire output.
/// Uses `try_send` to avoid blocking the reader if the writer falls behind.
async fn read_capped_tee<R: AsyncRead + Unpin>(
    mut r: R,
    cap: usize,
    tx: mpsc::Sender<Vec<u8>>,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(cap.min(8192));
    let mut chunk = vec![0u8; 8192];
    loop {
        let n = match r.read(&mut chunk).await {
            Ok(0) => break,
            Ok(n) => n,
            Err(_) => break,
        };
        // Always stream to stderr (drop chunk if channel full to avoid blocking).
        let _ = tx.try_send(chunk[..n].to_vec());
        // Only collect up to cap for the tool result.
        if out.len() < cap {
            let take = n.min(cap - out.len());
            out.extend_from_slice(&chunk[..take]);
        }
    }
    // Drain remaining data so the child's pipe buffer never fills.
    let mut sink = tokio::io::sink();
    let _ = tokio::io::copy(&mut r, &mut sink).await;
    out
}
