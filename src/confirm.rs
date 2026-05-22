use std::io::{BufRead, Write};

use anyhow::Result;

/// Print a y/n prompt to stderr and return the answer. Empty / non-y answers
/// count as "no". Used by tools when --yolo is off.
pub async fn confirm(prompt: &str) -> Result<bool> {
    let prompt = format!("pi: {prompt} [y/N] ");
    let line = tokio::task::spawn_blocking(move || -> std::io::Result<String> {
        let stderr = std::io::stderr();
        let mut stderr = stderr.lock();
        stderr.write_all(prompt.as_bytes())?;
        stderr.flush()?;
        let stdin = std::io::stdin();
        let mut buf = String::new();
        stdin.lock().read_line(&mut buf)?;
        Ok(buf)
    })
    .await??;
    let ans = line.trim().to_ascii_lowercase();
    Ok(matches!(ans.as_str(), "y" | "yes"))
}
