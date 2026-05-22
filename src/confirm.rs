use anyhow::Result;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

/// Print a y/n prompt to stderr and return the answer. Empty / non-y answers
/// count as "no". Used by tools when --yolo is off.
pub async fn confirm(prompt: &str) -> Result<bool> {
    let mut stderr = tokio::io::stderr();
    stderr
        .write_all(format!("pi: {prompt} [y/N] ").as_bytes())
        .await?;
    stderr.flush().await?;

    let stdin = tokio::io::stdin();
    let mut reader = BufReader::new(stdin).lines();
    let line = reader.next_line().await?.unwrap_or_default();
    let ans = line.trim().to_ascii_lowercase();
    Ok(matches!(ans.as_str(), "y" | "yes"))
}
