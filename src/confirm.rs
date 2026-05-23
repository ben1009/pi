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
    Ok(parse_confirm(&line))
}

fn parse_confirm(line: &str) -> bool {
    let ans = line.trim().to_ascii_lowercase();
    matches!(ans.as_str(), "y" | "yes")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_confirm_yes_lowercase() {
        assert!(parse_confirm("y"));
        assert!(parse_confirm("yes"));
    }

    #[test]
    fn parse_confirm_yes_uppercase() {
        assert!(parse_confirm("Y"));
        assert!(parse_confirm("YES"));
        assert!(parse_confirm("Yes"));
    }

    #[test]
    fn parse_confirm_no() {
        assert!(!parse_confirm("n"));
        assert!(!parse_confirm("no"));
        assert!(!parse_confirm("N"));
        assert!(!parse_confirm("NO"));
    }

    #[test]
    fn parse_confirm_empty() {
        assert!(!parse_confirm(""));
        assert!(!parse_confirm("   "));
        assert!(!parse_confirm("\n"));
    }

    #[test]
    fn parse_confirm_other() {
        assert!(!parse_confirm("maybe"));
        assert!(!parse_confirm("1"));
        assert!(!parse_confirm("yep"));
    }
}
