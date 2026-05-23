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

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx() -> ToolCtx {
        ToolCtx {
            yolo: true,
            max_output: 1024 * 1024,
            stream_stderr: false,
        }
    }

    #[tokio::test]
    async fn write_creates_file() {
        let dir = std::env::temp_dir().join(format!("pi-rs-write-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("output.txt");
        let input =
            serde_json::json!({"path": path.display().to_string(), "content": "hello world"});
        let result = WriteTool.run(ctx(), input).await.unwrap();
        assert!(result.contains("wrote 11 bytes"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "hello world");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn write_overwrites_existing() {
        let dir = std::env::temp_dir().join(format!("pi-rs-write-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("overwrite.txt");
        std::fs::write(&path, "old content").unwrap();
        let input =
            serde_json::json!({"path": path.display().to_string(), "content": "new content"});
        let result = WriteTool.run(ctx(), input).await.unwrap();
        assert!(result.contains("wrote"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "new content");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn write_creates_parent_dirs() {
        let dir = std::env::temp_dir().join(format!("pi-rs-write-test-{}", uuid::Uuid::new_v4()));
        let path = dir.join("a").join("b").join("c.txt");
        let input = serde_json::json!({"path": path.display().to_string(), "content": "nested"});
        let result = WriteTool.run(ctx(), input).await.unwrap();
        assert!(result.contains("wrote"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "nested");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn write_empty_content() {
        let dir = std::env::temp_dir().join(format!("pi-rs-write-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("empty.txt");
        let input = serde_json::json!({"path": path.display().to_string(), "content": ""});
        let result = WriteTool.run(ctx(), input).await.unwrap();
        assert!(result.contains("wrote 0 bytes"));
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn is_outside_cwd_relative_path() {
        // Relative paths resolve inside CWD.
        assert!(!is_outside_cwd(Path::new("foo.txt")));
        assert!(!is_outside_cwd(Path::new("./foo.txt")));
    }

    #[test]
    fn is_outside_cwd_absolute_outside() {
        assert!(is_outside_cwd(Path::new("/tmp/definitely-outside-cwd-xyz")));
    }

    #[test]
    fn canonicalize_or_parent_existing_file() {
        let dir = std::env::temp_dir().join(format!("pi-rs-write-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("exists.txt");
        std::fs::write(&path, "data").unwrap();
        let resolved = canonicalize_or_parent(&path);
        assert!(resolved.is_absolute());
        assert!(resolved.ends_with("exists.txt"));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn canonicalize_or_parent_nonexistent_nested() {
        let dir = std::env::temp_dir().join(format!("pi-rs-write-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("a").join("b").join("new.txt");
        let resolved = canonicalize_or_parent(&path);
        assert!(resolved.is_absolute());
        assert!(resolved.ends_with("b/new.txt") || resolved.ends_with("b\\new.txt"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
