use std::path::PathBuf;

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::llm::Message;

/// A persisted conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub created_at: String,
    pub first_prompt: String,
    pub messages: Vec<Message>,
}

/// Directory where sessions are stored: `$XDG_DATA_HOME/pi-rs/sessions/`.
pub fn sessions_dir() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("pi-rs").join("sessions"))
}

/// Generate a new session ID (UUID v4, 16 hex chars = 64 bits).
pub fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()[..16].to_owned()
}

/// Validate a session ID contains only hex characters.
fn validate_id(id: &str) -> Result<()> {
    if id.is_empty() || !id.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(anyhow!("invalid session ID: must be non-empty hex string"));
    }
    Ok(())
}

/// Save a session to disk. Creates the sessions directory if needed.
/// Uses atomic write (temp file + rename) to prevent corruption on crash.
pub fn save(session: &Session) -> Result<PathBuf> {
    validate_id(&session.id)?;
    let dir = sessions_dir().ok_or_else(|| anyhow!("cannot determine data directory"))?;
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.json", session.id));
    let tmp_path = dir.join(format!("{}.json.tmp", session.id));
    let json = serde_json::to_string_pretty(session)?;
    std::fs::write(&tmp_path, &json)?;
    std::fs::rename(&tmp_path, &path)?;
    Ok(path)
}

/// Load a session by ID.
pub fn load(id: &str) -> Result<Session> {
    validate_id(id)?;
    let dir = sessions_dir().ok_or_else(|| anyhow!("cannot determine data directory"))?;
    let path = dir.join(format!("{id}.json"));
    let json =
        std::fs::read_to_string(&path).map_err(|e| anyhow!("session '{id}' not found: {e}"))?;
    let session: Session = serde_json::from_str(&json)?;
    Ok(session)
}

/// List all saved sessions, sorted by creation time (newest first).
pub fn list() -> Result<Vec<Session>> {
    let dir = match sessions_dir() {
        Some(d) if d.exists() => d,
        _ => return Ok(Vec::new()),
    };
    let mut sessions = Vec::new();
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json") {
            match std::fs::read_to_string(&path) {
                Ok(json) => match serde_json::from_str::<Session>(&json) {
                    Ok(s) => sessions.push(s),
                    Err(e) => {
                        eprintln!(
                            "pi: warning: skipping malformed session {}: {e}",
                            path.display()
                        );
                        continue;
                    }
                },
                Err(e) => {
                    eprintln!("pi: warning: cannot read {}: {e}", path.display());
                    continue;
                }
            }
        }
    }
    // Sort newest first. Relies on ISO 8601 format being lexicographically sortable.
    sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(sessions)
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;

    // Serialize tests that mutate XDG_DATA_HOME (process-wide env var).
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn new_id_is_length_16() {
        let id = new_id();
        assert_eq!(id.len(), 16);
    }

    #[test]
    fn new_ids_are_unique() {
        let a = new_id();
        let b = new_id();
        assert_ne!(a, b);
    }

    #[test]
    fn validate_id_accepts_hex() {
        assert!(validate_id("abc123").is_ok());
        assert!(validate_id("0000000000000000").is_ok());
    }

    #[test]
    fn validate_id_rejects_empty() {
        assert!(validate_id("").is_err());
    }

    #[test]
    fn validate_id_rejects_non_hex() {
        assert!(validate_id("not-hex!").is_err());
        assert!(validate_id("xyz").is_err());
    }

    #[test]
    fn save_load_roundtrip() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("pi-rs-test-{}", new_id()));
        unsafe { std::env::set_var("XDG_DATA_HOME", &dir) };

        let session = Session {
            id: "abcd1234abcd1234".to_owned(),
            created_at: "2026-05-23T12:00:00Z".to_owned(),
            first_prompt: "hello".to_owned(),
            messages: vec![],
        };

        let path = save(&session).unwrap();
        assert!(path.exists());

        let loaded = load("abcd1234abcd1234").unwrap();
        assert_eq!(loaded.id, session.id);
        assert_eq!(loaded.first_prompt, session.first_prompt);

        std::fs::remove_dir_all(&dir).ok();
        unsafe { std::env::remove_var("XDG_DATA_HOME") };
    }

    #[test]
    fn list_returns_saved_sessions() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("pi-rs-test-{}", new_id()));
        unsafe { std::env::set_var("XDG_DATA_HOME", &dir) };

        for i in 0..3 {
            let session = Session {
                id: format!("{:016x}", i),
                created_at: format!("2026-05-23T12:0{i}:00Z"),
                first_prompt: format!("prompt {i}"),
                messages: vec![],
            };
            save(&session).unwrap();
        }

        let sessions = list().unwrap();
        assert_eq!(sessions.len(), 3);
        // Newest first
        assert!(sessions[0].created_at >= sessions[1].created_at);

        std::fs::remove_dir_all(&dir).ok();
        unsafe { std::env::remove_var("XDG_DATA_HOME") };
    }

    #[test]
    fn load_nonexistent_errors() {
        let _lock = ENV_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        let dir = std::env::temp_dir().join(format!("pi-rs-test-{}", new_id()));
        unsafe { std::env::set_var("XDG_DATA_HOME", &dir) };

        assert!(load("nonexistent000000").is_err());

        std::fs::remove_dir_all(&dir).ok();
        unsafe { std::env::remove_var("XDG_DATA_HOME") };
    }
}
