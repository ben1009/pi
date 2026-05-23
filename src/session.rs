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

/// Generate a new session ID (UUID v4, short form).
pub fn new_id() -> String {
    uuid::Uuid::new_v4().to_string()[..8].to_owned()
}

/// Save a session to disk. Creates the sessions directory if needed.
pub fn save(session: &Session) -> Result<PathBuf> {
    let dir = sessions_dir().ok_or_else(|| anyhow!("cannot determine data directory"))?;
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{}.json", session.id));
    let json = serde_json::to_string_pretty(session)?;
    std::fs::write(&path, json)?;
    Ok(path)
}

/// Load a session by ID.
pub fn load(id: &str) -> Result<Session> {
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
        if entry.path().extension().is_some_and(|e| e == "json") {
            match std::fs::read_to_string(entry.path()) {
                Ok(json) => match serde_json::from_str::<Session>(&json) {
                    Ok(s) => sessions.push(s),
                    Err(_) => continue,
                },
                Err(_) => continue,
            }
        }
    }
    // Sort newest first.
    sessions.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Ok(sessions)
}
