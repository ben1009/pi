pub mod config;
pub mod confirm;
pub mod context;
pub mod llm;
pub mod mcp;
pub mod session;
pub mod tools;

pub fn system_prompt() -> String {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "<unknown>".to_owned());
    let os = std::env::consts::OS;
    let date = today_utc();
    format!(
        "You are pi-rs, a CLI coding agent. You help the user edit and run code in their working directory.\n\n\
         Working directory: {cwd}\n\
         Operating system: {os}\n\
         Date: {date}\n\n\
         Prefer using the provided tools (bash, read, write, edit) over guessing. \
         When a tool returns an error, read the error and try a different approach. \
         Be concise."
    )
}

fn today_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let (y, m, d) = days_to_ymd(days);
    format!("{y:04}-{m:02}-{d:02}")
}

/// Howard Hinnant's civil_from_days. Days are signed days since 1970-01-01.
pub fn days_to_ymd(mut days: i64) -> (i32, u32, u32) {
    days += 719_468;
    let era = if days >= 0 { days } else { days - 146_096 } / 146_097;
    let doe = (days - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32;
    let y = if m <= 2 { y + 1 } else { y };
    (y as i32, m, d)
}

/// Shared test utilities.
#[cfg(test)]
pub mod test_util {
    use std::sync::Mutex;

    /// Global lock for tests that mutate process-wide environment variables.
    /// Shared across all modules to prevent cross-module race conditions.
    pub static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// RAII guard that restores an environment variable on drop.
    /// Safe to use in tests that may panic — cleanup always runs.
    pub struct EnvGuard {
        key: &'static str,
        had_value: Option<String>,
    }

    impl EnvGuard {
        /// Set an environment variable and return a guard that restores the old value.
        pub fn set(key: &'static str, value: &str) -> Self {
            let had_value = std::env::var(key).ok();
            unsafe { std::env::set_var(key, value) };
            Self { key, had_value }
        }

        /// Remove an environment variable and return a guard that restores the old value.
        pub fn remove(key: &'static str) -> Self {
            let had_value = std::env::var(key).ok();
            unsafe { std::env::remove_var(key) };
            Self { key, had_value }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            match &self.had_value {
                Some(v) => unsafe { std::env::set_var(self.key, v) },
                None => unsafe { std::env::remove_var(self.key) },
            }
        }
    }
}
