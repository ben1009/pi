use std::{collections::HashMap, sync::Arc};

use anyhow::Result;
use async_trait::async_trait;

use crate::llm::{ToolDef, ToolDefFunction};

mod bash;
mod edit;
mod read_file;
mod write_file;

pub use bash::BashTool;
pub use edit::EditTool;
pub use read_file::ReadTool;
pub use write_file::WriteTool;

/// Per-call permissions and limits passed by the agent to each tool invocation.
#[derive(Debug, Clone, Copy)]
pub struct ToolCtx {
    pub yolo: bool,
    pub max_output: usize,
    /// When true, stream tool output to stderr in real-time (REPL mode).
    pub stream_stderr: bool,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &'static str;
    fn description(&self) -> &'static str;
    fn schema(&self) -> serde_json::Value;
    async fn run(&self, ctx: ToolCtx, input: serde_json::Value) -> Result<String>;
}

pub struct Registry {
    tools: HashMap<&'static str, Arc<dyn Tool>>,
}

impl Registry {
    pub fn with_defaults() -> Self {
        let mut tools: HashMap<&'static str, Arc<dyn Tool>> = HashMap::new();
        for t in [
            Arc::new(BashTool) as Arc<dyn Tool>,
            Arc::new(ReadTool),
            Arc::new(WriteTool),
            Arc::new(EditTool),
        ] {
            tools.insert(t.name(), t);
        }
        Self { tools }
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Register a tool. If a tool with the same name already exists, it is replaced.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        let name: &'static str = tool.name();
        self.tools.insert(name, Arc::from(tool));
    }

    pub fn definitions(&self) -> Vec<ToolDef> {
        let mut defs: Vec<_> = self
            .tools
            .values()
            .map(|t| ToolDef {
                kind: "function",
                function: ToolDefFunction {
                    name: t.name(),
                    description: t.description(),
                    parameters: t.schema(),
                },
            })
            .collect();
        // Stable order so LLM tool listings don't churn between runs.
        defs.sort_by_key(|d| d.function.name);
        defs
    }
}

/// UTF-8-safe truncation for tool output. Always reports byte count of the
/// dropped tail so the model knows it was truncated.
pub fn truncate(s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let cut = s
        .char_indices()
        .map(|(i, _)| i)
        .take_while(|&i| i <= max)
        .last()
        .unwrap_or(0);
    let dropped = s.len() - cut;
    let mut out = String::with_capacity(cut + 64);
    out.push_str(&s[..cut]);
    out.push_str(&format!("\n... <truncated, {dropped} more bytes>"));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_short_string_unchanged() {
        let s = "hello".to_owned();
        assert_eq!(truncate(s.clone(), 100), s);
    }

    #[test]
    fn truncate_exact_limit_unchanged() {
        let s = "abcde".to_owned();
        assert_eq!(truncate(s.clone(), 5), s);
    }

    #[test]
    fn truncate_long_string_is_truncated() {
        let s = "a".repeat(200);
        let result = truncate(s, 100);
        assert!(result.len() <= 100 + 64); // cap + suffix
        assert!(result.contains("truncated"));
        assert!(result.contains("more bytes"));
    }

    #[test]
    fn truncate_utf8_safe() {
        // 'ñ' is 2 bytes. Truncation should not split it.
        let s = "a".repeat(99) + "ñ";
        let result = truncate(s, 100);
        // The 'ñ' (2 bytes at position 99-100) should be included since byte 99 <= 100.
        assert!(result.starts_with(&"a".repeat(99)));
    }

    #[test]
    fn registry_with_defaults_has_four_tools() {
        let reg = Registry::with_defaults();
        assert!(reg.get("bash").is_some());
        assert!(reg.get("read").is_some());
        assert!(reg.get("write").is_some());
        assert!(reg.get("edit").is_some());
        assert!(reg.get("nonexistent").is_none());
    }

    #[test]
    fn registry_definitions_sorted_by_name() {
        let reg = Registry::with_defaults();
        let defs = reg.definitions();
        let names: Vec<&str> = defs.iter().map(|d| d.function.name).collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn registry_register_custom_tool() {
        struct Dummy;
        #[async_trait]
        impl Tool for Dummy {
            fn name(&self) -> &'static str {
                "dummy"
            }

            fn description(&self) -> &'static str {
                "a dummy"
            }

            fn schema(&self) -> serde_json::Value {
                serde_json::json!({})
            }

            async fn run(
                &self,
                _ctx: ToolCtx,
                _input: serde_json::Value,
            ) -> anyhow::Result<String> {
                Ok("ran".to_owned())
            }
        }

        let mut reg = Registry::with_defaults();
        reg.register(Box::new(Dummy));
        assert!(reg.get("dummy").is_some());
        assert_eq!(reg.definitions().len(), 5);
    }

    #[tokio::test]
    async fn custom_tool_run() {
        struct Dummy;
        #[async_trait]
        impl Tool for Dummy {
            fn name(&self) -> &'static str {
                "dummy"
            }

            fn description(&self) -> &'static str {
                "a dummy"
            }

            fn schema(&self) -> serde_json::Value {
                serde_json::json!({})
            }

            async fn run(
                &self,
                _ctx: ToolCtx,
                _input: serde_json::Value,
            ) -> anyhow::Result<String> {
                Ok("ran".to_owned())
            }
        }

        let tool = Dummy;
        let ctx = ToolCtx {
            yolo: true,
            max_output: 4096,
            stream_stderr: false,
        };
        let result = tool.run(ctx, serde_json::json!({})).await.unwrap();
        assert_eq!(result, "ran");
    }
}
