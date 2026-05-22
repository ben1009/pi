use std::collections::HashMap;
use std::sync::Arc;

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
