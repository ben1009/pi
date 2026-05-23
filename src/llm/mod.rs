use std::pin::Pin;

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

pub mod openai_compat;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: Role::System,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "tool_call_type")]
    pub kind: String,
    pub function: ToolCallFunction,
}

fn tool_call_type() -> String {
    "function".to_owned()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ToolDefFunction,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefFunction {
    pub name: &'static str,
    pub description: &'static str,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub max_tokens: u32,
}

#[derive(Debug, Clone)]
#[allow(dead_code)] // finish_reason / usage consumed in M2 (tool loop, /tokens).
pub struct ChatResponse {
    pub message: Message,
    pub finish_reason: String,
    pub usage: Option<Usage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// A single event from a streaming chat completion.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A chunk of assistant text content.
    ContentDelta(String),
    /// A tool call being built (index, id, function name, argument chunk).
    ToolCallDelta {
        index: usize,
        id: Option<String>,
        function_name: Option<String>,
        arguments_delta: Option<String>,
    },
    /// Stream finished with a stop reason.
    Done {
        finish_reason: String,
        usage: Option<Usage>,
    },
    /// An error occurred during streaming.
    Error(String),
}

pub type EventStream = Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>;

#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn complete(&self, req: ChatRequest) -> Result<ChatResponse>;
    async fn complete_stream(&self, req: ChatRequest) -> Result<EventStream>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_system_constructor() {
        let msg = Message::system("you are helpful");
        assert!(matches!(msg.role, Role::System));
        assert_eq!(msg.content.as_deref(), Some("you are helpful"));
        assert!(msg.tool_calls.is_none());
        assert!(msg.tool_call_id.is_none());
    }

    #[test]
    fn message_user_constructor() {
        let msg = Message::user("hello");
        assert!(matches!(msg.role, Role::User));
        assert_eq!(msg.content.as_deref(), Some("hello"));
    }

    #[test]
    fn message_system_accepts_string() {
        let msg = Message::system(String::from("test"));
        assert_eq!(msg.content.as_deref(), Some("test"));
    }

    #[test]
    fn role_serialization_roundtrip() {
        let roles = vec![Role::System, Role::User, Role::Assistant, Role::Tool];
        for role in roles {
            let json = serde_json::to_string(&role).unwrap();
            let deserialized: Role = serde_json::from_str(&json).unwrap();
            assert_eq!(format!("{:?}", role), format!("{:?}", deserialized));
        }
    }

    #[test]
    fn message_serialization_roundtrip() {
        let msg = Message::user("test message");
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.content, msg.content);
        assert!(matches!(deserialized.role, Role::User));
    }

    #[test]
    fn message_with_tool_calls_serializes() {
        let msg = Message {
            role: Role::Assistant,
            content: None,
            tool_calls: Some(vec![ToolCall {
                id: "call_123".to_owned(),
                kind: "function".to_owned(),
                function: ToolCallFunction {
                    name: "bash".to_owned(),
                    arguments: "{\"command\":\"ls\"}".to_owned(),
                },
            }]),
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("call_123"));
        assert!(json.contains("bash"));
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.tool_calls.unwrap().len(), 1);
    }

    #[test]
    fn tool_call_type_default() {
        assert_eq!(tool_call_type(), "function");
    }

    #[test]
    fn usage_serialization() {
        let usage = Usage {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };
        let json = serde_json::to_string(&usage).unwrap();
        let deserialized: Usage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.total_tokens, 150);
    }

    #[test]
    fn stream_event_content_delta() {
        let event = StreamEvent::ContentDelta("hello".to_owned());
        match event {
            StreamEvent::ContentDelta(s) => assert_eq!(s, "hello"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn stream_event_done() {
        let event = StreamEvent::Done {
            finish_reason: "stop".to_owned(),
            usage: None,
        };
        match event {
            StreamEvent::Done {
                finish_reason,
                usage,
            } => {
                assert_eq!(finish_reason, "stop");
                assert!(usage.is_none());
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn stream_event_error() {
        let event = StreamEvent::Error("something went wrong".to_owned());
        match event {
            StreamEvent::Error(s) => assert_eq!(s, "something went wrong"),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn tool_def_serialization() {
        let def = ToolDef {
            kind: "function",
            function: ToolDefFunction {
                name: "bash",
                description: "run a command",
                parameters: serde_json::json!({"type": "object"}),
            },
        };
        let json = serde_json::to_string(&def).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("bash"));
    }
}
