use serde_json::{Value, json};
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

use pi::llm::openai_compat::OpenAiCompatClient;
use pi::llm::{ChatRequest, LlmClient, Message, Role, ToolCall, ToolCallFunction};

fn req(model: &str, messages: Vec<Message>) -> ChatRequest {
    ChatRequest {
        model: model.to_owned(),
        messages,
        tools: vec![],
        max_tokens: 256,
    }
}

fn assistant_text(text: &str) -> Value {
    json!({
        "choices": [{
            "message": { "role": "assistant", "content": text },
            "finish_reason": "stop"
        }]
    })
}

fn assistant_tool_call(id: &str, name: &str, args: &str) -> Value {
    json!({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": null,
                "tool_calls": [{
                    "id": id,
                    "type": "function",
                    "function": { "name": name, "arguments": args }
                }]
            },
            "finish_reason": "tool_calls"
        }]
    })
}

#[tokio::test]
async fn plain_reply() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(assistant_text("hello back")))
        .mount(&server)
        .await;

    let client = OpenAiCompatClient::new(server.uri(), "test-key".into());
    let resp = client
        .complete(req("m", vec![Message::user("hi")]))
        .await
        .expect("complete should succeed");
    assert_eq!(resp.message.content.as_deref(), Some("hello back"));
    assert_eq!(resp.finish_reason, "stop");
}

#[tokio::test]
async fn tool_call_round_trip() {
    let server = MockServer::start().await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(assistant_tool_call(
            "call_1",
            "bash",
            "{\"command\":\"echo hi\"}",
        )))
        .up_to_n_times(1)
        .mount(&server)
        .await;

    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(assistant_text("done")))
        .mount(&server)
        .await;

    let client = OpenAiCompatClient::new(server.uri(), "test-key".into());

    let mut messages = vec![Message::user("run echo hi")];
    let first = client
        .complete(ChatRequest {
            model: "m".into(),
            messages: messages.clone(),
            tools: vec![pi::llm::ToolDef {
                kind: "function",
                function: pi::llm::ToolDefFunction {
                    name: "bash",
                    description: "run a shell command",
                    parameters: json!({"type":"object"}),
                },
            }],
            max_tokens: 256,
        })
        .await
        .expect("first call");
    let calls = first.message.tool_calls.clone().unwrap_or_default();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].function.name, "bash");
    messages.push(Message {
        role: Role::Assistant,
        content: None,
        tool_calls: Some(vec![ToolCall {
            id: calls[0].id.clone(),
            kind: "function".into(),
            function: ToolCallFunction {
                name: "bash".into(),
                arguments: calls[0].function.arguments.clone(),
            },
        }]),
        tool_call_id: None,
    });
    messages.push(Message {
        role: Role::Tool,
        content: Some("hi\n".into()),
        tool_calls: None,
        tool_call_id: Some(calls[0].id.clone()),
    });

    let second = client
        .complete(ChatRequest {
            model: "m".into(),
            messages: messages.clone(),
            tools: vec![pi::llm::ToolDef {
                kind: "function",
                function: pi::llm::ToolDefFunction {
                    name: "bash",
                    description: "run a shell command",
                    parameters: json!({"type":"object"}),
                },
            }],
            max_tokens: 256,
        })
        .await
        .expect("second call");
    assert_eq!(second.message.content.as_deref(), Some("done"));

    let received = server.received_requests().await.expect("requests");
    assert_eq!(received.len(), 2);
    let second_body: Value = parse_body(&received[1]);
    let tools = second_body
        .get("tools")
        .and_then(|t| t.as_array())
        .expect("tools array on second request");
    assert!(!tools.is_empty(), "tools must be present on second request");
    let messages_arr = second_body
        .get("messages")
        .and_then(|m| m.as_array())
        .expect("messages array");
    let has_tool_role = messages_arr
        .iter()
        .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool"));
    assert!(has_tool_role, "second request must include a tool-role message");
}

#[tokio::test]
async fn api_error_body_truncation() {
    let server = MockServer::start().await;
    let big_body = "x".repeat(5_000);
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string(big_body))
        .mount(&server)
        .await;

    let client = OpenAiCompatClient::new(server.uri(), "test-key".into());
    let err = client
        .complete(req("m", vec![Message::user("hi")]))
        .await
        .expect_err("should error on 500");
    let msg = format!("{err:#}");
    assert!(msg.contains(&server.uri()), "error should include URL: {msg}");
    assert!(
        msg.contains("truncated"),
        "error should mention truncation: {msg}"
    );
    assert!(
        msg.len() < 5_000,
        "error message should be shorter than the raw body: len={}",
        msg.len()
    );
}

#[tokio::test]
async fn missing_finish_reason_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": { "role": "assistant", "content": "hi" }
            }]
        })))
        .mount(&server)
        .await;

    let client = OpenAiCompatClient::new(server.uri(), "test-key".into());
    let err = client
        .complete(req("m", vec![Message::user("hi")]))
        .await
        .expect_err("should error when finish_reason missing");
    let msg = format!("{err:#}");
    assert!(
        msg.contains("finish_reason"),
        "error should mention finish_reason: {msg}"
    );
}

fn parse_body(req: &Request) -> Value {
    serde_json::from_slice(&req.body).expect("decode JSON body")
}
