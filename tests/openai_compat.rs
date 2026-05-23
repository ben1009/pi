use pi_rs::llm::{
    ChatRequest, LlmClient, Message, Role, ToolCall, ToolCallFunction,
    openai_compat::OpenAiCompatClient,
};
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, Request, ResponseTemplate,
    matchers::{method, path},
};

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
        .respond_with(
            ResponseTemplate::new(200).set_body_json(assistant_tool_call(
                "call_1",
                "bash",
                "{\"command\":\"echo hi\"}",
            )),
        )
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
            tools: vec![pi_rs::llm::ToolDef {
                kind: "function",
                function: pi_rs::llm::ToolDefFunction {
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
            tools: vec![pi_rs::llm::ToolDef {
                kind: "function",
                function: pi_rs::llm::ToolDefFunction {
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
    assert!(
        has_tool_role,
        "second request must include a tool-role message"
    );
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
    assert!(
        msg.contains(&server.uri()),
        "error should include URL: {msg}"
    );
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

fn sse_content_delta(text: &str) -> String {
    format!(
        "data: {}\n\n",
        serde_json::json!({
            "choices": [{"delta": {"content": text}, "finish_reason": null}]
        })
    )
}

fn sse_done() -> &'static str {
    "data: [DONE]\n\n"
}

#[tokio::test]
async fn streaming_content() {
    let server = MockServer::start().await;
    let body = format!(
        "{}{}{}",
        sse_content_delta("hello "),
        sse_content_delta("world"),
        sse_done()
    );
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let client = OpenAiCompatClient::new(server.uri(), "test-key".into());
    let mut stream = client
        .complete_stream(req("m", vec![Message::user("hi")]))
        .await
        .unwrap();

    use futures_util::StreamExt;
    let mut texts = Vec::new();
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            pi_rs::llm::StreamEvent::ContentDelta(t) => texts.push(t),
            pi_rs::llm::StreamEvent::Done { .. } => break,
            _ => {}
        }
    }
    assert_eq!(texts.join(""), "hello world");
}

#[tokio::test]
async fn streaming_tool_calls() {
    let server = MockServer::start().await;
    let body = format!(
        "{}{}{}{}",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"function\":{\"name\":\"bash\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"comm\"}}]},\"finish_reason\":null}]}\n\n",
        "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"and\\\":\\\"ls\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
        sse_done()
    );
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let client = OpenAiCompatClient::new(server.uri(), "test-key".into());
    let mut stream = client
        .complete_stream(req("m", vec![Message::user("hi")]))
        .await
        .unwrap();

    use futures_util::StreamExt;
    let mut tool_calls: std::collections::BTreeMap<usize, (String, String, String)> =
        std::collections::BTreeMap::new();
    while let Some(event) = stream.next().await {
        match event.unwrap() {
            pi_rs::llm::StreamEvent::ToolCallDelta {
                index,
                id,
                function_name,
                arguments_delta,
            } => {
                let entry = tool_calls.entry(index).or_default();
                if let Some(id) = id {
                    entry.0 = id;
                }
                if let Some(name) = function_name {
                    entry.1 = name;
                }
                if let Some(args) = arguments_delta {
                    entry.2.push_str(&args);
                }
            }
            pi_rs::llm::StreamEvent::Done { .. } => break,
            _ => {}
        }
    }
    assert_eq!(tool_calls.len(), 1);
    let (id, name, args) = tool_calls.get(&0).unwrap();
    assert_eq!(id, "call_1");
    assert_eq!(name, "bash");
    assert!(args.contains("ls"));
}

#[tokio::test]
async fn streaming_error() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(500).set_body_string("server error"))
        .mount(&server)
        .await;

    let client = OpenAiCompatClient::new(server.uri(), "test-key".into());
    let result = client
        .complete_stream(req("m", vec![Message::user("hi")]))
        .await;
    assert!(result.is_err());
}

#[tokio::test]
async fn streaming_no_choices() {
    let server = MockServer::start().await;
    let body = format!(
        "data: {}\n\n{}",
        serde_json::json!({"choices": []}),
        sse_done()
    );
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(body)
                .insert_header("content-type", "text/event-stream"),
        )
        .mount(&server)
        .await;

    let client = OpenAiCompatClient::new(server.uri(), "test-key".into());
    let mut stream = client
        .complete_stream(req("m", vec![Message::user("hi")]))
        .await
        .unwrap();

    use futures_util::StreamExt;
    let mut found_done = false;
    while let Some(event) = stream.next().await {
        if matches!(event.unwrap(), pi_rs::llm::StreamEvent::Done { .. }) {
            found_done = true;
            break;
        }
    }
    assert!(found_done);
}

#[tokio::test]
async fn complete_no_choices_errors() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": []
        })))
        .mount(&server)
        .await;

    let client = OpenAiCompatClient::new(server.uri(), "test-key".into());
    let err = client
        .complete(req("m", vec![Message::user("hi")]))
        .await
        .expect_err("should error on empty choices");
    assert!(format!("{err:#}").contains("no choices"));
}

#[tokio::test]
async fn complete_with_usage() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(json!({
            "choices": [{
                "message": { "role": "assistant", "content": "ok" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 }
        })))
        .mount(&server)
        .await;

    let client = OpenAiCompatClient::new(server.uri(), "test-key".into());
    let resp = client
        .complete(req("m", vec![Message::user("hi")]))
        .await
        .unwrap();
    assert!(resp.usage.is_some());
    assert_eq!(resp.usage.unwrap().total_tokens, 15);
}

#[tokio::test]
async fn complete_sends_bearer_auth() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/chat/completions"))
        .respond_with(ResponseTemplate::new(200).set_body_json(assistant_text("ok")))
        .mount(&server)
        .await;

    let client = OpenAiCompatClient::new(server.uri(), "my-secret-key".into());
    client
        .complete(req("m", vec![Message::user("hi")]))
        .await
        .unwrap();

    let requests = server.received_requests().await.unwrap();
    let auth = requests[0]
        .headers
        .get("authorization")
        .unwrap()
        .to_str()
        .unwrap();
    assert_eq!(auth, "Bearer my-secret-key");
}
