// Raw SSE probe: prints every data: line's content/reasoning_content, bypassing our adapter entirely.
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base_url = std::env::var("AI_SDK_GATEWAY_BASE_URL")?;
    let api_key = std::env::var("AI_SDK_GATEWAY_API_KEY")?;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{base_url}/chat/completions"))
        .bearer_auth(&api_key)
        .json(&serde_json::json!({
            "model": "deepseek-v4-flash",
            "messages": [{"role": "user", "content": "Reply with exactly: STREAMING-OK"}],
            "stream": true
        }))
        .send()
        .await?;
    let mut bytes = resp.bytes_stream();
    use futures::StreamExt;
    let mut buf: Vec<u8> = Vec::new();
    let mut text_parts: Vec<String> = Vec::new();
    let mut n = 0usize;
    while let Some(chunk) = bytes.next().await {
        let chunk = chunk?;
        buf.extend_from_slice(&chunk);
        while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line).trim().to_string();
            if let Some(data) = line.strip_prefix("data: ") {
                n += 1;
                if data == "[DONE]" {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    let content = v
                        .pointer("/choices/0/delta/content")
                        .and_then(|c| c.as_str())
                        .unwrap_or("");
                    let reasoning = v
                        .pointer("/choices/0/delta/reasoning_content")
                        .and_then(|c| c.as_str())
                        .unwrap_or("");
                    let finish = v
                        .pointer("/choices/0/finish_reason")
                        .and_then(|c| c.as_str())
                        .unwrap_or("");
                    if !content.is_empty() {
                        text_parts.push(content.to_string());
                    }
                    if !finish.is_empty() {
                        eprintln!("FINISH: {finish}");
                    }
                    let _ = reasoning;
                }
            }
        }
    }
    eprintln!("RAW SSE events: {n}");
    eprintln!("RAW content parts: {text_parts:?}");
    eprintln!("RAW full text: {:?}", text_parts.concat());
    Ok(())
}
