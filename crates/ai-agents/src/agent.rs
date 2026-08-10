//! The agent runtime: instructions + model + tools + memory with an
//! explicit tool loop, lifecycle states, HITL hooks, and event emission.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use ai_core::{ChatRequest, Model, ToolDefinition};
use ai_errors::{AgentError, AiError};
use ai_memory::{InProcessMemory, Memory};
use ai_observability::{EventCollector, EventKind, EventStatus};
use ai_runtime::RetryPolicy;
use ai_tools::{Tool, ToolContext, ToolOutput, ToolRegistry, run_tool};
use ai_types::{ContentPart, Message, Role, Usage};

/// The agent's lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    Idle,
    Running,
    AwaitingInput,
    Completed,
    Failed,
}

/// The outcome of an agent run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    pub text: String,
    pub tool_calls_used: usize,
    pub iterations: u32,
    pub state: AgentState,
    pub usage: Usage,
    /// Tool outputs observed during the run (for debugging/audit).
    pub tool_outputs: Vec<String>,
}

/// Human-in-the-loop hook: called before sensitive tool executions.
#[async_trait]
pub trait HumanInTheLoop: Send + Sync {
    async fn request_confirmation(
        &self,
        tool: &str,
        arguments: &serde_json::Value,
    ) -> Result<bool, AiError>;
}

/// No-op HITL hook (all tools auto-approved).
pub struct AutoApprove;

#[async_trait]
impl HumanInTheLoop for AutoApprove {
    async fn request_confirmation(
        &self,
        _tool: &str,
        _arguments: &serde_json::Value,
    ) -> Result<bool, AiError> {
        Ok(true)
    }
}

/// Builder for an [`Agent`].
pub struct AgentBuilder {
    id: String,
    instructions: String,
    model: Arc<dyn Model>,
    tools: ToolRegistry,
    memory: Arc<dyn Memory>,
    max_iterations: u32,
    retry: RetryPolicy,
    collector: Option<EventCollector>,
    hitl: Arc<dyn HumanInTheLoop>,
}

impl AgentBuilder {
    pub fn new(
        id: impl Into<String>,
        instructions: impl Into<String>,
        model: Arc<dyn Model>,
    ) -> Self {
        Self {
            id: id.into(),
            instructions: instructions.into(),
            model,
            tools: ToolRegistry::new(),
            memory: Arc::new(InProcessMemory::new(100)),
            max_iterations: 10,
            retry: RetryPolicy::default(),
            collector: None,
            hitl: Arc::new(AutoApprove),
        }
    }

    pub fn with_tools(mut self, tools: ToolRegistry) -> Self {
        self.tools = tools;
        self
    }

    pub fn with_memory(mut self, memory: Arc<dyn Memory>) -> Self {
        self.memory = memory;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations.max(1);
        self
    }

    pub fn with_retry(mut self, retry: RetryPolicy) -> Self {
        self.retry = retry;
        self
    }

    pub fn with_collector(mut self, collector: EventCollector) -> Self {
        self.collector = Some(collector);
        self
    }

    pub fn with_hitl(mut self, hitl: Arc<dyn HumanInTheLoop>) -> Self {
        self.hitl = hitl;
        self
    }

    pub fn build(self) -> Agent {
        Agent {
            id: self.id,
            instructions: self.instructions,
            model: self.model,
            tools: self.tools,
            memory: self.memory,
            max_iterations: self.max_iterations,
            retry: self.retry,
            collector: self.collector,
            hitl: self.hitl,
        }
    }
}

/// A tool-using agent.
pub struct Agent {
    id: String,
    instructions: String,
    model: Arc<dyn Model>,
    tools: ToolRegistry,
    memory: Arc<dyn Memory>,
    max_iterations: u32,
    retry: RetryPolicy,
    collector: Option<EventCollector>,
    hitl: Arc<dyn HumanInTheLoop>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("id", &self.id)
            .field("instructions", &self.instructions)
            .field("tools", &self.tools.names())
            .field("max_iterations", &self.max_iterations)
            .finish()
    }
}

impl Agent {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn tools(&self) -> &ToolRegistry {
        &self.tools
    }

    fn record(&self, kind: EventKind, operation: impl Into<String>, status: EventStatus) {
        if let Some(collector) = &self.collector {
            collector.record(kind, operation, status, Default::default());
        }
    }

    fn model_definitions(&self) -> Vec<ToolDefinition> {
        self.tools.definitions()
    }

    /// Runs the agent on `user_input` with a fresh conversation.
    pub async fn run(&self, user_input: &str) -> Result<AgentResult, AiError> {
        self.record(
            EventKind::AgentState,
            format!("agent:{}:start", self.id),
            EventStatus::Started,
        );
        let result = self
            .run_with_messages(vec![Message::text(Role::User, user_input)])
            .await;
        match &result {
            Ok(r) => self.record(
                EventKind::AgentState,
                format!(
                    "agent:{}:{}",
                    self.id,
                    match r.state {
                        AgentState::Completed => "completed",
                        AgentState::Failed => "failed",
                        _ => "ended",
                    }
                ),
                if r.state == AgentState::Failed {
                    EventStatus::Failed
                } else {
                    EventStatus::Succeeded
                },
            ),
            Err(_) => self.record(
                EventKind::AgentState,
                format!("agent:{}:error", self.id),
                EventStatus::Failed,
            ),
        }
        result
    }

    /// Runs the agent, continuing from `history` (the last message must be
    /// the new user input).
    pub async fn run_with_messages(&self, history: Vec<Message>) -> Result<AgentResult, AiError> {
        let mut messages = history;
        let mut tool_calls_used = 0usize;
        let mut tool_outputs = Vec::new();
        #[allow(unused_assignments)] // read at the end; overwritten every iteration
        let mut usage: Option<Usage> = None;
        let mut iterations = 0u32;

        let system = Message::text(Role::System, &self.instructions);

        loop {
            iterations += 1;
            if iterations > self.max_iterations {
                return Err(AiError::Agent(AgentError::new(
                    &self.id,
                    format!("exceeded max iterations ({})", self.max_iterations),
                )));
            }

            // Assemble the request: system + memory + conversation.
            let memory_messages = self.memory.retrieve(&self.id).await?;
            let mut request_messages = vec![system.clone()];
            request_messages.extend(memory_messages);
            request_messages.extend(messages.clone());

            let request = ChatRequest::new(request_messages)
                .with_tools(self.model_definitions())
                .with_max_tokens(4096);

            self.record(
                EventKind::ModelCall,
                format!("agent:{}:generate", self.id),
                EventStatus::Started,
            );
            let completion = match self.model.generate(request).await {
                Ok(completion) => completion,
                Err(_err) => {
                    // Self-healing: retry the generation with backoff,
                    // replaying the same request context.
                    self.record(
                        EventKind::Retry,
                        format!("agent:{}:retry", self.id),
                        EventStatus::Retrying,
                    );
                    let retry_request = ChatRequest::new(messages_for_retry(&messages, &system))
                        .with_tools(self.model_definitions())
                        .with_max_tokens(4096);
                    let model = self.model.clone();
                    ai_runtime::retry(
                        &self.retry,
                        &format!("agent:{}:generate", self.id),
                        move || {
                            let request = retry_request.clone();
                            let model = model.clone();
                            async move { model.generate(request).await }
                        },
                    )
                    .await
                    .map_err(|e| {
                        AiError::Agent(AgentError::with_source(
                            &self.id,
                            "generation failed after retries",
                            e,
                        ))
                    })?
                }
            };
            self.record(
                EventKind::ModelCall,
                format!("agent:{}:generate", self.id),
                EventStatus::Succeeded,
            );
            usage = Some(completion.usage);

            // Track the assistant turn.
            let mut assistant_parts = vec![ContentPart::text(&completion.text)];
            for call in &completion.tool_calls {
                assistant_parts.push(ContentPart::tool_call(
                    &call.id,
                    &call.name,
                    &call.arguments,
                ));
            }
            messages.push(Message::new(Role::Assistant, assistant_parts));

            if completion.tool_calls.is_empty() {
                break;
            }

            // Execute tool calls (with HITL + validation).
            for call in &completion.tool_calls {
                tool_calls_used += 1;
                self.record(
                    EventKind::ToolCall,
                    format!("tool:{}", call.name),
                    EventStatus::Started,
                );
                let tool = self.tools.get(&call.name).ok_or_else(|| {
                    AiError::Agent(AgentError::new(
                        &self.id,
                        format!("model requested unknown tool `{}`", call.name),
                    ))
                })?;

                let arguments: serde_json::Value = serde_json::from_str(&call.arguments)
                    .unwrap_or(serde_json::json!({"raw": call.arguments}));

                let approved = self
                    .hitl
                    .request_confirmation(&call.name, &arguments)
                    .await?;
                if !approved {
                    messages.push(Message::new(
                        Role::Tool,
                        vec![ContentPart::tool_result(
                            &call.id,
                            &call.name,
                            r#"{"error":"rejected by human"}"#,
                            true,
                        )],
                    ));
                    continue;
                }

                let context = ToolContext {
                    permissions: ai_security::Permissions::new()
                        .allow("net:http")
                        .allow("fs:read"),
                    execution_id: Some(call.id.clone()),
                    deadline: None,
                    max_response_bytes: Some(16 * 1024),
                };

                let output = match run_tool(tool.as_ref(), arguments, &context).await {
                    Ok(output) => output,
                    Err(e) => ToolOutput::error(e.to_string()),
                };
                tool_outputs.push(output.content.clone());
                messages.push(Message::new(
                    Role::Tool,
                    vec![ContentPart::tool_result(
                        &call.id,
                        &call.name,
                        &output.content,
                        output.is_error,
                    )],
                ));
                self.record(
                    EventKind::ToolCall,
                    format!("tool:{}", call.name),
                    EventStatus::Succeeded,
                );
            }

            // Persist the conversation so far.
            for message in &messages {
                self.memory.store(&self.id, message.clone()).await?;
            }
            messages.clear();
        }

        // Persist the final exchange.
        for message in &messages {
            self.memory.store(&self.id, message.clone()).await?;
        }

        Ok(AgentResult {
            text: messages
                .iter()
                .filter(|&m| m.role == Role::Assistant)
                .map(|m| m.text_content())
                .collect::<Vec<_>>()
                .join("\n"),
            tool_calls_used,
            iterations,
            state: AgentState::Completed,
            usage: usage.unwrap_or_default(),
            tool_outputs,
        })
    }

    /// Exposes this agent as a tool (sub-agent pattern): the caller's model
    /// can delegate a task, and the sub-agent runs with its own memory.
    pub fn as_tool(&self) -> Arc<dyn Tool> {
        Arc::new(SubAgentTool {
            agent: Arc::new(SubAgentHandle::new(self)),
        })
    }
}

/// Builds the full context for a retried generation: system + memory +
/// conversation history.
fn messages_for_retry(messages: &[Message], system: &Message) -> Vec<Message> {
    let mut out = vec![system.clone()];
    out.extend(messages.iter().cloned());
    out
}

/// Shared handle so the sub-agent tool can be cloned.
pub struct SubAgentHandle {
    agent: Agent,
}

impl SubAgentHandle {
    fn new(agent: &Agent) -> Self {
        // Sub-agents share configuration but get their own conversation id.
        Self {
            agent: AgentBuilder::new(
                format!("{}-sub", agent.id),
                &agent.instructions,
                agent.model.clone(),
            )
            .with_tools(agent.tools.clone())
            .with_memory(agent.memory.clone())
            .with_max_iterations(agent.max_iterations)
            .with_retry(agent.retry.clone())
            .build(),
        }
    }
}

/// A tool that delegates to a sub-agent (PRD §3.3 subagent delegation).
pub struct SubAgentTool {
    agent: Arc<SubAgentHandle>,
}

#[async_trait]
impl Tool for SubAgentTool {
    fn name(&self) -> &str {
        "delegate"
    }

    fn description(&self) -> &str {
        "Delegates a task to a sub-agent and returns its final answer"
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "The task for the sub-agent"}
            },
            "required": ["task"]
        })
    }

    async fn execute(
        &self,
        arguments: serde_json::Value,
        _context: &ToolContext,
    ) -> Result<ToolOutput, AiError> {
        let task = arguments
            .get("task")
            .and_then(|t| t.as_str())
            .ok_or_else(|| {
                AiError::Tool(ai_errors::ToolError::new("delegate", "missing `task`"))
            })?;
        let result = self.agent.agent.run(task).await?;
        Ok(ToolOutput::ok(result.text))
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use ai_core::{ChatRequest, Model};
    use ai_models::{ModelCapabilities, ModelInfo};
    use ai_types::{Completion, ModelId, ProviderId, StreamEvent};

    /// Deterministic in-memory model for unit tests (ADR-007: unit tests
    /// use mocked LLMs; live tests use the real gateway).
    pub struct ScriptedModel {
        script: Vec<Completion>,
        index: std::sync::Mutex<usize>,
    }

    impl ScriptedModel {
        pub fn new(script: Vec<Completion>) -> Self {
            Self {
                script,
                index: std::sync::Mutex::new(0),
            }
        }
    }

    pub fn completion(text: &str, tool_calls: Vec<ai_types::ToolCall>) -> Completion {
        Completion {
            provider: ProviderId::new("test"),
            model: ModelId::new("scripted"),
            text: text.to_string(),
            tool_calls,
            usage: Usage::new(10, 5),
            reasoning: None,
            raw: serde_json::Value::Null,
            finish_reason: Some("stop".to_string()),
        }
    }

    #[async_trait]
    impl Model for ScriptedModel {
        fn info(&self) -> &ModelInfo {
            static INFO: std::sync::OnceLock<ModelInfo> = std::sync::OnceLock::new();
            INFO.get_or_init(|| {
                ModelInfo::new(
                    ProviderId::new("test"),
                    ModelId::new("scripted"),
                    128_000,
                    8_192,
                )
                .with_capabilities(ModelCapabilities {
                    supports_tools: true,
                    ..Default::default()
                })
            })
        }

        async fn generate(&self, _request: ChatRequest) -> Result<Completion, AiError> {
            let mut index = self.index.lock().unwrap();
            let completion = self
                .script
                .get(*index)
                .cloned()
                .unwrap_or_else(|| completion("", vec![]));
            *index += 1;
            Ok(completion)
        }

        async fn stream(&self, request: ChatRequest) -> Result<ai_core::EventStream, AiError> {
            let completion = self.generate(request).await?;
            let events = vec![
                Ok(StreamEvent::TextDelta {
                    delta: completion.text.clone(),
                }),
                Ok(StreamEvent::Completed {
                    finish_reason: completion.finish_reason.clone(),
                }),
            ];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    fn calculator_tool() -> Arc<dyn Tool> {
        Arc::new(ai_tools::FunctionTool::new(
            "calculator",
            "Evaluates an arithmetic expression",
            serde_json::json!({
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"]
            }),
            |args| {
                let expression = args["expression"].as_str().unwrap_or("");
                let result = ai_tools::evaluate_expression(expression)
                    .map_err(|e| AiError::Tool(ai_errors::ToolError::new("calculator", e)))?;
                Ok(ToolOutput::ok(result.to_string()))
            },
        ))
    }

    #[tokio::test]
    async fn agent_completes_without_tools() {
        let model = Arc::new(ScriptedModel::new(vec![completion("Hello!", vec![])]));
        let agent = AgentBuilder::new("a1", "Be helpful", model).build();
        let result = agent.run("hi").await.unwrap();
        assert_eq!(result.state, AgentState::Completed);
        assert!(result.text.contains("Hello!"));
        assert_eq!(result.tool_calls_used, 0);
    }

    #[tokio::test]
    async fn agent_executes_tool_loop() {
        let model = Arc::new(ScriptedModel::new(vec![
            completion(
                "",
                vec![ai_types::ToolCall {
                    id: "c1".into(),
                    name: "calculator".into(),
                    arguments: r#"{"expression":"6 * 7"}"#.into(),
                }],
            ),
            completion("The answer is 42.", vec![]),
        ]));
        let mut tools = ToolRegistry::new();
        tools.register(calculator_tool());
        let agent = AgentBuilder::new("a2", "Use tools when needed", model)
            .with_tools(tools)
            .build();
        let result = agent.run("What is 6*7?").await.unwrap();
        assert_eq!(result.tool_calls_used, 1);
        assert!(result.text.contains("42"), "{:?}", result.text);
        assert!(result.tool_outputs.iter().any(|o| o.contains("42")));
    }

    #[tokio::test]
    async fn agent_stops_at_max_iterations() {
        let tool_call = completion(
            "",
            vec![ai_types::ToolCall {
                id: "c1".into(),
                name: "calculator".into(),
                arguments: r#"{"expression":"1+1"}"#.into(),
            }],
        );
        let model = Arc::new(ScriptedModel::new(vec![tool_call.clone(); 3]));
        let mut tools = ToolRegistry::new();
        tools.register(calculator_tool());
        let agent = AgentBuilder::new("a3", "keep calling tools", model)
            .with_tools(tools)
            .with_max_iterations(3)
            .build();
        let err = agent.run("go").await.unwrap_err();
        assert!(err.to_string().contains("max iterations"), "{err}");
    }

    #[tokio::test]
    async fn hitl_rejection_is_recorded_as_tool_result() {
        struct RejectAll;
        #[async_trait]
        impl HumanInTheLoop for RejectAll {
            async fn request_confirmation(
                &self,
                _tool: &str,
                _args: &serde_json::Value,
            ) -> Result<bool, AiError> {
                Ok(false)
            }
        }
        let model = Arc::new(ScriptedModel::new(vec![
            completion(
                "",
                vec![ai_types::ToolCall {
                    id: "c1".into(),
                    name: "calculator".into(),
                    arguments: r#"{"expression":"1+1"}"#.into(),
                }],
            ),
            completion("fine", vec![]),
        ]));
        let mut tools = ToolRegistry::new();
        tools.register(calculator_tool());
        let agent = AgentBuilder::new("a4", "x", model)
            .with_tools(tools)
            .with_hitl(Arc::new(RejectAll))
            .build();
        let result = agent.run("calc").await.unwrap();
        // The model saw the rejection and moved on.
        assert!(result.text.contains("fine"));
    }
}
