//! Agent swarms (PRD §3.10): a bounded group of agents executing tasks with
//! bounded concurrency and aggregated results.

use std::sync::Arc;

use ai_errors::{AgentError, AiError};
use ai_runtime::Parallel;

use crate::agent::{Agent, AgentResult};

/// The outcome of a swarm run.
#[derive(Debug, Clone)]
pub struct SwarmResult {
    /// Results keyed by input index, in input order.
    pub results: Vec<AgentResult>,
    pub succeeded: usize,
    pub failed: usize,
}

/// A bounded swarm of agents (a single agent template executed over many
/// inputs, PRD §3.10.1).
pub struct AgentSwarm {
    agent: Arc<Agent>,
    /// Maximum concurrent agent executions.
    pub concurrency: usize,
}

impl AgentSwarm {
    pub fn new(agent: Arc<Agent>) -> Self {
        Self {
            agent,
            concurrency: 4,
        }
    }

    pub fn with_concurrency(mut self, concurrency: usize) -> Self {
        self.concurrency = concurrency.max(1);
        self
    }

    /// Runs the agent over every input with bounded concurrency.
    pub async fn run(&self, inputs: Vec<String>) -> Result<SwarmResult, AiError> {
        let tasks: Vec<ai_runtime::Task<AgentResult>> = inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                let agent = self.agent.clone();
                let input = input.clone();
                ai_runtime::Task::new(
                    format!("swarm:{index}"),
                    async move { agent.run(&input).await },
                )
            })
            .collect();

        let results = Parallel::new()
            .with_limit(self.concurrency)
            .execute(tasks)
            .await;

        let mut outcomes = Vec::with_capacity(results.len());
        let mut succeeded = 0;
        for result in results {
            match result.outcome {
                Ok(agent_result) => {
                    succeeded += 1;
                    outcomes.push(agent_result);
                }
                Err(e) => {
                    return Err(AiError::Agent(AgentError::with_source(
                        "swarm",
                        format!("swarm task `{}` failed", result.name),
                        e,
                    )));
                }
            }
        }
        Ok(SwarmResult {
            results: outcomes,
            succeeded,
            failed: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::tests::{ScriptedModel, completion};
    use ai_types::{Message, Role};

    #[tokio::test]
    async fn swarm_runs_all_inputs_bounded() {
        let model = Arc::new(ScriptedModel::new(vec![completion("done", vec![]); 8]));
        let agent = Arc::new(crate::AgentBuilder::new("swarm-agent", "be concise", model).build());
        let swarm = AgentSwarm::new(agent).with_concurrency(2);

        let inputs: Vec<String> = (0..8).map(|i| format!("task {i}")).collect();
        let result = swarm.run(inputs).await.unwrap();
        assert_eq!(result.results.len(), 8);
        assert_eq!(result.succeeded, 8);
        assert_eq!(result.failed, 0);
        assert!(result.results.iter().all(|r| r.text.contains("done")));
        let _ = Message::text(Role::User, "unused");
    }
}
