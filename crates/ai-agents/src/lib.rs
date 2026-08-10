//! Agent runtime (PRD §3.3, spec §9): lifecycle states, tool loops,
//! sub-agents, HITL hooks, event emission, and self-healing retries.
//!
//! The loop is explicit and observable: build messages → generate → execute
//! tool calls → repeat until no tool calls or the iteration cap. Every step
//! emits an [`ExecutionEvent`] when a collector is attached.

mod agent;
mod swarm;

pub use agent::{Agent, AgentBuilder, AgentResult, AgentState, HumanInTheLoop};
pub use swarm::{AgentSwarm, SwarmResult};

/// Convenience: run an agent tool loop with sub-agent support.
pub use agent::SubAgentTool;

#[cfg(test)]
mod tests {
    // Unit tests use a deterministic in-memory model (per ADR-007: unit
    // tests mock the LLM; live tests use the real gateway).
}
