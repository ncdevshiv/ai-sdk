//! Parallel execution engine for the AI SDK.
//!
//! First-class concurrency control — never blind task spawning:
//!
//! - [`RetryPolicy`] / [`retry`] — error-class-aware retries with
//!   exponential backoff and jitter.
//! - [`ConcurrencyLimiter`] — bounded concurrency keyed by provider/model/
//!   tool (or any logical resource).
//! - [`Parallel`] — fan-out/fan-in with bounded concurrency, per-task
//!   deadlines, cancellation, partial results and aggregation.
//! - [`race`] / [`fallback`] — first-success and provider-fallback
//!   strategies.
//! - [`CircuitBreaker`] — failure thresholds, recovery timeout, half-open
//!   probing.
//!
//! All operations are cancellation-safe: dropping the future stops the work
//! (timeouts and cancellations propagate through `tokio`).

mod circuit_breaker;
mod concurrency;
mod parallel;
mod retry;

pub use circuit_breaker::{CircuitBreaker, CircuitState};
pub use concurrency::{ConcurrencyLimiter, Permit};
pub use parallel::{Parallel, ParallelResult, Task, fallback, race};
pub use retry::{RetryPolicy, backoff_delay, retry};

/// Convenience alias for the runtime result type.
pub type Result<T, E = ai_errors::AiError> = ai_errors::Result<T, E>;
