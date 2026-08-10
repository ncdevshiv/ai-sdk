//! Bounded concurrency keyed by logical resource (provider/model/tool).

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use ai_errors::{AiError, InternalError};

/// Maximum number of distinct keys tracked simultaneously. When exceeded,
/// the least-recently-created keys are dropped (their semaphores are
/// re-created on next use).
const MAX_KEYS: usize = 1024;

/// A permit guarding one unit of a resource's concurrency budget.
pub struct Permit {
    _permit: Option<OwnedSemaphorePermit>,
}

/// Bounded concurrency keyed by resource name.
///
/// `limit = 0` means unlimited (no semaphore). The limiter itself is cheap:
/// semaphores are created lazily per key and shared via `Arc`.
#[derive(Debug, Clone, Default)]
pub struct ConcurrencyLimiter {
    semaphores: Arc<Mutex<HashMap<String, Arc<Semaphore>>>>,
    limits: Arc<Mutex<HashMap<String, usize>>>,
}

impl ConcurrencyLimiter {
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the concurrency limit for `key`. Existing permits are unaffected;
    /// subsequent acquisitions use the new limit.
    pub fn set_limit(&self, key: impl Into<String>, limit: usize) {
        let key = key.into();
        self.limits.lock().insert(key.clone(), limit);
        if limit == 0 {
            // Unlimited: no semaphore needed.
            self.semaphores.lock().remove(&key);
        } else {
            self.semaphores
                .lock()
                .insert(key, Arc::new(Semaphore::new(limit)));
        }
    }

    /// Acquires a permit for `key`, waiting if the budget is exhausted.
    ///
    /// Returns `Err` only on internal errors (e.g. semaphore closed).
    pub async fn acquire(&self, key: &str) -> Result<Permit, AiError> {
        let limit = self.limits.lock().get(key).copied().unwrap_or(0);
        if limit == 0 {
            return Ok(Permit { _permit: None });
        }

        let semaphore = {
            let mut map = self.semaphores.lock();
            let sem = map
                .get(key)
                .cloned()
                .unwrap_or_else(|| Arc::new(Semaphore::new(limit)));
            if map.len() >= MAX_KEYS && !map.contains_key(key) {
                // Evict an arbitrary key to bound memory.
                if let Some(oldest) = map.keys().next().cloned() {
                    map.remove(&oldest);
                }
            }
            map.entry(key.to_string())
                .or_insert_with(|| Arc::new(Semaphore::new(limit)));
            sem
        };

        let permit =
            semaphore.clone().acquire_owned().await.map_err(|_| {
                AiError::Internal(InternalError::new("concurrency semaphore closed"))
            })?;
        Ok(Permit {
            _permit: Some(permit),
        })
    }

    /// Current concurrency limit for `key` (0 = unlimited).
    pub fn limit(&self, key: &str) -> usize {
        self.limits.lock().get(key).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[tokio::test]
    async fn limit_is_enforced() {
        let limiter = ConcurrencyLimiter::new();
        limiter.set_limit("model:gpt-4o", 2);

        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let limiter = limiter.clone();
            let active = active.clone();
            let max_active = max_active.clone();
            handles.push(tokio::spawn(async move {
                let _permit = limiter.acquire("model:gpt-4o").await.unwrap();
                let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                max_active.fetch_max(now, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(10)).await;
                active.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert!(
            max_active.load(Ordering::SeqCst) <= 2,
            "limit exceeded: {}",
            max_active.load(Ordering::SeqCst)
        );
    }

    #[tokio::test]
    async fn unlimited_when_zero() {
        let limiter = ConcurrencyLimiter::new();
        assert_eq!(limiter.limit("tool:calc"), 0);
        let permit = limiter.acquire("tool:calc").await.unwrap();
        // Zero-cost path: no semaphore involved.
        drop(permit);
        assert_eq!(limiter.limit("tool:calc"), 0);
    }

    #[tokio::test]
    async fn limits_are_independent_per_key() {
        let limiter = ConcurrencyLimiter::new();
        limiter.set_limit("provider:a", 1);
        limiter.set_limit("provider:b", 1);
        let a = limiter.acquire("provider:a").await.unwrap();
        let b = limiter.acquire("provider:b").await.unwrap();
        drop(a);
        drop(b);
    }
}
