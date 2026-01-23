//! RAII timing guard for profiling operations.

use std::time::Instant;

use super::categories::OpCategory;
use super::Profiler;
use super::PROFILER;

/// RAII timing guard that records duration on drop.
pub struct TimedOp {
    category: OpCategory,
    start: Instant,
    elements: usize,
}

impl TimedOp {
    /// Create a new timing guard.
    pub fn new(category: OpCategory, elements: usize) -> Self {
        Self {
            category,
            start: Instant::now(),
            elements,
        }
    }
}

impl Drop for TimedOp {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        PROFILER.with(|p| {
            if let Some(ref mut profiler) = *p.borrow_mut() {
                profiler.record(self.category.clone(), duration, self.elements);
            }
        });
    }
}

/// Create a timing guard if profiling is enabled.
///
/// Returns `Some(TimedOp)` if profiling is enabled, `None` otherwise.
/// The guard records the operation duration when dropped.
///
/// # Example
/// ```ignore
/// let _timer = timed(OpCategory::Matmul, m * n);
/// // ... operation code ...
/// // Duration recorded when _timer goes out of scope
/// ```
pub fn timed(category: OpCategory, elements: usize) -> Option<TimedOp> {
    PROFILER.with(|p| {
        if p.borrow().is_some() {
            Some(TimedOp::new(category, elements))
        } else {
            None
        }
    })
}

/// Check if profiling is currently enabled.
pub fn is_profiling_enabled() -> bool {
    PROFILER.with(|p| p.borrow().is_some())
}

/// RAII guard for profiling context.
///
/// Automatically pops the context tag when dropped.
pub struct ProfileContext {
    enabled: bool,
}

impl ProfileContext {
    /// Create a new context guard that pushes a tag onto the profiler's context stack.
    pub fn new(tag: &str) -> Self {
        let enabled = is_profiling_enabled();
        if enabled {
            Profiler::push_tag(tag);
        }
        Self { enabled }
    }
}

impl Drop for ProfileContext {
    fn drop(&mut self) {
        if self.enabled {
            Profiler::pop_tag();
        }
    }
}

/// Create an RAII context guard for hierarchical profiling.
///
/// When the returned guard is dropped, the context tag is automatically popped.
///
/// # Example
/// ```ignore
/// let _ctx = context("attn");
/// // Operations here are profiled as "attn.OpName"
/// {
///     let _ctx2 = context("qkv");
///     // Operations here are profiled as "attn.qkv.OpName"
/// }
/// // Back to "attn.OpName"
/// ```
pub fn context(tag: &str) -> ProfileContext {
    ProfileContext::new(tag)
}
