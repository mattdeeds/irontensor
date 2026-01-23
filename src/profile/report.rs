//! Profile report generation and formatting.

use std::collections::HashMap;
use std::time::Duration;

use super::categories::{OpCategory, Phase};
use crate::logging::{LayerTimingRecord, MatmulShapeRecord, OpTimingRecord, ProfileReportRecord};

/// Statistics for a single operation type.
#[derive(Clone, Debug)]
pub struct OpStat {
    /// Operation name (may include context prefix, e.g., "attn.qkv.Matmul")
    pub name: String,
    pub total_time: Duration,
    pub count: usize,
    pub total_elements: usize,
}

impl OpStat {
    pub fn new(category: OpCategory) -> Self {
        Self {
            name: category.short_name(),
            total_time: Duration::ZERO,
            count: 0,
            total_elements: 0,
        }
    }

    pub fn new_with_name(name: String) -> Self {
        Self {
            name,
            total_time: Duration::ZERO,
            count: 0,
            total_elements: 0,
        }
    }

    pub fn record(&mut self, duration: Duration, elements: usize) {
        self.total_time += duration;
        self.count += 1;
        self.total_elements += elements;
    }

    pub fn avg_time(&self) -> Duration {
        if self.count > 0 {
            self.total_time / self.count as u32
        } else {
            Duration::ZERO
        }
    }
}

/// Layer timing breakdown.
#[derive(Clone, Debug, Default)]
pub struct LayerTiming {
    pub forward: Duration,
    pub backward: Duration,
}

impl LayerTiming {
    pub fn total(&self) -> Duration {
        self.forward + self.backward
    }
}

/// Statistics for matmul operations grouped by shape.
#[derive(Clone, Debug, Default)]
pub struct MatmulShapeStat {
    pub total_time: Duration,
    pub count: usize,
}

impl MatmulShapeStat {
    pub fn avg_time(&self) -> Duration {
        if self.count > 0 {
            self.total_time / self.count as u32
        } else {
            Duration::ZERO
        }
    }
}

/// Complete profiling report.
#[derive(Clone, Debug)]
pub struct ProfileReport {
    pub total_time: Duration,
    pub steps_recorded: usize,
    pub phase_breakdown: HashMap<Phase, Duration>,
    pub layer_breakdown: Vec<LayerTiming>,
    pub op_stats: Vec<OpStat>,
    /// Matmul statistics grouped by shape (e.g., "[4096,256]x[256,512]")
    pub matmul_by_shape: HashMap<String, MatmulShapeStat>,
}

impl ProfileReport {
    /// Print a formatted report to stdout.
    pub fn print(&self) {
        let divider = "=".repeat(80);
        let thin_divider = "-".repeat(80);

        println!("{}", divider);
        println!("{:^80}", "IronTensor Profiling Report");
        println!("{}", divider);

        // Summary
        let total_ms = self.total_time.as_secs_f64() * 1000.0;
        let avg_step_ms = if self.steps_recorded > 0 {
            total_ms / self.steps_recorded as f64
        } else {
            0.0
        };
        println!(
            "Total Time: {:.2}s | Steps: {} | Avg Step: {:.2}ms",
            self.total_time.as_secs_f64(),
            self.steps_recorded,
            avg_step_ms
        );
        println!();

        // Phase breakdown
        println!("Phase Breakdown:");
        let phases = [Phase::Forward, Phase::Backward, Phase::Optimizer];
        for phase in phases {
            if let Some(&duration) = self.phase_breakdown.get(&phase) {
                let ms = duration.as_secs_f64() * 1000.0 / self.steps_recorded.max(1) as f64;
                let pct = if self.total_time.as_nanos() > 0 {
                    duration.as_secs_f64() / self.total_time.as_secs_f64() * 100.0
                } else {
                    0.0
                };
                println!("  {:15} {:8.2}ms ({:5.1}%)", format!("{}:", phase), ms, pct);
            }
        }
        println!();

        // Layer breakdown (if available)
        if !self.layer_breakdown.is_empty() {
            println!("Layer Breakdown (average per step):");
            println!(
                "  {:>5} | {:>10} | {:>10} | {:>10}",
                "Layer", "Forward", "Backward", "Total"
            );
            println!("  {}+{}+{}+{}", "-".repeat(5), "-".repeat(12), "-".repeat(12), "-".repeat(12));

            for (i, timing) in self.layer_breakdown.iter().enumerate() {
                let fwd_ms =
                    timing.forward.as_secs_f64() * 1000.0 / self.steps_recorded.max(1) as f64;
                let bwd_ms =
                    timing.backward.as_secs_f64() * 1000.0 / self.steps_recorded.max(1) as f64;
                let total_ms = fwd_ms + bwd_ms;
                println!(
                    "  {:>5} | {:>8.2}ms | {:>8.2}ms | {:>8.2}ms",
                    i, fwd_ms, bwd_ms, total_ms
                );
            }
            println!();
        }

        // Top operations by time
        println!("Top Operations by Time:");
        println!(
            "  {:>4} | {:20} | {:>10} | {:>8} | {:>10} | {:>8}",
            "Rank", "Operation", "Total", "Count", "Avg", "% Total"
        );
        println!("{}", thin_divider);

        // Sort operations by total time (descending)
        let mut sorted_ops: Vec<_> = self.op_stats.iter().collect();
        sorted_ops.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        for (rank, op) in sorted_ops.iter().take(15).enumerate() {
            let total_secs = op.total_time.as_secs_f64();
            let avg_ms = op.avg_time().as_secs_f64() * 1000.0;
            let pct = if self.total_time.as_nanos() > 0 {
                total_secs / self.total_time.as_secs_f64() * 100.0
            } else {
                0.0
            };

            let time_str = if total_secs >= 1.0 {
                format!("{:.2}s", total_secs)
            } else {
                format!("{:.1}ms", total_secs * 1000.0)
            };

            println!(
                "  {:>4} | {:20} | {:>10} | {:>8} | {:>8.3}ms | {:>7.1}%",
                rank + 1,
                &op.name,
                time_str,
                op.count,
                avg_ms,
                pct
            );
        }
        println!("{}", divider);

        // Print matmul shape breakdown if available
        if !self.matmul_by_shape.is_empty() {
            println!();
            println!("Matmul by Shape:");
            println!(
                "  {:>4} | {:40} | {:>10} | {:>8} | {:>10}",
                "Rank", "Shape", "Total", "Count", "Avg"
            );
            println!("{}", thin_divider);

            let mut sorted_shapes: Vec<_> = self.matmul_by_shape.iter().collect();
            sorted_shapes.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

            for (rank, (shape, stat)) in sorted_shapes.iter().take(10).enumerate() {
                let total_secs = stat.total_time.as_secs_f64();
                let avg_ms = stat.avg_time().as_secs_f64() * 1000.0;

                let time_str = if total_secs >= 1.0 {
                    format!("{:.2}s", total_secs)
                } else {
                    format!("{:.1}ms", total_secs * 1000.0)
                };

                println!(
                    "  {:>4} | {:40} | {:>10} | {:>8} | {:>8.3}ms",
                    rank + 1,
                    shape,
                    time_str,
                    stat.count,
                    avg_ms
                );
            }
            println!("{}", divider);
        }
    }

    /// Convert to a serializable record for logging.
    pub fn to_record(&self) -> ProfileReportRecord {
        let total_time_ms = self.total_time.as_secs_f64() as f32 * 1000.0;
        let avg_step_ms = if self.steps_recorded > 0 {
            total_time_ms / self.steps_recorded as f32
        } else {
            0.0
        };

        // Convert phase breakdown
        let phase_breakdown: HashMap<String, f32> = self
            .phase_breakdown
            .iter()
            .map(|(phase, duration)| {
                let ms = duration.as_secs_f64() as f32 * 1000.0 / self.steps_recorded.max(1) as f32;
                (format!("{}", phase), ms)
            })
            .collect();

        // Convert layer breakdown
        let layer_breakdown: Vec<LayerTimingRecord> = self
            .layer_breakdown
            .iter()
            .enumerate()
            .map(|(i, timing)| {
                let fwd_ms = timing.forward.as_secs_f64() as f32 * 1000.0
                    / self.steps_recorded.max(1) as f32;
                let bwd_ms = timing.backward.as_secs_f64() as f32 * 1000.0
                    / self.steps_recorded.max(1) as f32;
                LayerTimingRecord {
                    layer: i,
                    forward_ms: fwd_ms,
                    backward_ms: bwd_ms,
                    total_ms: fwd_ms + bwd_ms,
                }
            })
            .collect();

        // Convert top operations (sorted by time, top 15)
        let mut sorted_ops: Vec<_> = self.op_stats.iter().collect();
        sorted_ops.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        let top_operations: Vec<OpTimingRecord> = sorted_ops
            .iter()
            .take(15)
            .map(|op| OpTimingRecord {
                op: op.name.clone(),
                time_ms: op.total_time.as_secs_f64() as f32 * 1000.0,
                count: op.count,
                avg_ms: op.avg_time().as_secs_f64() as f32 * 1000.0,
            })
            .collect();

        // Convert matmul shape statistics
        let matmul_by_shape = if self.matmul_by_shape.is_empty() {
            None
        } else {
            let mut sorted_shapes: Vec<_> = self.matmul_by_shape.iter().collect();
            sorted_shapes.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));

            Some(
                sorted_shapes
                    .iter()
                    .take(10)
                    .map(|(shape, stat)| MatmulShapeRecord {
                        shape: (*shape).clone(),
                        time_ms: stat.total_time.as_secs_f64() as f32 * 1000.0,
                        count: stat.count,
                        avg_ms: stat.avg_time().as_secs_f64() as f32 * 1000.0,
                    })
                    .collect(),
            )
        };

        ProfileReportRecord {
            total_time_ms,
            steps_recorded: self.steps_recorded,
            avg_step_ms,
            phase_breakdown,
            layer_breakdown,
            top_operations,
            matmul_by_shape,
        }
    }

    /// Convert the report to a JSON string.
    pub fn to_json(&self) -> String {
        let mut json = String::from("{\n");

        // Summary
        json.push_str(&format!(
            "  \"total_time_ms\": {:.3},\n",
            self.total_time.as_secs_f64() * 1000.0
        ));
        json.push_str(&format!("  \"steps_recorded\": {},\n", self.steps_recorded));

        // Phase breakdown
        json.push_str("  \"phase_breakdown\": {\n");
        let phases: Vec<_> = self.phase_breakdown.iter().collect();
        for (i, (phase, duration)) in phases.iter().enumerate() {
            let comma = if i < phases.len() - 1 { "," } else { "" };
            json.push_str(&format!(
                "    \"{}\": {:.3}{}\n",
                phase,
                duration.as_secs_f64() * 1000.0,
                comma
            ));
        }
        json.push_str("  },\n");

        // Layer breakdown
        json.push_str("  \"layer_breakdown\": [\n");
        for (i, timing) in self.layer_breakdown.iter().enumerate() {
            let comma = if i < self.layer_breakdown.len() - 1 {
                ","
            } else {
                ""
            };
            json.push_str(&format!(
                "    {{\"forward_ms\": {:.3}, \"backward_ms\": {:.3}}}{}\n",
                timing.forward.as_secs_f64() * 1000.0,
                timing.backward.as_secs_f64() * 1000.0,
                comma
            ));
        }
        json.push_str("  ],\n");

        // Op stats
        json.push_str("  \"operations\": [\n");
        for (i, op) in self.op_stats.iter().enumerate() {
            let comma = if i < self.op_stats.len() - 1 { "," } else { "" };
            json.push_str(&format!(
                "    {{\"name\": \"{}\", \"total_ms\": {:.3}, \"count\": {}, \"avg_ms\": {:.3}}}{}\n",
                op.name,
                op.total_time.as_secs_f64() * 1000.0,
                op.count,
                op.avg_time().as_secs_f64() * 1000.0,
                comma
            ));
        }
        json.push_str("  ],\n");

        // Matmul by shape
        json.push_str("  \"matmul_by_shape\": [\n");
        let mut sorted_shapes: Vec<_> = self.matmul_by_shape.iter().collect();
        sorted_shapes.sort_by(|a, b| b.1.total_time.cmp(&a.1.total_time));
        for (i, (shape, stat)) in sorted_shapes.iter().take(10).enumerate() {
            let comma = if i < sorted_shapes.len().min(10) - 1 { "," } else { "" };
            json.push_str(&format!(
                "    {{\"shape\": \"{}\", \"total_ms\": {:.3}, \"count\": {}, \"avg_ms\": {:.3}}}{}\n",
                shape,
                stat.total_time.as_secs_f64() * 1000.0,
                stat.count,
                stat.avg_time().as_secs_f64() * 1000.0,
                comma
            ));
        }
        json.push_str("  ]\n");

        json.push_str("}\n");
        json
    }
}
