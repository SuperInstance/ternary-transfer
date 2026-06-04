//! Transfer score: measure how much the transfer helped.

use crate::{KnowledgeMatrix, TargetTask};

/// Measures the quality and benefit of a transfer.
#[derive(Debug, Clone, PartialEq)]
pub struct TransferScore {
    /// Estimated performance after transfer (0.0 to 1.0).
    pub estimated_performance: f64,
    /// Improvement over baseline (positive = transfer helped).
    pub improvement: f64,
    /// Relative improvement ratio.
    pub relative_improvement: f64,
    /// Confidence in the score (0.0 to 1.0).
    pub confidence: f64,
}

impl TransferScore {
    /// Compute a transfer score.
    pub fn compute(
        source_performance: f64,
        baseline_performance: f64,
        transferred: &KnowledgeMatrix,
        target: &TargetTask,
    ) -> Self {
        // Estimate performance: weighted sum of transferred knowledge alignment
        // with target feature importance
        let target_importance: Vec<f64> = target.features.iter()
            .map(|f| f.importance.abs())
            .collect();

        let total_importance: f64 = target_importance.iter().sum();
        if total_importance == 0.0 || transferred.feature_count() == 0 {
            return Self {
                estimated_performance: baseline_performance,
                improvement: 0.0,
                relative_improvement: 0.0,
                confidence: 0.0,
            };
        }

        // Alignment: how well transferred weights match target ternary biases
        let mut alignment = 0.0;
        for (i, feat) in target.features.iter().enumerate() {
            if i < transferred.feature_count() {
                let w = transferred.weights()[i];
                let bias = feat.ternary_bias.to_f64();
                // Reward when weight direction matches bias
                alignment += (w * bias) * feat.importance;
            }
        }

        let norm_alignment = alignment / total_importance;
        let estimated = (baseline_performance + 0.3 * norm_alignment + 0.2 * source_performance)
            .clamp(0.0, 1.0);

        let improvement = estimated - baseline_performance;
        let relative = if baseline_performance > 0.0 {
            improvement / baseline_performance
        } else if improvement > 0.0 {
            1.0
        } else {
            0.0
        };

        // Confidence based on how many features overlap
        let overlap_ratio = if target.feature_count() > 0 {
            transferred.feature_count().min(target.feature_count()) as f64
                / target.feature_count() as f64
        } else {
            0.0
        };

        Self {
            estimated_performance: estimated,
            improvement,
            relative_improvement: relative,
            confidence: overlap_ratio,
        }
    }

    /// Returns true if the transfer was beneficial.
    pub fn is_positive(&self) -> bool {
        self.improvement > 0.0
    }

    /// Returns true if the transfer hurt performance.
    pub fn is_negative(&self) -> bool {
        self.improvement < -0.05
    }
}
