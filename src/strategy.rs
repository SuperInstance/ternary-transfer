//! Transfer strategies: how to move knowledge from source to target.

use crate::{
    DomainGap, KnowledgeMatrix, NegativeTransferDetector, SourceTask, TargetTask, TransferResult,
    TransferScore,
};

/// How to transfer knowledge between tasks.
#[derive(Debug, Clone, PartialEq)]
pub enum TransferStrategy {
    /// Copy source knowledge directly to target.
    DirectCopy,
    /// Blend source and target baseline with weight alpha (0.0 = all target, 1.0 = all source).
    WeightedBlend { alpha: f64 },
    /// Transfer only features that are shared between source and target.
    SelectiveTransfer { feature_indices: Vec<usize> },
    /// Progressive transfer: start with high alpha, decay over steps.
    Progressive {
        initial_alpha: f64,
        decay: f64,
        steps: usize,
    },
}

impl TransferStrategy {
    /// Execute the transfer strategy.
    pub fn execute(&self, source: &SourceTask, target: &TargetTask) -> TransferResult {
        let domain_gap = DomainGap::compute(source, target);

        let default_baseline = KnowledgeMatrix::zeros(target.feature_count());
        let knowledge = match self {
            TransferStrategy::DirectCopy => {
                let count = target.feature_count().min(source.knowledge.feature_count());
                let mut weights = vec![0.0; target.feature_count()];
                weights[..count].copy_from_slice(&source.knowledge.weights()[..count]);
                KnowledgeMatrix::new(weights)
            }

            TransferStrategy::WeightedBlend { alpha } => {
                let baseline = target
                    .baseline_knowledge
                    .as_ref()
                    .unwrap_or(&default_baseline);
                // Result is always sized to the target, regardless of whether
                // the source has fewer or more features than the target.
                blend_to_target(&source.knowledge, baseline, *alpha, target.feature_count())
            }

            TransferStrategy::SelectiveTransfer { feature_indices } => {
                let baseline = target
                    .baseline_knowledge
                    .as_ref()
                    .unwrap_or(&default_baseline);
                let n = target.feature_count();
                let mut weights = vec![0.0; n];

                // Start with baseline (size-safe: baseline length may differ).
                let copy_n = n.min(baseline.weights().len());
                weights[..copy_n].copy_from_slice(&baseline.weights()[..copy_n]);

                // Overwrite selected features from source.
                for &idx in feature_indices {
                    if idx < n && idx < source.knowledge.feature_count() {
                        weights[idx] = source.knowledge.weights()[idx];
                    }
                }
                KnowledgeMatrix::new(weights)
            }

            TransferStrategy::Progressive {
                initial_alpha,
                decay,
                steps,
            } => {
                let baseline = target
                    .baseline_knowledge
                    .as_ref()
                    .unwrap_or(&default_baseline);
                let n = target.feature_count();
                let mut alpha = (*initial_alpha).clamp(0.0, 1.0);
                let mut current: Vec<f64> = (0..n)
                    .map(|i| baseline.weights().get(i).copied().unwrap_or(0.0))
                    .collect();
                for _ in 0..*steps {
                    let mut next = vec![0.0; n];
                    for (i, slot) in next.iter_mut().enumerate() {
                        let s = source.knowledge.weights().get(i).copied().unwrap_or(0.0);
                        let c = current[i];
                        *slot = (s * alpha + c * (1.0 - alpha)).clamp(-1.0, 1.0);
                    }
                    current = next;
                    alpha *= decay;
                    if alpha < 0.01 {
                        break;
                    }
                }
                KnowledgeMatrix::new(current)
            }
        };

        let score = TransferScore::compute(
            source.source_performance,
            target.baseline_performance,
            &knowledge,
            target,
        );

        let negative_detected = NegativeTransferDetector::detect(&score, &domain_gap);

        TransferResult {
            knowledge,
            score,
            negative_detected,
            domain_gap,
        }
    }

    /// Get a human-readable name for this strategy.
    pub fn name(&self) -> &'static str {
        match self {
            TransferStrategy::DirectCopy => "direct_copy",
            TransferStrategy::WeightedBlend { .. } => "weighted_blend",
            TransferStrategy::SelectiveTransfer { .. } => "selective_transfer",
            TransferStrategy::Progressive { .. } => "progressive",
        }
    }
}

/// Blend `source` and `baseline` per-element into a vector of length `n`,
/// treating out-of-range entries on either side as 0.0.
///
/// Result element `i` = `(source[i] * alpha + baseline[i] * (1 - alpha))`
/// clamped to [-1, 1], with `alpha` clamped to [0, 1].
fn blend_to_target(
    source: &KnowledgeMatrix,
    baseline: &KnowledgeMatrix,
    alpha: f64,
    n: usize,
) -> KnowledgeMatrix {
    let alpha = alpha.clamp(0.0, 1.0);
    let mut weights = vec![0.0; n];
    for (i, slot) in weights.iter_mut().enumerate() {
        let s = source.weights().get(i).copied().unwrap_or(0.0);
        let t = baseline.weights().get(i).copied().unwrap_or(0.0);
        *slot = (s * alpha + t * (1.0 - alpha)).clamp(-1.0, 1.0);
    }
    KnowledgeMatrix::new(weights)
}
