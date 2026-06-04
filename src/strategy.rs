//! Transfer strategies: how to move knowledge from source to target.

use crate::{KnowledgeMatrix, SourceTask, TargetTask, TransferResult, DomainGap, TransferScore, NegativeTransferDetector};

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
    Progressive { initial_alpha: f64, decay: f64, steps: usize },
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
                for i in 0..count {
                    weights[i] = source.knowledge.weights()[i];
                }
                KnowledgeMatrix::new(weights)
            }

            TransferStrategy::WeightedBlend { alpha } => {
                let baseline = target.baseline_knowledge.as_ref()
                    .unwrap_or(&default_baseline);
                source.knowledge.blend(baseline, *alpha)
            }

            TransferStrategy::SelectiveTransfer { feature_indices } => {
                let mut weights = vec![0.0; target.feature_count()];
                let baseline = target.baseline_knowledge.as_ref()
                    .unwrap_or(&default_baseline);

                // Start with baseline
                weights.copy_from_slice(baseline.weights());

                // Overwrite selected features from source
                for &idx in feature_indices {
                    if idx < weights.len() && idx < source.knowledge.feature_count() {
                        weights[idx] = source.knowledge.weights()[idx];
                    }
                }
                KnowledgeMatrix::new(weights)
            }

            TransferStrategy::Progressive { initial_alpha, decay, steps } => {
                let baseline = target.baseline_knowledge.as_ref()
                    .unwrap_or(&default_baseline);
                let mut alpha = *initial_alpha;
                let mut current = baseline.clone();
                for _ in 0..*steps {
                    current = source.knowledge.blend(&current, alpha);
                    alpha *= decay;
                    if alpha < 0.01 {
                        break;
                    }
                }
                current
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
