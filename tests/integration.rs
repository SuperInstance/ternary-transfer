#[cfg(test)]
mod tests {
    use ternary_transfer::negative;
    use ternary_transfer::task;
    use ternary_transfer::*;

    fn make_feature(name: &str, importance: f64, bias: Ternary) -> task::FeatureDescriptor {
        task::FeatureDescriptor {
            name: name.to_string(),
            importance,
            ternary_bias: bias,
        }
    }

    fn make_source(weights: Vec<f64>) -> SourceTask {
        let count = weights.len();
        let features: Vec<_> = (0..count)
            .map(|i| make_feature(&format!("f{}", i), 0.5, Ternary::Positive))
            .collect();
        SourceTask::new("source", features, KnowledgeMatrix::new(weights))
    }

    fn make_target(count: usize) -> TargetTask {
        let features: Vec<_> = (0..count)
            .map(|i| make_feature(&format!("f{}", i), 0.5, Ternary::Positive))
            .collect();
        TargetTask::new("target", features)
    }

    // --- Ternary tests ---

    #[test]
    fn ternary_from_i8() {
        assert_eq!(Ternary::from_i8(-1), Some(Ternary::Negative));
        assert_eq!(Ternary::from_i8(0), Some(Ternary::Neutral));
        assert_eq!(Ternary::from_i8(1), Some(Ternary::Positive));
        assert_eq!(Ternary::from_i8(2), None);
    }

    #[test]
    fn ternary_to_f64() {
        assert_eq!(Ternary::Negative.to_f64(), -1.0);
        assert_eq!(Ternary::Neutral.to_f64(), 0.0);
        assert_eq!(Ternary::Positive.to_f64(), 1.0);
    }

    // --- KnowledgeMatrix tests ---

    #[test]
    fn knowledge_matrix_clamps_weights() {
        let km = KnowledgeMatrix::new(vec![-2.0, 0.5, 1.5]);
        assert_eq!(km.weights(), &[-1.0, 0.5, 1.0]);
    }

    #[test]
    fn knowledge_matrix_zeros() {
        let km = KnowledgeMatrix::zeros(5);
        assert_eq!(km.feature_count(), 5);
        assert!(km.weights().iter().all(|&w| w == 0.0));
    }

    #[test]
    fn knowledge_matrix_ternary_signs() {
        let km = KnowledgeMatrix::new(vec![-0.5, 0.0, 0.8]);
        let signs = km.ternary_signs();
        assert_eq!(
            signs,
            vec![Ternary::Negative, Ternary::Neutral, Ternary::Positive]
        );
    }

    #[test]
    fn cosine_similarity_identical() {
        let km = KnowledgeMatrix::new(vec![0.5, -0.3, 0.8]);
        let sim = km.cosine_similarity(&km);
        assert!((sim - 1.0).abs() < 1e-9);
    }

    #[test]
    fn cosine_similarity_orthogonal() {
        let a = KnowledgeMatrix::new(vec![1.0, 0.0]);
        let b = KnowledgeMatrix::new(vec![0.0, 1.0]);
        let sim = a.cosine_similarity(&b);
        assert!(sim.abs() < 1e-9);
    }

    #[test]
    fn knowledge_matrix_blend() {
        let a = KnowledgeMatrix::new(vec![1.0, 1.0]);
        let b = KnowledgeMatrix::new(vec![-1.0, -1.0]);
        let blended = a.blend(&b, 0.5);
        assert_eq!(blended.weights(), &[0.0, 0.0]);
    }

    #[test]
    fn knowledge_matrix_from_seed_deterministic() {
        let a = KnowledgeMatrix::from_seed(10, 42);
        let b = KnowledgeMatrix::from_seed(10, 42);
        assert_eq!(a, b);
    }

    // --- DomainGap tests ---

    #[test]
    fn domain_gap_identical_tasks() {
        let source = make_source(vec![0.5, 0.5]);
        let target = make_target(2);
        let gap = DomainGap::compute(&source, &target);
        assert!(
            gap.gap < 0.1,
            "identical tasks should have low gap, got {}",
            gap.gap
        );
        assert!(gap.feature_overlap > 0.9);
    }

    #[test]
    fn domain_gap_different_tasks() {
        let features = vec![make_feature("x", 0.9, Ternary::Negative)];
        let source = SourceTask::new("src", features, KnowledgeMatrix::new(vec![0.5]));
        let target_features = vec![make_feature("z", 0.1, Ternary::Positive)];
        let target = TargetTask::new("tgt", target_features);
        let gap = DomainGap::compute(&source, &target);
        assert!(
            gap.gap > 0.5,
            "unrelated tasks should have high gap, got {}",
            gap.gap
        );
    }

    #[test]
    fn domain_gap_recommend_strategy() {
        let close = DomainGap {
            gap: 0.1,
            feature_overlap: 1.0,
            importance_divergence: 0.0,
            bias_disagreement: 0.0,
        };
        assert_eq!(close.recommend_strategy(), "direct_copy");

        let distant = DomainGap {
            gap: 0.8,
            feature_overlap: 0.0,
            importance_divergence: 1.0,
            bias_disagreement: 1.0,
        };
        assert_eq!(distant.recommend_strategy(), "progressive");
    }

    // --- TransferStrategy tests ---

    #[test]
    fn direct_copy_transfer() {
        let source = make_source(vec![0.8, -0.6]);
        let target = make_target(2);
        let strategy = TransferStrategy::DirectCopy;
        let result = strategy.execute(&source, &target);
        assert_eq!(result.knowledge.weights()[0], 0.8);
        assert_eq!(result.knowledge.weights()[1], -0.6);
    }

    #[test]
    fn weighted_blend_transfer() {
        let source = make_source(vec![1.0, 1.0]);
        let target = make_target(2);
        let strategy = TransferStrategy::WeightedBlend { alpha: 0.0 };
        let result = strategy.execute(&source, &target);
        // alpha=0 means all target (zeros)
        assert_eq!(result.knowledge.weights()[0], 0.0);
    }

    #[test]
    fn selective_transfer() {
        let source = make_source(vec![1.0, -1.0, 0.5]);
        let target = make_target(3);
        let strategy = TransferStrategy::SelectiveTransfer {
            feature_indices: vec![0, 2],
        };
        let result = strategy.execute(&source, &target);
        assert_eq!(result.knowledge.weights()[0], 1.0);
        assert_eq!(result.knowledge.weights()[1], 0.0); // not selected
        assert_eq!(result.knowledge.weights()[2], 0.5);
    }

    #[test]
    fn progressive_transfer() {
        let source = make_source(vec![1.0]);
        let target = make_target(1);
        let strategy = TransferStrategy::Progressive {
            initial_alpha: 0.9,
            decay: 0.5,
            steps: 5,
        };
        let result = strategy.execute(&source, &target);
        // Should converge somewhere between 0 and 1
        assert!(result.knowledge.weights()[0] > 0.0);
    }

    // --- TransferScore tests ---

    #[test]
    fn transfer_score_positive_when_aligned() {
        let features = vec![make_feature("f0", 1.0, Ternary::Positive)];
        let target = TargetTask::new("tgt", features);
        let km = KnowledgeMatrix::new(vec![0.8]);
        let score = TransferScore::compute(0.9, 0.5, &km, &target);
        assert!(score.improvement > 0.0);
        assert!(score.is_positive());
    }

    #[test]
    fn transfer_score_negative_when_misaligned() {
        let features = vec![make_feature("f0", 1.0, Ternary::Positive)];
        let target = TargetTask::new("tgt", features);
        let km = KnowledgeMatrix::new(vec![-0.8]);
        let score = TransferScore::compute(0.9, 0.5, &km, &target);
        assert!(score.improvement < 0.0);
    }

    // --- NegativeTransferDetector tests ---

    #[test]
    fn negative_detector_no_negative() {
        let score = TransferScore {
            estimated_performance: 0.8,
            improvement: 0.2,
            relative_improvement: 0.3,
            confidence: 0.9,
        };
        let gap = DomainGap {
            gap: 0.2,
            feature_overlap: 0.9,
            importance_divergence: 0.1,
            bias_disagreement: 0.0,
        };
        assert!(!NegativeTransferDetector::detect(&score, &gap));
    }

    #[test]
    fn negative_detector_catches_negative() {
        let score = TransferScore {
            estimated_performance: 0.3,
            improvement: -0.2,
            relative_improvement: -0.4,
            confidence: 0.5,
        };
        let gap = DomainGap {
            gap: 0.3,
            feature_overlap: 0.5,
            importance_divergence: 0.3,
            bias_disagreement: 0.2,
        };
        assert!(NegativeTransferDetector::detect(&score, &gap));
    }

    #[test]
    fn negative_detector_full_analysis() {
        let detector = NegativeTransferDetector::new().with_threshold(-0.1);
        let score = TransferScore {
            estimated_performance: 0.7,
            improvement: 0.1,
            relative_improvement: 0.15,
            confidence: 0.8,
        };
        let gap = DomainGap {
            gap: 0.3,
            feature_overlap: 0.8,
            importance_divergence: 0.2,
            bias_disagreement: 0.1,
        };
        let report = detector.analyze(&score, &gap);
        assert!(!report.negative_detected);
        assert_eq!(report.risk_level, negative::RiskLevel::Low);
    }

    // --- Coverage for the round4 fixes and previously-untested branches ---

    fn gap_at(value: f64) -> DomainGap {
        DomainGap {
            gap: value,
            feature_overlap: 0.0,
            importance_divergence: 0.0,
            bias_disagreement: 0.0,
        }
    }

    fn score_at(improvement: f64, confidence: f64) -> TransferScore {
        TransferScore {
            estimated_performance: 0.5,
            improvement,
            relative_improvement: 0.0,
            confidence,
        }
    }

    #[test]
    fn from_seed_uses_full_range() {
        // The old LCG scaling only produced [0, 1); verify the full [-1, 1]
        // range is now reachable, including negative weights.
        let km = KnowledgeMatrix::from_seed(200, 42);
        assert_eq!(km.feature_count(), 200);
        let min = km.weights().iter().cloned().fold(f64::INFINITY, f64::min);
        let max = km
            .weights()
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        assert!(min < 0.0, "expected negative weights, min = {min}");
        assert!(max > 0.0, "expected positive weights, max = {max}");
        assert!(
            km.weights().iter().all(|&w| (-1.0..=1.0).contains(&w)),
            "weights must stay within [-1, 1]"
        );
    }

    #[test]
    fn from_seed_remains_deterministic() {
        let a = KnowledgeMatrix::from_seed(64, 7);
        let b = KnowledgeMatrix::from_seed(64, 7);
        assert_eq!(a, b);
    }

    #[test]
    fn blend_unequal_lengths_uses_zero_fill() {
        // zip()-based blend silently truncated to the shorter side; the
        // result must now span the longer length with 0.0 fill.
        let a = KnowledgeMatrix::new(vec![1.0, 1.0, 1.0]);
        let b = KnowledgeMatrix::new(vec![-1.0]);
        let blended = a.blend(&b, 0.5);
        assert_eq!(blended.feature_count(), 3);
        // index 0: 0.5*1 + 0.5*(-1) = 0 ; indices 1,2: 0.5*1 + 0.5*0 = 0.5
        assert!((blended.weights()[0] - 0.0).abs() < 1e-9);
        assert!((blended.weights()[1] - 0.5).abs() < 1e-9);
        assert!((blended.weights()[2] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn blend_clamps_alpha() {
        let a = KnowledgeMatrix::new(vec![0.4]);
        let b = KnowledgeMatrix::new(vec![-0.4]);
        let over = a.blend(&b, 5.0);
        let under = a.blend(&b, -5.0);
        assert_eq!(over.weights()[0], 0.4); // alpha clamped to 1.0 -> all a
        assert_eq!(under.weights()[0], -0.4); // alpha clamped to 0.0 -> all b
    }

    #[test]
    fn weighted_blend_target_sized_when_source_smaller() {
        // Regression: previously produced a 1-element matrix for a 3-feature
        // target because blend() truncated to the source length.
        let source = make_source(vec![0.9]);
        let target = make_target(3);
        let result = TransferStrategy::WeightedBlend { alpha: 0.5 }.execute(&source, &target);
        assert_eq!(result.knowledge.feature_count(), 3);
    }

    #[test]
    fn weighted_blend_target_sized_when_source_larger() {
        let source = make_source(vec![0.9, 0.9, 0.9, 0.9]);
        let target = make_target(2);
        let result = TransferStrategy::WeightedBlend { alpha: 0.5 }.execute(&source, &target);
        assert_eq!(result.knowledge.feature_count(), 2);
    }

    #[test]
    fn direct_copy_pads_when_source_smaller_and_truncates_when_larger() {
        let small = make_source(vec![0.7]);
        let big = make_source(vec![0.1, 0.2, 0.3, 0.4]);
        let target = make_target(3);

        let padded = TransferStrategy::DirectCopy.execute(&small, &target);
        assert_eq!(padded.knowledge.feature_count(), 3);
        assert_eq!(padded.knowledge.weights()[0], 0.7);
        assert_eq!(padded.knowledge.weights()[1], 0.0);

        let truncated = TransferStrategy::DirectCopy.execute(&big, &target);
        assert_eq!(truncated.knowledge.feature_count(), 3);
        assert_eq!(truncated.knowledge.weights()[2], 0.3);
    }

    #[test]
    fn selective_transfer_ignores_out_of_range_indices() {
        let source = make_source(vec![0.9, 0.9, 0.9]);
        let target = make_target(3);
        // index 5 is out of range for both source and target -> ignored.
        let strategy = TransferStrategy::SelectiveTransfer {
            feature_indices: vec![0, 5],
        };
        let result = strategy.execute(&source, &target);
        assert_eq!(result.knowledge.weights()[0], 0.9);
        assert_eq!(result.knowledge.feature_count(), 3);
    }

    #[test]
    fn progressive_zero_steps_returns_baseline() {
        let source = make_source(vec![1.0, 1.0]);
        let target = make_target(2);
        let strategy = TransferStrategy::Progressive {
            initial_alpha: 0.9,
            decay: 0.5,
            steps: 0,
        };
        let result = strategy.execute(&source, &target);
        // No steps -> knowledge stays at the (zero) baseline.
        assert_eq!(result.knowledge.weights(), &[0.0, 0.0]);
    }

    #[test]
    fn risk_level_boundaries_match_readme() {
        let detector = NegativeTransferDetector::new();
        let score = score_at(0.0, 1.0);
        assert_eq!(
            detector.analyze(&score, &gap_at(0.39)).risk_level,
            negative::RiskLevel::Low
        );
        // >= 0.4 -> Medium (previously > 0.4 left 0.4 as Low).
        assert_eq!(
            detector.analyze(&score, &gap_at(0.4)).risk_level,
            negative::RiskLevel::Medium
        );
        assert_eq!(
            detector.analyze(&score, &gap_at(0.59)).risk_level,
            negative::RiskLevel::Medium
        );
        // >= 0.6 -> High
        assert_eq!(
            detector.analyze(&score, &gap_at(0.6)).risk_level,
            negative::RiskLevel::High
        );
        assert_eq!(
            detector.analyze(&score, &gap_at(0.79)).risk_level,
            negative::RiskLevel::High
        );
        // >= 0.8 -> Critical (previously > 0.8 left 0.8 as High).
        assert_eq!(
            detector.analyze(&score, &gap_at(0.8)).risk_level,
            negative::RiskLevel::Critical
        );
    }

    #[test]
    fn gap_risk_threshold_default_is_seventy_not_sixty() {
        // README and detect() use 0.7; the old default (0.6) flagged gaps in
        // [0.6, 0.7] as negative. Verify the default now matches the docs.
        let detector = NegativeTransferDetector::new();
        let low_conf = score_at(0.0, 0.1);
        // gap = 0.65, confidence low, no improvement -> NOT negative at 0.7.
        assert!(!detector.analyze(&low_conf, &gap_at(0.65)).negative_detected);
        // gap = 0.75, confidence low -> negative.
        assert!(detector.analyze(&low_conf, &gap_at(0.75)).negative_detected);
    }

    #[test]
    fn detect_high_gap_low_confidence_branch() {
        // Covers the second arm of NegativeTransferDetector::detect that no
        // existing test exercised: gap > 0.7 AND confidence < 0.3.
        let score = score_at(0.0, 0.2);
        let gap = gap_at(0.8);
        assert!(NegativeTransferDetector::detect(&score, &gap));
        // Same gap but high confidence -> not flagged by this arm.
        let high_conf = score_at(0.0, 0.9);
        assert!(!NegativeTransferDetector::detect(&high_conf, &gap));
    }

    #[test]
    fn recommend_strategy_boundaries() {
        assert_eq!(gap_at(0.0).recommend_strategy(), "direct_copy");
        assert_eq!(gap_at(0.2).recommend_strategy(), "weighted_blend");
        assert_eq!(gap_at(0.4).recommend_strategy(), "selective_transfer");
        assert_eq!(gap_at(0.6).recommend_strategy(), "progressive");
        assert_eq!(gap_at(1.0).recommend_strategy(), "progressive");
    }

    #[test]
    fn cosine_similarity_dimension_and_zero_handling() {
        let a = KnowledgeMatrix::new(vec![0.5, 0.5]);
        let mismatched = KnowledgeMatrix::new(vec![0.5, 0.5, 0.5]);
        assert_eq!(a.cosine_similarity(&mismatched), 0.0);

        let zero = KnowledgeMatrix::zeros(3);
        let nonzero = KnowledgeMatrix::new(vec![0.5, 0.5, 0.5]);
        // Zero norm -> denominator 0 -> defined as 0.0.
        assert_eq!(zero.cosine_similarity(&nonzero), 0.0);
    }

    #[test]
    fn transfer_score_zero_importance_returns_baseline() {
        let features = vec![make_feature("f0", 0.0, Ternary::Neutral)];
        let target = TargetTask::new("tgt", features);
        let km = KnowledgeMatrix::new(vec![0.8]);
        let score = TransferScore::compute(0.9, 0.5, &km, &target);
        assert_eq!(score.estimated_performance, 0.5);
        assert_eq!(score.improvement, 0.0);
        assert_eq!(score.confidence, 0.0);
        assert!(!score.is_positive());
        assert!(!score.is_negative());
    }

    #[test]
    fn domain_gap_both_empty_is_maximal() {
        let source = SourceTask::new("src", vec![], KnowledgeMatrix::zeros(0));
        let target = TargetTask::new("tgt", vec![]);
        let gap = DomainGap::compute(&source, &target);
        assert_eq!(gap.feature_overlap, 0.0);
        assert_eq!(gap.importance_divergence, 1.0);
        assert_eq!(gap.bias_disagreement, 1.0);
        assert!((gap.gap - 1.0).abs() < 1e-9);
        assert!(!gap.is_close());
        assert!(gap.is_distant());
    }

    #[test]
    fn strategy_names() {
        assert_eq!(TransferStrategy::DirectCopy.name(), "direct_copy");
        assert_eq!(
            TransferStrategy::WeightedBlend { alpha: 0.5 }.name(),
            "weighted_blend"
        );
        assert_eq!(
            TransferStrategy::SelectiveTransfer {
                feature_indices: vec![]
            }
            .name(),
            "selective_transfer"
        );
        assert_eq!(
            TransferStrategy::Progressive {
                initial_alpha: 0.5,
                decay: 0.5,
                steps: 1
            }
            .name(),
            "progressive"
        );
    }
}
