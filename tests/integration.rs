#[cfg(test)]
mod tests {
    use ternary_transfer::*;
    use ternary_transfer::task;
    use ternary_transfer::negative;

    fn make_feature(name: &str, importance: f64, bias: Ternary) -> task::FeatureDescriptor {
        task::FeatureDescriptor { name: name.to_string(), importance, ternary_bias: bias }
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
        assert_eq!(signs, vec![Ternary::Negative, Ternary::Neutral, Ternary::Positive]);
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
        assert!(gap.gap < 0.1, "identical tasks should have low gap, got {}", gap.gap);
        assert!(gap.feature_overlap > 0.9);
    }

    #[test]
    fn domain_gap_different_tasks() {
        let features = vec![make_feature("x", 0.9, Ternary::Negative)];
        let source = SourceTask::new("src", features, KnowledgeMatrix::new(vec![0.5]));
        let target_features = vec![make_feature("z", 0.1, Ternary::Positive)];
        let target = TargetTask::new("tgt", target_features);
        let gap = DomainGap::compute(&source, &target);
        assert!(gap.gap > 0.5, "unrelated tasks should have high gap, got {}", gap.gap);
    }

    #[test]
    fn domain_gap_recommend_strategy() {
        let close = DomainGap { gap: 0.1, feature_overlap: 1.0, importance_divergence: 0.0, bias_disagreement: 0.0 };
        assert_eq!(close.recommend_strategy(), "direct_copy");

        let distant = DomainGap { gap: 0.8, feature_overlap: 0.0, importance_divergence: 1.0, bias_disagreement: 1.0 };
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
        let strategy = TransferStrategy::SelectiveTransfer { feature_indices: vec![0, 2] };
        let result = strategy.execute(&source, &target);
        assert_eq!(result.knowledge.weights()[0], 1.0);
        assert_eq!(result.knowledge.weights()[1], 0.0); // not selected
        assert_eq!(result.knowledge.weights()[2], 0.5);
    }

    #[test]
    fn progressive_transfer() {
        let source = make_source(vec![1.0]);
        let target = make_target(1);
        let strategy = TransferStrategy::Progressive { initial_alpha: 0.9, decay: 0.5, steps: 5 };
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
        let gap = DomainGap { gap: 0.2, feature_overlap: 0.9, importance_divergence: 0.1, bias_disagreement: 0.0 };
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
        let gap = DomainGap { gap: 0.3, feature_overlap: 0.5, importance_divergence: 0.3, bias_disagreement: 0.2 };
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
        let gap = DomainGap { gap: 0.3, feature_overlap: 0.8, importance_divergence: 0.2, bias_disagreement: 0.1 };
        let report = detector.analyze(&score, &gap);
        assert!(!report.negative_detected);
        assert_eq!(report.risk_level, negative::RiskLevel::Low);
    }
}
