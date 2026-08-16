"""Machine-learning utilities for the real-label solve model and temporal evaluation."""

from .evaluation import (
    build_model_comparison_report,
    evaluate_model_suite,
    run_ablation_study,
    summarize_feature_importance,
    temporal_split_user_aware,
)
from .training_data import (
    DEFAULT_OUTCOME_WINDOW_HOURS,
    build_real_solve_training_dataset,
    build_user_problem_event_dataset,
    validate_event_dataset,
)

__all__ = [
    "DEFAULT_OUTCOME_WINDOW_HOURS",
    "build_real_solve_training_dataset",
    "build_user_problem_event_dataset",
    "validate_event_dataset",
    "temporal_split_user_aware",
    "build_model_comparison_report",
    "summarize_feature_importance",
    "evaluate_model_suite",
    "run_ablation_study",
]
