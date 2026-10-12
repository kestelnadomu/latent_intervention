"""Downstream outcome predictor p: Z -> Y and its confusion-matrix metrics."""

from src.downstream.predictor import (
    OutcomePredictor,
    confusion_matrix,
    grouped_confusion,
    load_outcome_predictor,
    save_outcome_predictor,
    train_outcome_predictor,
)

__all__ = [
    "OutcomePredictor",
    "confusion_matrix",
    "grouped_confusion",
    "load_outcome_predictor",
    "save_outcome_predictor",
    "train_outcome_predictor",
]
