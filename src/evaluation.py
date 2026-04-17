from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def sigmoid_confidence(raw_scores: np.ndarray) -> np.ndarray:
    clipped_scores = np.clip(raw_scores, -25.0, 25.0)
    return 1.0 / (1.0 + np.exp(-clipped_scores))


def build_metric_row(
    experiment_name: str,
    model_family: str,
    weighting: str,
    selection_strategy: str,
    split_name: str,
    selected_c: float,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    metrics = compute_metrics(y_true=y_true, y_pred=y_pred)
    return {
        "experiment_name": experiment_name,
        "model_family": model_family,
        "weighting": weighting,
        "selection_strategy": selection_strategy,
        "split": split_name,
        "selected_c": selected_c,
        "n_examples": int(len(y_true)),
        **metrics,
    }


def build_example_predictions(
    frame: pd.DataFrame,
    y_pred: np.ndarray,
    raw_scores: np.ndarray,
    experiment_name: str,
    selection_strategy: str,
    split_name: str,
    max_examples: int = 6,
) -> pd.DataFrame:
    examples = frame.copy()
    examples["predicted_label"] = y_pred
    examples["raw_score"] = raw_scores
    examples["positive_probability"] = sigmoid_confidence(raw_scores)
    examples["confidence"] = np.where(
        examples["predicted_label"] == 1,
        examples["positive_probability"],
        1.0 - examples["positive_probability"],
    )
    examples["is_correct"] = examples["label"] == examples["predicted_label"]

    mistake_count = max_examples // 2
    mistake_examples = (
        examples[~examples["is_correct"]]
        .sort_values("confidence", ascending=False)
        .head(mistake_count)
    )
    correct_examples = (
        examples[examples["is_correct"]]
        .sort_values("confidence", ascending=False)
        .head(max_examples - len(mistake_examples))
    )
    selected_examples = pd.concat([mistake_examples, correct_examples], ignore_index=True)
    selected_examples["text_excerpt"] = selected_examples["text"].str.slice(0, 240)
    selected_examples["experiment_name"] = experiment_name
    selected_examples["selection_strategy"] = selection_strategy
    selected_examples["split"] = split_name
    selected_examples["true_label_name"] = selected_examples["label"].map({0: "negative", 1: "positive"})
    selected_examples["predicted_label_name"] = selected_examples["predicted_label"].map(
        {0: "negative", 1: "positive"}
    )

    return selected_examples[
        [
            "experiment_name",
            "selection_strategy",
            "split",
            "domain_name",
            "true_label_name",
            "predicted_label_name",
            "confidence",
            "positive_probability",
            "text_excerpt",
        ]
    ]


def build_metrics_report(
    performance_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    dataset_summary: dict[str, Any],
    reweighting_diagnostics: dict[str, Any],
) -> str:
    source_only_results = performance_frame[performance_frame["selection_strategy"] == "source_only"]
    source_test_results = source_only_results[source_only_results["split"] == "source_test"]
    target_test_results = source_only_results[source_only_results["split"] == "target_test"]

    merged_results = source_test_results.merge(
        target_test_results,
        on=["experiment_name", "model_family", "weighting", "selection_strategy", "selected_c"],
        suffixes=("_source", "_target"),
    )
    merged_results["macro_f1_drop"] = (
        merged_results["macro_f1_source"] - merged_results["macro_f1_target"]
    )

    best_source_experiment = (
        source_test_results.sort_values("macro_f1", ascending=False).head(1)["experiment_name"].iloc[0]
    )
    best_target_experiment = (
        target_test_results.sort_values("macro_f1", ascending=False).head(1)["experiment_name"].iloc[0]
    )
    average_drop = float(merged_results["macro_f1_drop"].mean())

    source_choice_rows = validation_frame[validation_frame["best_by_source_only"]]
    mixed_choice_rows = validation_frame[validation_frame["best_by_mixed_validation"]]
    differing_choices = int(
        (
            source_choice_rows.set_index(["model_family", "weighting"])["candidate_c"]
            != mixed_choice_rows.set_index(["model_family", "weighting"])["candidate_c"]
        ).sum()
    )

    lines = [
        "Shift Robustness Audit",
        "======================",
        "",
        f"Source domain: {dataset_summary['source_domain']}",
        f"Target domain: {dataset_summary['target_domain']}",
        f"Source sampled reviews: {dataset_summary['source_sampled_reviews']}",
        f"Target sampled reviews: {dataset_summary['target_sampled_reviews']}",
        "",
        "Core findings",
        "-------------",
        f"Average macro F1 drop from source test to target test: {average_drop:.3f}",
        f"Best source-domain model under source-only selection: {best_source_experiment}",
        f"Best target-domain model under source-only selection: {best_target_experiment}",
        f"Model families with different hyperparameter choices under source-only vs mixed validation: {differing_choices}",
        "",
        "Shift diagnostics",
        "-----------------",
        f"Domain-discriminator ROC AUC on source train vs target validation: {reweighting_diagnostics['train']['domain_auc']:.3f}",
        f"Mean normalized training weight on source examples: {reweighting_diagnostics['train']['mean_weight']:.3f}",
        f"Training weight range after clipping: {reweighting_diagnostics['train']['min_weight']:.3f} to {reweighting_diagnostics['train']['max_weight']:.3f}",
        "",
        "Artifacts",
        "---------",
        "performance_summary.csv",
        "validation_selection_summary.csv",
        "example_predictions.csv",
        "performance_drop.png",
        "weighted_comparison.png",
        "validation_strategy.png",
    ]
    return "\n".join(lines)
