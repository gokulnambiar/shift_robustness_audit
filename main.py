from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.baseline_model import decision_scores as tfidf_decision_scores
from src.baseline_model import fit_tfidf_classifier
from src.data_loader import prepare_domain_splits
from src.embedding_features import FrozenSentenceEncoder
from src.embedding_features import decision_scores as embedding_decision_scores
from src.embedding_features import fit_embedding_classifier
from src.evaluation import build_example_predictions, build_metric_row, build_metrics_report
from src.reweighting import estimate_importance_weights
from src.visualization import plot_domain_shift_drop, plot_validation_strategy, plot_weighting_comparison

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
EMBEDDING_CACHE_DIR = DATA_DIR / "model_cache"
RANDOM_STATE = 42
TFIDF_C_VALUES = [0.5, 1.0, 2.0]
EMBEDDING_C_VALUES = [0.5, 1.0, 2.0]


def build_mixed_validation_frame(
    source_val: pd.DataFrame,
    target_val: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    sample_size = min(len(source_val), len(target_val))
    source_sample = source_val.sample(n=sample_size, random_state=random_state)
    target_sample = target_val.sample(n=sample_size, random_state=random_state)
    return (
        pd.concat([source_sample, target_sample], ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def choose_best_candidates(candidate_frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    best_source_only = candidate_frame.sort_values(
        ["source_val_macro_f1", "mixed_val_macro_f1"],
        ascending=False,
    ).iloc[0]
    best_mixed_validation = candidate_frame.sort_values(
        ["mixed_val_macro_f1", "source_val_macro_f1"],
        ascending=False,
    ).iloc[0]
    return best_source_only, best_mixed_validation


def run_tfidf_candidate_search(
    source_train: pd.DataFrame,
    source_val: pd.DataFrame,
    target_val: pd.DataFrame,
    mixed_val: pd.DataFrame,
    sample_weight: np.ndarray | None,
    weighting_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for c_value in TFIDF_C_VALUES:
        model = fit_tfidf_classifier(
            texts=source_train["text"],
            labels=source_train["label"].to_numpy(),
            c_value=c_value,
            sample_weight=sample_weight,
        )
        source_predictions = model.predict(source_val["text"])
        mixed_predictions = model.predict(mixed_val["text"])
        target_predictions = model.predict(target_val["text"])

        rows.append(
            {
                "model_family": "tfidf",
                "weighting": weighting_name,
                "candidate_c": c_value,
                "source_val_macro_f1": build_metric_row(
                    experiment_name="",
                    model_family="tfidf",
                    weighting=weighting_name,
                    selection_strategy="source_only",
                    split_name="source_val",
                    selected_c=c_value,
                    y_true=source_val["label"].to_numpy(),
                    y_pred=source_predictions,
                )["macro_f1"],
                "mixed_val_macro_f1": build_metric_row(
                    experiment_name="",
                    model_family="tfidf",
                    weighting=weighting_name,
                    selection_strategy="mixed_validation",
                    split_name="mixed_val",
                    selected_c=c_value,
                    y_true=mixed_val["label"].to_numpy(),
                    y_pred=mixed_predictions,
                )["macro_f1"],
                "target_val_macro_f1": build_metric_row(
                    experiment_name="",
                    model_family="tfidf",
                    weighting=weighting_name,
                    selection_strategy="mixed_validation",
                    split_name="target_val",
                    selected_c=c_value,
                    y_true=target_val["label"].to_numpy(),
                    y_pred=target_predictions,
                )["macro_f1"],
            }
        )

    candidate_frame = pd.DataFrame(rows)
    best_source_only, best_mixed_validation = choose_best_candidates(candidate_frame)
    candidate_frame["best_by_source_only"] = candidate_frame["candidate_c"] == best_source_only["candidate_c"]
    candidate_frame["best_by_mixed_validation"] = (
        candidate_frame["candidate_c"] == best_mixed_validation["candidate_c"]
    )
    return candidate_frame


def run_embedding_candidate_search(
    train_embeddings: np.ndarray,
    source_train_labels: np.ndarray,
    source_val_embeddings: np.ndarray,
    source_val_labels: np.ndarray,
    target_val_embeddings: np.ndarray,
    target_val_labels: np.ndarray,
    mixed_val_embeddings: np.ndarray,
    mixed_val_labels: np.ndarray,
    sample_weight: np.ndarray | None,
    weighting_name: str,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for c_value in EMBEDDING_C_VALUES:
        classifier = fit_embedding_classifier(
            embeddings=train_embeddings,
            labels=source_train_labels,
            c_value=c_value,
            sample_weight=sample_weight,
        )
        source_predictions = classifier.predict(source_val_embeddings)
        mixed_predictions = classifier.predict(mixed_val_embeddings)
        target_predictions = classifier.predict(target_val_embeddings)

        rows.append(
            {
                "model_family": "embedding",
                "weighting": weighting_name,
                "candidate_c": c_value,
                "source_val_macro_f1": build_metric_row(
                    experiment_name="",
                    model_family="embedding",
                    weighting=weighting_name,
                    selection_strategy="source_only",
                    split_name="source_val",
                    selected_c=c_value,
                    y_true=source_val_labels,
                    y_pred=source_predictions,
                )["macro_f1"],
                "mixed_val_macro_f1": build_metric_row(
                    experiment_name="",
                    model_family="embedding",
                    weighting=weighting_name,
                    selection_strategy="mixed_validation",
                    split_name="mixed_val",
                    selected_c=c_value,
                    y_true=mixed_val_labels,
                    y_pred=mixed_predictions,
                )["macro_f1"],
                "target_val_macro_f1": build_metric_row(
                    experiment_name="",
                    model_family="embedding",
                    weighting=weighting_name,
                    selection_strategy="mixed_validation",
                    split_name="target_val",
                    selected_c=c_value,
                    y_true=target_val_labels,
                    y_pred=target_predictions,
                )["macro_f1"],
            }
        )

    candidate_frame = pd.DataFrame(rows)
    best_source_only, best_mixed_validation = choose_best_candidates(candidate_frame)
    candidate_frame["best_by_source_only"] = candidate_frame["candidate_c"] == best_source_only["candidate_c"]
    candidate_frame["best_by_mixed_validation"] = (
        candidate_frame["candidate_c"] == best_mixed_validation["candidate_c"]
    )
    return candidate_frame


def finalize_tfidf_results(
    source_full_train: pd.DataFrame,
    source_test: pd.DataFrame,
    target_test: pd.DataFrame,
    selected_c: float,
    weighting_name: str,
    selection_strategy: str,
    sample_weight: np.ndarray | None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    experiment_name = f"tfidf_{weighting_name}"
    model = fit_tfidf_classifier(
        texts=source_full_train["text"],
        labels=source_full_train["label"].to_numpy(),
        c_value=selected_c,
        sample_weight=sample_weight,
    )

    source_predictions = model.predict(source_test["text"])
    target_predictions = model.predict(target_test["text"])
    source_scores = tfidf_decision_scores(model, source_test["text"])
    target_scores = tfidf_decision_scores(model, target_test["text"])

    metric_rows = [
        build_metric_row(
            experiment_name=experiment_name,
            model_family="tfidf",
            weighting=weighting_name,
            selection_strategy=selection_strategy,
            split_name="source_test",
            selected_c=selected_c,
            y_true=source_test["label"].to_numpy(),
            y_pred=source_predictions,
        ),
        build_metric_row(
            experiment_name=experiment_name,
            model_family="tfidf",
            weighting=weighting_name,
            selection_strategy=selection_strategy,
            split_name="target_test",
            selected_c=selected_c,
            y_true=target_test["label"].to_numpy(),
            y_pred=target_predictions,
        ),
    ]

    examples = build_example_predictions(
        frame=target_test,
        y_pred=target_predictions,
        raw_scores=target_scores,
        experiment_name=experiment_name,
        selection_strategy=selection_strategy,
        split_name="target_test",
    )
    examples["selected_c"] = selected_c

    return metric_rows, examples


def finalize_embedding_results(
    source_full_train_embeddings: np.ndarray,
    source_full_train_labels: np.ndarray,
    source_test: pd.DataFrame,
    source_test_embeddings: np.ndarray,
    target_test: pd.DataFrame,
    target_test_embeddings: np.ndarray,
    selected_c: float,
    weighting_name: str,
    selection_strategy: str,
    sample_weight: np.ndarray | None,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    experiment_name = f"embedding_{weighting_name}"
    classifier = fit_embedding_classifier(
        embeddings=source_full_train_embeddings,
        labels=source_full_train_labels,
        c_value=selected_c,
        sample_weight=sample_weight,
    )

    source_predictions = classifier.predict(source_test_embeddings)
    target_predictions = classifier.predict(target_test_embeddings)
    source_scores = embedding_decision_scores(classifier, source_test_embeddings)
    target_scores = embedding_decision_scores(classifier, target_test_embeddings)

    metric_rows = [
        build_metric_row(
            experiment_name=experiment_name,
            model_family="embedding",
            weighting=weighting_name,
            selection_strategy=selection_strategy,
            split_name="source_test",
            selected_c=selected_c,
            y_true=source_test["label"].to_numpy(),
            y_pred=source_predictions,
        ),
        build_metric_row(
            experiment_name=experiment_name,
            model_family="embedding",
            weighting=weighting_name,
            selection_strategy=selection_strategy,
            split_name="target_test",
            selected_c=selected_c,
            y_true=target_test["label"].to_numpy(),
            y_pred=target_predictions,
        ),
    ]

    examples = build_example_predictions(
        frame=target_test,
        y_pred=target_predictions,
        raw_scores=target_scores,
        experiment_name=experiment_name,
        selection_strategy=selection_strategy,
        split_name="target_test",
    )
    examples["selected_c"] = selected_c

    return metric_rows, examples


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bundle = prepare_domain_splits(data_dir=DATA_DIR, random_state=RANDOM_STATE)
    source_full_train = pd.concat([bundle.source_train, bundle.source_val], ignore_index=True)
    mixed_val = build_mixed_validation_frame(
        source_val=bundle.source_val,
        target_val=bundle.target_val,
        random_state=RANDOM_STATE,
    )

    train_weights, train_weight_diagnostics = estimate_importance_weights(
        source_texts=bundle.source_train["text"],
        target_texts=bundle.target_val["text"],
        random_state=RANDOM_STATE,
    )
    full_train_weights, full_train_weight_diagnostics = estimate_importance_weights(
        source_texts=source_full_train["text"],
        target_texts=bundle.target_val["text"],
        random_state=RANDOM_STATE,
    )
    reweighting_diagnostics = {
        "train": train_weight_diagnostics,
        "full_train": full_train_weight_diagnostics,
    }

    validation_frames = [
        run_tfidf_candidate_search(
            source_train=bundle.source_train,
            source_val=bundle.source_val,
            target_val=bundle.target_val,
            mixed_val=mixed_val,
            sample_weight=None,
            weighting_name="unweighted",
        ),
        run_tfidf_candidate_search(
            source_train=bundle.source_train,
            source_val=bundle.source_val,
            target_val=bundle.target_val,
            mixed_val=mixed_val,
            sample_weight=train_weights,
            weighting_name="importance_weighted",
        ),
    ]

    encoder = FrozenSentenceEncoder(cache_dir=EMBEDDING_CACHE_DIR)
    source_train_embeddings = encoder.encode(bundle.source_train["text"])
    source_val_embeddings = encoder.encode(bundle.source_val["text"])
    target_val_embeddings = encoder.encode(bundle.target_val["text"])
    mixed_val_embeddings = encoder.encode(mixed_val["text"])
    source_test_embeddings = encoder.encode(bundle.source_test["text"])
    target_test_embeddings = encoder.encode(bundle.target_test["text"])
    source_full_train_embeddings = np.vstack([source_train_embeddings, source_val_embeddings])

    validation_frames.extend(
        [
            run_embedding_candidate_search(
                train_embeddings=source_train_embeddings,
                source_train_labels=bundle.source_train["label"].to_numpy(),
                source_val_embeddings=source_val_embeddings,
                source_val_labels=bundle.source_val["label"].to_numpy(),
                target_val_embeddings=target_val_embeddings,
                target_val_labels=bundle.target_val["label"].to_numpy(),
                mixed_val_embeddings=mixed_val_embeddings,
                mixed_val_labels=mixed_val["label"].to_numpy(),
                sample_weight=None,
                weighting_name="unweighted",
            ),
            run_embedding_candidate_search(
                train_embeddings=source_train_embeddings,
                source_train_labels=bundle.source_train["label"].to_numpy(),
                source_val_embeddings=source_val_embeddings,
                source_val_labels=bundle.source_val["label"].to_numpy(),
                target_val_embeddings=target_val_embeddings,
                target_val_labels=bundle.target_val["label"].to_numpy(),
                mixed_val_embeddings=mixed_val_embeddings,
                mixed_val_labels=mixed_val["label"].to_numpy(),
                sample_weight=train_weights,
                weighting_name="importance_weighted",
            ),
        ]
    )

    validation_summary = pd.concat(validation_frames, ignore_index=True)

    performance_rows: list[dict[str, object]] = []
    example_frames: list[pd.DataFrame] = []

    for _, selection_row in validation_summary[validation_summary["best_by_source_only"]].iterrows():
        if selection_row["model_family"] == "tfidf":
            metric_rows, examples = finalize_tfidf_results(
                source_full_train=source_full_train,
                source_test=bundle.source_test,
                target_test=bundle.target_test,
                selected_c=float(selection_row["candidate_c"]),
                weighting_name=str(selection_row["weighting"]),
                selection_strategy="source_only",
                sample_weight=full_train_weights if selection_row["weighting"] == "importance_weighted" else None,
            )
        else:
            metric_rows, examples = finalize_embedding_results(
                source_full_train_embeddings=source_full_train_embeddings,
                source_full_train_labels=source_full_train["label"].to_numpy(),
                source_test=bundle.source_test,
                source_test_embeddings=source_test_embeddings,
                target_test=bundle.target_test,
                target_test_embeddings=target_test_embeddings,
                selected_c=float(selection_row["candidate_c"]),
                weighting_name=str(selection_row["weighting"]),
                selection_strategy="source_only",
                sample_weight=full_train_weights if selection_row["weighting"] == "importance_weighted" else None,
            )

        performance_rows.extend(metric_rows)
        example_frames.append(examples)

    for _, selection_row in validation_summary[validation_summary["best_by_mixed_validation"]].iterrows():
        if selection_row["model_family"] == "tfidf":
            metric_rows, examples = finalize_tfidf_results(
                source_full_train=source_full_train,
                source_test=bundle.source_test,
                target_test=bundle.target_test,
                selected_c=float(selection_row["candidate_c"]),
                weighting_name=str(selection_row["weighting"]),
                selection_strategy="mixed_validation",
                sample_weight=full_train_weights if selection_row["weighting"] == "importance_weighted" else None,
            )
        else:
            metric_rows, examples = finalize_embedding_results(
                source_full_train_embeddings=source_full_train_embeddings,
                source_full_train_labels=source_full_train["label"].to_numpy(),
                source_test=bundle.source_test,
                source_test_embeddings=source_test_embeddings,
                target_test=bundle.target_test,
                target_test_embeddings=target_test_embeddings,
                selected_c=float(selection_row["candidate_c"]),
                weighting_name=str(selection_row["weighting"]),
                selection_strategy="mixed_validation",
                sample_weight=full_train_weights if selection_row["weighting"] == "importance_weighted" else None,
            )

        performance_rows.extend(metric_rows)
        example_frames.append(examples)

    performance_summary = pd.DataFrame(performance_rows).sort_values(
        ["selection_strategy", "experiment_name", "split"]
    )
    example_predictions = pd.concat(example_frames, ignore_index=True)

    performance_summary.to_csv(OUTPUT_DIR / "performance_summary.csv", index=False)
    validation_summary.sort_values(["model_family", "weighting", "candidate_c"]).to_csv(
        OUTPUT_DIR / "validation_selection_summary.csv",
        index=False,
    )
    example_predictions.to_csv(OUTPUT_DIR / "example_predictions.csv", index=False)
    (OUTPUT_DIR / "dataset_summary.json").write_text(json.dumps(bundle.dataset_summary, indent=2))
    (OUTPUT_DIR / "reweighting_diagnostics.json").write_text(
        json.dumps(reweighting_diagnostics, indent=2)
    )

    report = build_metrics_report(
        performance_frame=performance_summary,
        validation_frame=validation_summary,
        dataset_summary=bundle.dataset_summary,
        reweighting_diagnostics=reweighting_diagnostics,
    )
    (OUTPUT_DIR / "metrics_report.txt").write_text(report)

    plot_domain_shift_drop(performance_frame=performance_summary, output_dir=OUTPUT_DIR)
    plot_weighting_comparison(performance_frame=performance_summary, output_dir=OUTPUT_DIR)
    plot_validation_strategy(performance_frame=performance_summary, output_dir=OUTPUT_DIR)


if __name__ == "__main__":
    main()
