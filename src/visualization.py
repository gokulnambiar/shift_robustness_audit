from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_domain_shift_drop(performance_frame: pd.DataFrame, output_dir: Path) -> None:
    source_only = performance_frame[performance_frame["selection_strategy"] == "source_only"]
    plot_frame = source_only[source_only["split"].isin(["source_test", "target_test"])].copy()

    pivot = plot_frame.pivot_table(
        index="experiment_name",
        columns="split",
        values="macro_f1",
    ).sort_index()

    x_positions = np.arange(len(pivot.index))
    width = 0.36

    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(
        x_positions - width / 2,
        pivot["source_test"],
        width=width,
        label="Source test",
        color="#1f77b4",
    )
    axis.bar(
        x_positions + width / 2,
        pivot["target_test"],
        width=width,
        label="Target test",
        color="#ff7f0e",
    )
    axis.set_ylabel("Macro F1")
    axis.set_title("In-domain vs out-of-domain performance")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(pivot.index, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "performance_drop.png", dpi=200)
    plt.close(fig)


def plot_weighting_comparison(performance_frame: pd.DataFrame, output_dir: Path) -> None:
    source_only = performance_frame[performance_frame["selection_strategy"] == "source_only"]
    plot_frame = source_only[source_only["split"] == "target_test"].copy()

    pivot = plot_frame.pivot_table(
        index="model_family",
        columns="weighting",
        values="macro_f1",
    ).sort_index()

    x_positions = np.arange(len(pivot.index))
    width = 0.36

    fig, axis = plt.subplots(figsize=(8.5, 5.0))
    axis.bar(
        x_positions - width / 2,
        pivot["unweighted"],
        width=width,
        label="Unweighted",
        color="#4c78a8",
    )
    axis.bar(
        x_positions + width / 2,
        pivot["importance_weighted"],
        width=width,
        label="Importance weighted",
        color="#f58518",
    )
    axis.set_ylabel("Target-domain Macro F1")
    axis.set_title("Effect of weighting under domain shift")
    axis.set_xticks(x_positions)
    axis.set_xticklabels([label.upper() for label in pivot.index])
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "weighted_comparison.png", dpi=200)
    plt.close(fig)


def plot_validation_strategy(performance_frame: pd.DataFrame, output_dir: Path) -> None:
    plot_frame = performance_frame[performance_frame["split"] == "target_test"].copy()
    plot_frame["family_label"] = plot_frame["experiment_name"]

    pivot = plot_frame.pivot_table(
        index="family_label",
        columns="selection_strategy",
        values="macro_f1",
    ).sort_index()

    x_positions = np.arange(len(pivot.index))
    width = 0.36

    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.bar(
        x_positions - width / 2,
        pivot["source_only"],
        width=width,
        label="Source-only validation",
        color="#59a14f",
    )
    axis.bar(
        x_positions + width / 2,
        pivot["mixed_validation"],
        width=width,
        label="Mixed validation",
        color="#e15759",
    )
    axis.set_ylabel("Target-domain Macro F1")
    axis.set_title("Validation strategy vs shifted test performance")
    axis.set_xticks(x_positions)
    axis.set_xticklabels(pivot.index, rotation=20, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_strategy.png", dpi=200)
    plt.close(fig)
