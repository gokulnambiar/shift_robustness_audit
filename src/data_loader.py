from __future__ import annotations

import ast
import gzip
import json
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import derive_label, is_usable_text, join_review_fields

AMAZON_DATA_BASE_URL = "https://jmcauley.ucsd.edu/data/amazon_v2/categoryFilesSmall"
DEFAULT_SOURCE_DOMAIN = "Digital_Music"
DEFAULT_TARGET_DOMAIN = "Luxury_Beauty"

DOMAIN_METADATA: dict[str, dict[str, str]] = {
    "Digital_Music": {
        "display_name": "Digital Music",
        "filename": "Digital_Music_5.json.gz",
    },
    "Luxury_Beauty": {
        "display_name": "Luxury Beauty",
        "filename": "Luxury_Beauty_5.json.gz",
    },
}


@dataclass
class DatasetBundle:
    source_train: pd.DataFrame
    source_val: pd.DataFrame
    source_test: pd.DataFrame
    target_val: pd.DataFrame
    target_test: pd.DataFrame
    dataset_summary: dict[str, Any]


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as output_file:
        shutil.copyfileobj(response, output_file)


def ensure_raw_domain_file(data_dir: Path, domain_name: str) -> Path:
    if domain_name not in DOMAIN_METADATA:
        raise ValueError(f"Unsupported domain: {domain_name}")

    filename = DOMAIN_METADATA[domain_name]["filename"]
    raw_path = data_dir / "raw" / filename
    if raw_path.exists():
        return raw_path

    url = f"{AMAZON_DATA_BASE_URL}/{filename}"
    _download_file(url=url, destination=raw_path)
    return raw_path


def _parse_record(line: bytes) -> dict[str, Any]:
    decoded_line = line.decode("utf-8").strip()
    if not decoded_line:
        return {}

    try:
        return json.loads(decoded_line)
    except json.JSONDecodeError:
        return ast.literal_eval(decoded_line)


def load_domain_reviews(raw_path: Path, domain_name: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    display_name = DOMAIN_METADATA[domain_name]["display_name"]

    with gzip.open(raw_path, "rb") as compressed_file:
        for row_index, line in enumerate(compressed_file):
            review = _parse_record(line)
            text = join_review_fields(
                summary=review.get("summary", review.get("title")),
                review_text=review.get("reviewText", review.get("text")),
            )
            label = derive_label(review.get("overall", review.get("rating", review.get("stars"))))

            if label is None or not is_usable_text(text):
                continue

            records.append(
                {
                    "review_id": f"{domain_name}_{row_index}",
                    "text": text,
                    "label": label,
                    "rating": float(review.get("overall", review.get("rating", 0.0))),
                    "domain_key": domain_name,
                    "domain_name": display_name,
                    "asin": review.get("asin"),
                    "reviewer_id": review.get("reviewerID"),
                }
            )

    return pd.DataFrame.from_records(records)


def _balanced_subsample(frame: pd.DataFrame, max_examples: int, random_state: int) -> pd.DataFrame:
    if len(frame) <= max_examples:
        return frame.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    label_values = sorted(frame["label"].unique())
    per_label_cap = max_examples // len(label_values)
    sampled_parts: list[pd.DataFrame] = []
    sampled_indexes: list[int] = []

    for label in label_values:
        label_frame = frame[frame["label"] == label]
        take_count = min(len(label_frame), per_label_cap)
        sampled_frame = label_frame.sample(n=take_count, random_state=random_state)
        sampled_parts.append(sampled_frame)
        sampled_indexes.extend(sampled_frame.index.tolist())

    selected_count = sum(len(part) for part in sampled_parts)
    remaining_count = max_examples - selected_count
    if remaining_count > 0:
        remainder = frame.drop(index=sampled_indexes)
        if not remainder.empty:
            sampled_parts.append(
                remainder.sample(n=min(remaining_count, len(remainder)), random_state=random_state)
            )

    return (
        pd.concat(sampled_parts, ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )


def _split_source_frame(frame: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_frame, temp_frame = train_test_split(
        frame,
        test_size=0.4,
        random_state=random_state,
        stratify=frame["label"],
    )
    val_frame, test_frame = train_test_split(
        temp_frame,
        test_size=0.5,
        random_state=random_state,
        stratify=temp_frame["label"],
    )
    return (
        train_frame.reset_index(drop=True),
        val_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
    )


def _split_target_frame(frame: pd.DataFrame, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    val_frame, test_frame = train_test_split(
        frame,
        test_size=0.5,
        random_state=random_state,
        stratify=frame["label"],
    )
    return val_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def prepare_domain_splits(
    data_dir: Path,
    source_domain: str = DEFAULT_SOURCE_DOMAIN,
    target_domain: str = DEFAULT_TARGET_DOMAIN,
    max_source_examples: int = 12000,
    max_target_examples: int = 8000,
    random_state: int = 42,
    save_prepared_csv: bool = True,
) -> DatasetBundle:
    source_raw_path = ensure_raw_domain_file(data_dir=data_dir, domain_name=source_domain)
    target_raw_path = ensure_raw_domain_file(data_dir=data_dir, domain_name=target_domain)

    source_frame = load_domain_reviews(raw_path=source_raw_path, domain_name=source_domain)
    target_frame = load_domain_reviews(raw_path=target_raw_path, domain_name=target_domain)

    source_sampled = _balanced_subsample(
        frame=source_frame,
        max_examples=max_source_examples,
        random_state=random_state,
    )
    target_sampled = _balanced_subsample(
        frame=target_frame,
        max_examples=max_target_examples,
        random_state=random_state,
    )

    if save_prepared_csv:
        prepared_frame = pd.concat([source_sampled, target_sampled], ignore_index=True)
        prepared_frame.to_csv(data_dir / "prepared_reviews.csv", index=False)

    source_train, source_val, source_test = _split_source_frame(
        frame=source_sampled,
        random_state=random_state,
    )
    target_val, target_test = _split_target_frame(
        frame=target_sampled,
        random_state=random_state,
    )

    dataset_summary = {
        "source_domain": DOMAIN_METADATA[source_domain]["display_name"],
        "target_domain": DOMAIN_METADATA[target_domain]["display_name"],
        "source_raw_reviews": int(len(source_frame)),
        "target_raw_reviews": int(len(target_frame)),
        "source_sampled_reviews": int(len(source_sampled)),
        "target_sampled_reviews": int(len(target_sampled)),
        "source_label_balance": source_sampled["label"].value_counts().sort_index().to_dict(),
        "target_label_balance": target_sampled["label"].value_counts().sort_index().to_dict(),
        "splits": {
            "source_train": int(len(source_train)),
            "source_val": int(len(source_val)),
            "source_test": int(len(source_test)),
            "target_val": int(len(target_val)),
            "target_test": int(len(target_test)),
        },
    }

    return DatasetBundle(
        source_train=source_train,
        source_val=source_val,
        source_test=source_test,
        target_val=target_val,
        target_test=target_test,
        dataset_summary=dataset_summary,
    )
