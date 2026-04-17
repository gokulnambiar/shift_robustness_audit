from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def estimate_importance_weights(
    source_texts: Iterable[str],
    target_texts: Iterable[str],
    random_state: int = 42,
) -> tuple[np.ndarray, dict[str, float]]:
    source_list = list(source_texts)
    target_list = list(target_texts)
    combined_texts = source_list + target_list
    domain_labels = np.concatenate(
        [
            np.zeros(len(source_list), dtype=int),
            np.ones(len(target_list), dtype=int),
        ]
    )

    train_texts, valid_texts, train_labels, valid_labels = train_test_split(
        combined_texts,
        domain_labels,
        test_size=0.25,
        random_state=random_state,
        stratify=domain_labels,
    )

    validation_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=25000,
        sublinear_tf=True,
    )
    train_matrix = validation_vectorizer.fit_transform(train_texts)
    valid_matrix = validation_vectorizer.transform(valid_texts)

    validation_classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
    )
    validation_classifier.fit(train_matrix, train_labels)
    valid_probabilities = validation_classifier.predict_proba(valid_matrix)[:, 1]
    domain_auc = float(roc_auc_score(valid_labels, valid_probabilities))

    full_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=25000,
        sublinear_tf=True,
    )
    full_matrix = full_vectorizer.fit_transform(combined_texts)
    full_classifier = LogisticRegression(
        C=1.0,
        max_iter=1000,
    )
    full_classifier.fit(full_matrix, domain_labels)

    source_matrix = full_vectorizer.transform(source_list)
    target_probabilities = full_classifier.predict_proba(source_matrix)[:, 1]
    raw_weights = target_probabilities / np.clip(1.0 - target_probabilities, 1e-4, None)
    clipped_weights = np.clip(raw_weights, 0.2, 5.0)
    normalized_weights = clipped_weights / clipped_weights.mean()

    diagnostics = {
        "domain_auc": domain_auc,
        "mean_target_probability_on_source": float(target_probabilities.mean()),
        "min_weight": float(normalized_weights.min()),
        "max_weight": float(normalized_weights.max()),
        "mean_weight": float(normalized_weights.mean()),
    }
    return normalized_weights.astype(np.float32, copy=False), diagnostics
