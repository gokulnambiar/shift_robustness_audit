from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def fit_tfidf_classifier(
    texts: Iterable[str],
    labels: np.ndarray,
    c_value: float,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    pipeline = Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_df=0.95,
                    max_features=50000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=c_value,
                    dual="auto",
                    max_iter=5000,
                ),
            ),
        ]
    )

    fit_params: dict[str, np.ndarray] = {}
    if sample_weight is not None:
        fit_params["classifier__sample_weight"] = sample_weight

    pipeline.fit(list(texts), labels, **fit_params)
    return pipeline


def decision_scores(model: Pipeline, texts: Iterable[str]) -> np.ndarray:
    return model.decision_function(list(texts))
