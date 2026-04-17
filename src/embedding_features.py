from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression


class FrozenSentenceEncoder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Path | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(
                    "sentence-transformers is required for the embedding pipeline. "
                    "Install the project requirements first."
                ) from exc

            cache_folder = str(self.cache_dir) if self.cache_dir is not None else None
            self._model = SentenceTransformer(self.model_name, cache_folder=cache_folder)
        return self._model

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        model = self._load_model()
        embeddings = model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32, copy=False)


def fit_embedding_classifier(
    embeddings: np.ndarray,
    labels: np.ndarray,
    c_value: float,
    sample_weight: np.ndarray | None = None,
) -> LogisticRegression:
    classifier = LogisticRegression(
        C=c_value,
        solver="liblinear",
        max_iter=2000,
    )
    classifier.fit(embeddings, labels, sample_weight=sample_weight)
    return classifier


def decision_scores(classifier: LogisticRegression, embeddings: np.ndarray) -> np.ndarray:
    if hasattr(classifier, "decision_function"):
        return classifier.decision_function(embeddings)
    return classifier.predict_proba(embeddings)[:, 1]
