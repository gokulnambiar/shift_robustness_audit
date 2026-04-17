from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression
import re
import ssl

TOKEN_RE = re.compile(r"[a-z0-9']+")


class FrozenSentenceEncoder:
    def __init__(
        self,
        model_name: str = "glove-wiki-gigaword-50",
        cache_dir: Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self._model = None
        self._vector_size: int | None = None

    def _load_model(self):
        if self._model is None:
            try:
                import gensim.downloader as api
            except ImportError as exc:
                raise RuntimeError(
                    "gensim is required for the embedding pipeline. "
                    "Install the project requirements first."
                ) from exc

            if self.cache_dir is not None:
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                import os

                os.environ["GENSIM_DATA_DIR"] = str(self.cache_dir)

            ssl._create_default_https_context = ssl._create_unverified_context
            self._model = api.load(self.model_name)
            self._vector_size = int(self._model.vector_size)
        return self._model

    def _encode_text(self, text: str) -> np.ndarray:
        model = self._load_model()
        assert self._vector_size is not None

        tokens = TOKEN_RE.findall(text.lower())
        vectors = [model[token] for token in tokens if token in model]
        if not vectors:
            return np.zeros(self._vector_size, dtype=np.float32)
        return np.mean(vectors, axis=0).astype(np.float32, copy=False)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        embeddings = [self._encode_text(text) for text in texts]
        matrix = np.vstack(embeddings)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-8, None)


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
