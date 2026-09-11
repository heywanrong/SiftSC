"""Train and serialize the paper's logistic gate from labeled prompt records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .features import FEATURE_NAMES


@dataclass(frozen=True, slots=True)
class FittedLogisticProfile:
    weights: tuple[float, ...]
    intercept: float
    mean: tuple[float, ...]
    scale: tuple[float, ...]

    def as_json(self, *, name: str, threshold: float, provenance: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": name,
            "gate_type": "logistic",
            "feature_names": list(FEATURE_NAMES),
            "weights": list(self.weights),
            "intercept": self.intercept,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "threshold": threshold,
            "provenance": provenance,
        }


def fit_logistic(features: np.ndarray, labels: np.ndarray) -> FittedLogisticProfile:
    """Fit the class-balanced L2 logistic model specified by the paper."""

    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Calibration requires: pip install 'siftsc[calibrate]'") from exc
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"expected an (n, {len(FEATURE_NAMES)}) feature matrix")
    if labels.shape != (features.shape[0],):
        raise ValueError("labels must have shape (n,)")
    if len(np.unique(labels)) < 2:
        raise ValueError("calibration labels must contain both classes")
    scaler = StandardScaler().fit(features)
    model = LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        C=1.0,
        random_state=0,
    ).fit(scaler.transform(features), labels)
    return FittedLogisticProfile(
        weights=tuple(float(value) for value in model.coef_[0]),
        intercept=float(model.intercept_[0]),
        mean=tuple(float(value) for value in scaler.mean_),
        scale=tuple(float(value) for value in scaler.scale_),
    )
