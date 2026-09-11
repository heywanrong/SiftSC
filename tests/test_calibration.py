from __future__ import annotations

import numpy as np
import pytest

from siftsc.calibration import fit_logistic
from siftsc.features import FEATURE_NAMES


def test_fit_logistic_returns_serializable_profile() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(40, len(FEATURE_NAMES)))
    labels = (features[:, 0] + features[:, 1] > 0).astype(int)
    fitted = fit_logistic(features, labels)
    payload = fitted.as_json(name="test", threshold=0.4, provenance={"n": 40})
    assert payload["gate_type"] == "logistic"
    assert len(payload["weights"]) == len(FEATURE_NAMES)
    assert payload["provenance"] == {"n": 40}


@pytest.mark.parametrize(
    ("features", "labels", "message"),
    [
        (np.zeros((4, 3)), np.zeros(4), "feature matrix"),
        (np.zeros((4, len(FEATURE_NAMES))), np.zeros(3), "shape"),
        (np.zeros((4, len(FEATURE_NAMES))), np.zeros(4), "both classes"),
    ],
)
def test_fit_logistic_validates_inputs(
    features: np.ndarray, labels: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_logistic(features, labels)
