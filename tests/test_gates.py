from __future__ import annotations

import json

import pytest

from siftsc.features import FEATURE_NAMES
from siftsc.gates import ConfidenceGate, LogisticGate, list_profiles, load_profile


def _features() -> dict[str, float]:
    return {name: 0.0 for name in FEATURE_NAMES}


def test_confidence_gate_runs_on_low_confidence() -> None:
    gate = ConfidenceGate(threshold=0.6, confidence_min=0.2, confidence_max=0.8)
    low = _features() | {"greedy_conf": 0.2}
    high = _features() | {"greedy_conf": 0.8}
    assert gate.score(low) == pytest.approx(1.0)
    assert gate.score(high) == pytest.approx(0.0)


def test_logistic_gate_known_probability() -> None:
    gate = LogisticGate(
        weights=(0.0,) * len(FEATURE_NAMES),
        intercept=0.0,
        mean=(0.0,) * len(FEATURE_NAMES),
        scale=(1.0,) * len(FEATURE_NAMES),
        threshold=0.5,
    )
    assert gate.score(_features()) == pytest.approx(0.5)


def test_load_external_profile(tmp_path) -> None:
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "gate_type": "confidence",
                "threshold": 0.7,
                "confidence_min": 0.1,
                "confidence_max": 0.9,
            }
        )
    )
    assert load_profile(str(path)).threshold == 0.7


def test_bundled_profiles_are_discoverable() -> None:
    assert "qwen05b-q4-confidence" in list_profiles()
    for profile in list_profiles():
        gate = load_profile(profile)
        assert 0.0 <= gate.score(_features()) <= 1.0


def test_unknown_profile_has_actionable_error() -> None:
    with pytest.raises(ValueError, match="available"):
        load_profile("does-not-exist")
