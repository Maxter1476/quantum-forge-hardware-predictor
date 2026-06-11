"""Learned correction model for predicted output distributions.

The model learns a single scalar per case: the *residual mixing weight* w
that best morphs the engine's prediction toward the observed (synthetic)
hardware distribution. Positive w mixes toward uniform (engine was too
optimistic); negative w sharpens toward the dominant outcomes (engine was
too pessimistic). At inference time the predicted w is applied as a
distribution-space correction.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from app.circuits.ir import CircuitIR
from app.hardware.profile import HardwareProfile
from app.learning.features import FEATURE_NAMES, extract_features
from app.learning.synthetic import SyntheticExample, empirical_distribution
from app.prediction.engine import HardwarePredictionEngine
from app.prediction.noise import dist_to_vector, total_variation_distance
from app.prediction.transpile import MappingEstimate, estimate_mapping

MODEL_TYPES = {
    "ridge": lambda: Ridge(alpha=1.0),
    "random_forest": lambda: RandomForestRegressor(n_estimators=120, random_state=0),
    "gradient_boosting": lambda: GradientBoostingRegressor(random_state=0),
}

W_MIN, W_MAX = -0.4, 0.6


def _apply_weight(pred: np.ndarray, w: float) -> np.ndarray:
    """Mix toward uniform (w>0) or sharpen away from it (w<0)."""
    uniform = np.full_like(pred, 1.0 / len(pred))
    out = (1.0 - w) * pred + w * uniform
    out = np.clip(out, 0.0, None)
    s = out.sum()
    return out / s if s > 0 else np.full_like(pred, 1.0 / len(pred))


def optimal_weight(pred: np.ndarray, truth: np.ndarray) -> float:
    """Grid-search the mixing weight that minimizes TVD to the truth."""
    grid = np.linspace(W_MIN, W_MAX, 101)
    tvds = [total_variation_distance(_apply_weight(pred, w), truth) for w in grid]
    return float(grid[int(np.argmin(tvds))])


@dataclass
class TrainingMetrics:
    model_type: str
    num_examples: int
    mae_weight: float
    mean_tvd_before: float
    mean_tvd_after: float
    improvement: float


class CorrectionModel:
    """Wraps an sklearn regressor that predicts the residual mixing weight."""

    def __init__(self, model_type: str = "random_forest"):
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {sorted(MODEL_TYPES)}")
        self.model_type = model_type
        self.regressor = MODEL_TYPES[model_type]()
        self.feature_names = list(FEATURE_NAMES)
        self.trained = False
        self.metrics: TrainingMetrics | None = None

    # -- training ------------------------------------------------------------

    def _build_xy(
        self, examples: list[SyntheticExample], engine: HardwarePredictionEngine
    ) -> tuple[np.ndarray, np.ndarray, list[float], list[float]]:
        xs, ys, tvd_before, tvd_after = [], [], [], []
        for ex in examples:
            mapping = estimate_mapping(ex.circuit, ex.profile)
            pred_result = engine.predict(ex.circuit, ex.profile, shots=ex_shots(ex))
            m = len(pred_result.measured_qubits)
            pred = dist_to_vector(pred_result.predicted_probabilities, m)
            truth = empirical_distribution(ex.true_counts, m)
            w = optimal_weight(pred, truth)
            xs.append(extract_features(ex.circuit, ex.profile, mapping))
            ys.append(w)
            tvd_before.append(total_variation_distance(pred, truth))
            tvd_after.append(total_variation_distance(_apply_weight(pred, w), truth))
        return np.array(xs), np.array(ys), tvd_before, tvd_after

    def train(
        self, examples: list[SyntheticExample], engine: HardwarePredictionEngine | None = None
    ) -> TrainingMetrics:
        if len(examples) < 4:
            raise ValueError("need at least 4 training examples")
        engine = engine or HardwarePredictionEngine()
        x, y, tvd_before, tvd_after = self._build_xy(examples, engine)
        self.regressor.fit(x, y)
        pred_w = np.clip(self.regressor.predict(x), W_MIN, W_MAX)
        self.trained = True
        self.metrics = TrainingMetrics(
            model_type=self.model_type,
            num_examples=len(examples),
            mae_weight=float(mean_absolute_error(y, pred_w)),
            mean_tvd_before=float(np.mean(tvd_before)),
            mean_tvd_after=float(np.mean(tvd_after)),
            improvement=float(np.mean(tvd_before) - np.mean(tvd_after)),
        )
        return self.metrics

    # -- inference -------------------------------------------------------------

    def predict_weight(
        self, circuit: CircuitIR, profile: HardwareProfile, mapping: MappingEstimate
    ) -> float:
        if not self.trained:
            raise RuntimeError("correction model has not been trained")
        feats = extract_features(circuit, profile, mapping).reshape(1, -1)
        return float(np.clip(self.regressor.predict(feats)[0], W_MIN, W_MAX))

    def correct(
        self,
        circuit: CircuitIR,
        profile: HardwareProfile,
        mapping: MappingEstimate,
        predicted_vec: np.ndarray,
    ) -> np.ndarray:
        w = self.predict_weight(circuit, profile, mapping)
        return _apply_weight(predicted_vec, w)

    # -- persistence -------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(
                {
                    "model_type": self.model_type,
                    "regressor": self.regressor,
                    "feature_names": self.feature_names,
                    "trained": self.trained,
                    "metrics": self.metrics,
                },
                fh,
            )

    @classmethod
    def load(cls, path: str | Path) -> "CorrectionModel":
        with open(path, "rb") as fh:
            blob: dict[str, Any] = pickle.load(fh)
        model = cls(model_type=blob["model_type"])
        model.regressor = blob["regressor"]
        model.feature_names = blob["feature_names"]
        model.trained = blob["trained"]
        model.metrics = blob["metrics"]
        return model


def ex_shots(ex: SyntheticExample) -> int:
    return max(sum(ex.true_counts.values()), 1)
