"""Executable readout-error mitigation (confusion-matrix inversion).

The agents *recommend* measurement-error mitigation; this module performs
it. The predicted distribution is multiplied by the inverse of the tensored
per-qubit readout confusion matrices (the same asymmetric convention the
prediction engine applies), negatives are clipped, and the result is
renormalized — the standard matrix-inversion mitigation used on real
hardware.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from app.hardware.profile import HardwareProfile
from app.prediction.engine import PredictionResult
from app.prediction.noise import (
    apply_bit_channel,
    dist_to_vector,
    total_variation_distance,
    vector_to_dist,
)

# Must match the asymmetry used in HardwarePredictionEngine step 6.
FLIP_0TO1_SCALE = 0.8
FLIP_1TO0_SCALE = 1.2


def _inverse_confusion(e: float) -> np.ndarray:
    """Inverse of the per-qubit readout confusion matrix for error rate e."""
    m = np.array(
        [
            [1.0 - FLIP_0TO1_SCALE * e, FLIP_1TO0_SCALE * e],
            [FLIP_0TO1_SCALE * e, 1.0 - FLIP_1TO0_SCALE * e],
        ]
    )
    return np.linalg.inv(m)


def mitigate_readout(
    result: PredictionResult, profile: HardwareProfile
) -> dict[str, Any]:
    """Undo the modeled readout confusion on a prediction result.

    Returns the mitigated distribution plus TVD-to-ideal before/after so the
    caller can see whether mitigation actually helped.
    """
    measured = result.measured_qubits
    m = len(measured)
    layout = result.mapping.layout

    vec = dist_to_vector(result.predicted_probabilities, m)
    for axis, q in enumerate(measured):
        phys = layout[q]
        e = profile.readout_error.get(phys, profile.avg_readout_error())
        vec = apply_bit_channel(vec, m, axis, _inverse_confusion(e))

    # Inversion can produce small negatives; clip and renormalize.
    vec = np.clip(vec, 0.0, None)
    total = vec.sum()
    if total <= 0:
        raise ValueError("readout mitigation produced an empty distribution")
    vec /= total

    ideal_vec = dist_to_vector(result.ideal_probabilities, m)
    pred_vec = dist_to_vector(result.predicted_probabilities, m)
    tvd_before = total_variation_distance(pred_vec, ideal_vec)
    tvd_after = total_variation_distance(vec, ideal_vec)

    return {
        "mitigated_probabilities": vector_to_dist(vec, m),
        "tvd_to_ideal_before": round(tvd_before, 6),
        "tvd_to_ideal_after": round(tvd_after, 6),
        "improvement": round(tvd_before - tvd_after, 6),
        "method": "per-qubit confusion-matrix inversion",
    }
