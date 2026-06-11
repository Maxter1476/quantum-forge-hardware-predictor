# Learning Correction Layer

The analytic noise model is necessarily approximate. The learning layer
(`app/learning/`) trains a regressor to predict the *residual* between the
engine's prediction and observed hardware behavior, using synthetic data.

## Synthetic data generation (`synthetic.py`)

1. Generate random circuits (2–4 qubits, depth 3–10, ~35% two-qubit gates,
   measure all).
2. Build a **hidden-truth profile**: every error rate of the visible profile
   scaled by a random factor in [0.6, 1.8] (T1/T2 scaled inversely).
3. Run the prediction engine against the hidden profile and sample counts —
   these play the role of real hardware results. The visible profile is what
   the predictor sees, so a systematic residual exists to learn.

## Target: residual mixing weight

For each example, grid-search the scalar `w ∈ [−0.4, 0.6]` such that
`(1−w)·p_pred + w·uniform` minimizes TVD to the observed distribution.
Positive `w` means the engine was too optimistic; negative `w` means too
pessimistic (the correction sharpens).

## Features (`features.py`)

13 features: qubit count, depth, gate counts, two-qubit density, SWAP
estimate, topology penalty, average 1q/2q/readout errors, average T1/T2,
calibration age.

## Models (`correction.py`)

`ridge`, `random_forest` (default), or `gradient_boosting` from
scikit-learn. `train()` reports MAE on the weight, mean TVD before/after the
oracle correction, and the improvement. Models pickle to
`models/correction_<backend>_<type>.pkl`; the service layer auto-loads the
newest model for a backend when `use_correction=true`.

## Applying the correction

At inference, the predicted weight is clamped to [−0.4, 0.6] and applied as
the same mixing transform, then the distribution is renormalized. The
`LearningCorrectionAgent` reports whether a correction was active.

## Training

- API: `POST /train/synthetic` with `{"backend": ..., "num_examples": ...,
  "shots": ..., "model_type": ...}`.
- Dashboard: page 7, *Learning Correction Lab*.

Each training run is recorded in the `synthetic_training_runs` and
`model_registry` tables.
