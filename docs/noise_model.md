# Noise Model

The engine (`app/prediction/engine.py`) starts from the ideal distribution
over measured bitstrings and applies seven effects in order. Each effect
records the total variation distance (TVD) it introduces, reported in
`error_contributions`.

## 1. Gate depolarizing error

Every gate contributes its profile error rate (per-qubit for 1q gates,
per-edge for 2q gates, looked up through the chosen layout). The combined
infidelity `λ = 1 − Π(1 − εᵢ)` mixes the distribution with uniform:
`p ← (1−λ)p + λu`.

## 2. Topology / routing error

The estimated SWAP-induced CNOTs from the mapping estimator contribute
`1 − (1 − ε₂q)^addedCNOTs` of additional uniform mixing.

## 3. Crosstalk proxy

For each dependency layer, every pair of simultaneous two-qubit gates whose
physical qubits are topology-adjacent counts as one crosstalk event
(0.4% error each). Events compound multiplicatively into a mixing weight.

## 4. Decoherence

Circuit runtime is estimated layer-by-layer (two-qubit layers cost
`gate_duration_ns["two"]`, single-qubit layers `["single"]`, plus routing
overhead and measurement time). For each measured qubit:

- **T1 relaxation**: `p_relax = 1 − exp(−t/T1)` applied as the
  distribution-space channel `[[1, p],[0, 1−p]]` — probability mass on
  bit = 1 decays toward bit = 0. This produces the characteristic
  ground-state bias of real hardware.
- **T2 dephasing**: `1 − exp(−t/T2)`, scaled by 0.25, as uniform mixing
  (dephasing destroys interference, which at the distribution level mostly
  flattens structure).

## 5. Leakage proxy

Each two-qubit gate (including routing CNOTs) leaks 0.08% of probability
mass, redistributed uniformly — a stand-in for population escaping the
computational subspace and being misread.

## 6. Readout error

Per measured qubit, a column-stochastic confusion matrix is applied with
asymmetric flips (`0→1` at `0.8ε`, `1→0` at `1.2ε`), reflecting that
relaxation during readout makes `|1⟩` the more fragile state.

## 7. Calibration drift

`0.5 × drift_uncertainty()` of extra uniform mixing, plus every confidence
interval is widened by `1 + drift_uncertainty()`.

## Shot noise

Counts are drawn from a seeded multinomial over the predicted distribution.
Confidence intervals use the normal approximation
`p ± 1.96 · widen · √(p(1−p)/shots)`.

## Reliability score

`100 × Π(1 − λ_channel)` across all channels (with per-qubit decoherence
and readout factors), clamped to [0, 100]. ≥85 classifies as *high*,
60–85 *moderate*, below 60 *low*.

## Readout mitigation (executable)

`app/prediction/mitigation.py` performs confusion-matrix-inversion
measurement-error mitigation: each measured qubit's predicted distribution is
multiplied by the inverse of the same asymmetric confusion matrix the engine
applied, negatives are clipped, and the result is renormalized. The result
reports TVD-to-ideal before and after so you can verify it helped. Enable
with `apply_readout_mitigation` on `/predict` or the dashboard checkbox.

## Honest limitations

- Channels act on the classical output distribution, not the quantum state:
  coherent errors, correlated noise, and mid-circuit dynamics are out of
  scope.
- Depolarizing mixing toward uniform is pessimistic for structured states
  and optimistic for adversarial ones.
- The crosstalk and leakage models are proxies with fixed rates, not
  calibrated physics.
