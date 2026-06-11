# Examples

All example circuits live in `examples/*.qf` and are loadable from the
dashboard's Circuit Editor or `GET /examples`.

| file | what it demonstrates |
|------|----------------------|
| `bell.qf` | the canonical 2-qubit entangled pair; readout error visibly populates `01`/`10` |
| `ghz4.qf` | a 4-qubit entanglement chain; decoherence and gate error grow with depth |
| `superposition3.qf` | uniform superposition; noise barely changes an already-flat distribution, but reliability still reports gate/readout cost |
| `rotation_interference.qf` | parametric rotations + CZ interference; shows non-uniform ideal structure degrading |
| `deep_chain.qf` | a deliberately deep circuit; decoherence becomes a dominant source |
| `nonlocal_swap_stress.qf` | star-shaped interaction graph; on chain topologies the SWAP estimator and TopologyAgent light up |

## Suggested experiments

1. **Same circuit, different backends.** Run `bell.qf` on `pristine-8q`,
   `toy-5q`, and `noisy-edu-4q` and compare reliability scores
   (≈99 / ≈92 / ≈55) and histograms.
2. **Topology stress.** Run `nonlocal_swap_stress.qf` on `toy-5q` (chain —
   SWAPs required) vs `heavyhex-12q-mock` (more routing freedom).
3. **Drift.** `noisy-edu-4q` has 48-hour-old calibration; watch the
   DriftAgent's concern and the widened confidence intervals.
4. **Learned correction.** Train a model on `noisy-edu-4q` in the Learning
   Correction Lab, then re-run a prediction with *Use trained correction
   model* checked and compare.

## Programmatic example

```python
from app.circuits.dsl import parse_dsl
from app.hardware.presets import get_profile
from app.prediction.engine import HardwarePredictionEngine

circuit = parse_dsl(open("examples/ghz4.qf").read())
result = HardwarePredictionEngine().predict(
    circuit, get_profile("toy-5q"), shots=4096, seed=1
)
print(result.predicted_probabilities)
print(result.reliability_score, result.dominant_error_sources)
```
