# Hardware Model

A `HardwareProfile` (`app/hardware/profile.py`) describes a mock backend:

| field | meaning |
|-------|---------|
| `backend_name` | unique identifier |
| `num_qubits` | physical qubit count |
| `coupling_map` | undirected edges `(a, b)` where two-qubit gates are native |
| `basis_gates` | nominal native gate set (informational in the MVP) |
| `single_qubit_error` | depolarizing error per qubit per 1q gate |
| `two_qubit_error` | depolarizing error per edge (`"a-b"` keys) |
| `readout_error` | per-qubit measurement bit-flip probability |
| `t1_us`, `t2_us` | per-qubit relaxation / dephasing times (µs) |
| `gate_duration_ns` | `{"single": ..., "two": ...}` layer durations |
| `measurement_duration_ns` | readout time |
| `calibration_timestamp` | when these numbers were "measured" |
| `drift_model` | how uncertainty grows as calibration ages |

## Drift model

`drift_uncertainty()` returns
`min(age_hours × (error_growth + readout_growth), max_extra_uncertainty)`.
The prediction engine uses it twice: a small extra uniform mixing
(`0.5 × drift`) and a widening factor `1 + drift` on every confidence
interval.

## Built-in presets (`app/hardware/presets.py`)

- **toy-2q** — single edge, good parameters; for Bell-state experiments.
- **toy-5q** — linear chain with per-qubit/per-edge variation; the default.
- **heavyhex-12q-mock** — 12-qubit ring with sparse rungs, mimicking the
  degree distribution of IBM heavy-hex lattices; calibration 6 h old.
- **noisy-edu-4q** — error rates ~5–10× worse, T1/T2 ~4× shorter,
  calibration 48 h stale with an aggressive drift model. Useful for showing
  students what noise does.
- **pristine-8q** — ring topology with near-fault-tolerant-era parameters.

## Topology / transpilation estimator (`app/prediction/transpile.py`)

1. **Layout**: logical qubits sorted by gate usage are greedily placed on
   physical qubits, preferring neighbors of already-placed interaction
   partners, then high connectivity and low error.
2. **Routing estimate**: for each two-qubit gate, the BFS distance between
   its physical qubits is computed; distance *d* > 1 charges *d − 1* SWAPs,
   each costing 3 CNOTs.
3. **Outputs**: native/non-native gate counts, added SWAP/CNOT counts,
   estimated post-mapping depth, a 0–1 topology penalty, the estimated added
   gate error `1 − (1 − ε₂q)^addedCNOTs`, warnings, and a layout explanation.
