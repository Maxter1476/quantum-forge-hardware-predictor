# Quantum Forge Hardware Predictor

An AI-assisted **hardware-level quantum circuit output prediction system**.

Given a quantum circuit and a target hardware profile, it predicts what a
*real quantum processor* would output — not the textbook answer. It models
gate error, readout error, decoherence, crosstalk, leakage, calibration
drift, topology/routing overhead, and shot noise, then explains the result
through a council of deterministic analysis agents.

A small internal statevector simulator provides the ideal baseline; the main
deliverable is the **hardware output prediction layer** on top of it.

Beyond prediction it can also *act*: executable readout-error mitigation
(`apply_readout_mitigation`), custom hardware profile registration
(`POST /hardware-profiles`), multi-backend comparison
(`POST /predict/compare`), and a trainable correction layer.

## Install

```bash
cd quantum-forge-hardware-predictor
python3 -m pip install -e ".[dev]"
# or just the runtime deps:
python3 -m pip install numpy scipy fastapi uvicorn streamlit pydantic sqlalchemy matplotlib scikit-learn
```

Requires Python 3.11+. Qiskit interop is optional (`pip install qiskit`).

## Run the tests

```bash
python -m pytest
```

## Run the API

```bash
./scripts/run_api.sh            # http://127.0.0.1:8000/docs
```

## Run the dashboard

```bash
./scripts/run_dashboard.sh      # http://localhost:8501
```

## Quick demo

```bash
python3 scripts/demo_bell.py
```

## Example

Circuit DSL (`examples/bell.qf`):

```
circuit bell:
qubits 2
h q0
cx q0 q1
measure all
```

Predict via API:

```bash
curl -s -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"dsl": "circuit bell:\nqubits 2\nh q0\ncx q0 q1\nmeasure all\n", "backend": "toy-5q", "shots": 2048, "seed": 7}'
```

Typical predicted output on the `toy-5q` backend (ideal would be 50/50 over
`00`/`11`):

```json
{
  "predicted_probabilities": {"00": 0.4795, "01": 0.0280, "10": 0.0306, "11": 0.4618},
  "reliability_score": 92.25,
  "dominant_error_sources": ["readout", "decoherence", "gate_error"]
}
```

## Built-in mock backends

| name | qubits | character |
|------|--------|-----------|
| `toy-2q` | 2 | minimal Bell-state testbed |
| `toy-5q` | 5 | linear chain, per-qubit variation |
| `heavyhex-12q-mock` | 12 | IBM-style heavy-hex-inspired topology |
| `noisy-edu-4q` | 4 | deliberately noisy, stale calibration |
| `pristine-8q` | 8 | near-fault-tolerant-era quality |

## Documentation

See `docs/`: [architecture](docs/architecture.md),
[hardware model](docs/hardware_model.md), [noise model](docs/noise_model.md),
[learning layer](docs/learning_layer.md),
[circuit language](docs/circuit_language.md),
[API reference](docs/api_reference.md), [examples](docs/examples.md),
[development log](docs/development_log.md).
