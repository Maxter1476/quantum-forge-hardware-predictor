# Architecture

## Pipeline

```
DSL source ──parse──▶ CircuitIR ──┐
                                  ├─▶ MappingEstimate (topology fit, SWAP cost)
HardwareProfile ──────────────────┤
                                  ├─▶ IdealEngine (baseline distribution)
                                  ▼
                    HardwarePredictionEngine
        (gate error → topology → crosstalk → decoherence →
         leakage → readout → drift → optional learned correction)
                                  │
            ┌─────────────────────┼──────────────────────┐
            ▼                     ▼                      ▼
     PredictionResult      CouncilEngine          shot sampling + CIs
            │              (7 agents)                    │
            └──────────────┬──────────────┬──────────────┘
                           ▼              ▼
                  Markdown report     SQLite persistence
```

## Modules

| path | responsibility |
|------|----------------|
| `app/circuits/ir.py` | circuit IR, validation, depth/usage analytics, JSON |
| `app/circuits/dsl.py` | DSL parser with line-numbered errors |
| `app/circuits/qiskit_adapter.py` | optional Qiskit conversion (both directions) |
| `app/hardware/profile.py` | backend profile model, drift math |
| `app/hardware/presets.py` | five built-in mock backends |
| `app/prediction/transpile.py` | layout choice, SWAP/depth/error estimation |
| `app/prediction/ideal.py` | tensor-based statevector baseline (≤12 qubits) |
| `app/prediction/noise.py` | distribution-space noise channel primitives |
| `app/prediction/engine.py` | the hardware output prediction engine |
| `app/learning/` | synthetic data, features, correction model |
| `app/agents/council.py` | 7 deterministic agents + council |
| `app/reports/markdown.py` | per-run Markdown report generation |
| `app/storage/db.py` | SQLAlchemy models (9 tables), SQLite session |
| `app/utils/service.py` | orchestration shared by API and dashboard |
| `app/api/main.py` | FastAPI endpoints |
| `app/dashboard/app.py` | 9-page Streamlit dashboard |

## Key design decisions

- **Distribution-space noise modeling.** After the ideal baseline is
  computed, every hardware effect is a stochastic transformation of the
  classical probability vector over measured bitstrings. This keeps the
  predictor cheap (no density matrices) while capturing the *observable*
  consequences of each channel.
- **Per-source accounting.** Each channel records the total variation
  distance (TVD) it introduces, giving an honest "who distorted the
  histogram" breakdown that the agents and reports build on.
- **Reliability = survival probability.** The score multiplies the
  no-failure probabilities of all channels, mapped to 0–100.
- **The learned correction is a scalar residual.** The model predicts a
  single mixing weight (toward/away from uniform) rather than a full
  distribution, which trains reliably from tens of synthetic examples.
- **API and dashboard share one service layer**, so behavior cannot drift
  between the two front ends.
