# Development Log

## 2026-06-11 — Phase 1: circuit IR, DSL parser, hardware profiles

- **Files changed:** `app/circuits/ir.py`, `app/circuits/dsl.py`,
  `app/hardware/profile.py`, `app/hardware/presets.py`
- **Feature added:** pydantic circuit IR with depth/usage/dependency
  analytics and JSON round-trip; DSL parser with line-numbered errors,
  `pi`-expression angles, and flexible measure syntax; hardware profile
  model with drift math; five preset backends.
- **Tests added:** `tests/test_circuits.py` (10), profile-loading tests in
  `tests/test_hardware.py`.
- **Known limitations:** no classical control flow, no barriers, no
  mid-circuit measurement semantics.
- **Next step:** topology estimator + ideal baseline.

## 2026-06-11 — Phase 2: transpilation estimator and ideal engine

- **Files changed:** `app/prediction/transpile.py`, `app/prediction/ideal.py`
- **Feature added:** greedy interaction-aware layout, BFS-distance SWAP/depth
  estimation, topology penalty and warnings; tensor-based statevector
  simulator (≤12 qubits) with measured-qubit marginalization.
- **Tests added:** mapping tests in `tests/test_hardware.py`, ideal-engine
  tests in `tests/test_prediction.py`.
- **Known limitations:** layout is greedy (no SABRE-style iterative
  improvement); routing cost is an estimate, not an actual routed circuit.
- **Next step:** the hardware output prediction engine.

## 2026-06-11 — Phase 3: hardware output prediction engine

- **Files changed:** `app/prediction/noise.py`, `app/prediction/engine.py`
- **Feature added:** seven-channel distribution-space noise pipeline (gate
  depolarizing, topology, crosstalk proxy, T1/T2 decoherence, leakage proxy,
  asymmetric readout confusion, calibration drift) with per-channel TVD
  accounting, survival-probability reliability score, drift-widened
  confidence intervals, and seeded multinomial shot sampling.
- **Tests added:** noise-channel and engine tests in
  `tests/test_prediction.py` (normalization, ground-state bias,
  reproducibility, contribution keys).
- **Known limitations:** classical-distribution approximation (no coherent
  or correlated errors); crosstalk/leakage rates are fixed proxies.
- **Next step:** learning correction layer.

## 2026-06-11 — Phase 4: learning correction layer

- **Files changed:** `app/learning/features.py`, `app/learning/synthetic.py`,
  `app/learning/correction.py`
- **Feature added:** synthetic hardware-run generation via hidden perturbed
  profiles; 13-feature extraction; residual-mixing-weight regression
  (ridge / random forest / gradient boosting) with save/load and TVD
  improvement metrics.
- **Tests added:** `tests/test_learning.py` (7).
- **Known limitations:** correction is a single scalar per run, not a
  per-bitstring adjustment; trained on synthetic, not real, data.
- **Next step:** agent council and report generator.

## 2026-06-11 — Phase 5: agent council and Markdown reports

- **Files changed:** `app/agents/council.py`, `app/reports/markdown.py`
- **Feature added:** seven deterministic agents (noise, topology, readout,
  decoherence, drift, learning, teaching) each returning summary / evidence /
  concerns / confidence / mitigation; council classification and combined
  mitigation strategy; full per-run Markdown report with distribution table.
- **Tests added:** `tests/test_agents_reports.py`.
- **Known limitations:** agents are rule-based heuristics by design.
- **Next step:** persistence, API, dashboard.

## 2026-06-11 — Phase 6: storage, API, dashboard

- **Files changed:** `app/storage/db.py`, `app/utils/service.py`,
  `app/api/main.py`, `app/dashboard/app.py`, `examples/*.qf`, `scripts/*`
- **Feature added:** nine SQLAlchemy tables on SQLite; shared service layer
  (predict → council → report → persist, synthetic training, report
  retrieval); ten FastAPI endpoints; nine-page Streamlit dashboard with
  histograms, CI plots, council viewer, training lab, report export and doc
  viewer; six example circuits; run scripts and an end-to-end demo.
- **Tests added:** `tests/test_api.py` (9) using FastAPI TestClient with an
  in-memory database fixture.
- **Known limitations:** dashboard calls the service layer directly rather
  than the HTTP API; no authentication.
- **Next step:** docs polish, full quality-gate run.

## 2026-06-11 — Phase 7: docs and quality gates

- **Files changed:** `README.md`, `docs/*.md`, `pyproject.toml`,
  `.env.example`, `.gitignore`, `app/circuits/qiskit_adapter.py`
- **Feature added:** full documentation set; optional Qiskit adapter (both
  directions, gracefully absent); packaging metadata.
- **Tests:** full suite green (47 passed); `python -m compileall .` clean;
  API + dashboard import checks and live Bell-prediction smoke test pass.
- **Known limitations:** see `docs/noise_model.md` § honest limitations.
- **Next step:** see README "next improvements" candidates — real-hardware
  calibration ingestion, SABRE-style routing, per-bitstring correction.
