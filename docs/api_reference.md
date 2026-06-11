# API Reference

Base URL defaults to `http://127.0.0.1:8000`. Interactive docs at `/docs`.

## `GET /health`
Liveness check. → `{"status": "ok", ...}`

## `GET /hardware-profiles`
Summaries of all built-in backends.

## `GET /hardware-profiles/{name}`
Full profile JSON. 404 for unknown names.

## `POST /circuits/parse`
Body: `{"dsl": "<source>"}` → `{"ir": ..., "summary": ...}`.
422 with the line number on parse errors.

## `POST /predict`
Body:

```json
{
  "dsl": "circuit bell:\nqubits 2\nh q0\ncx q0 q1\nmeasure all\n",
  "backend": "toy-5q",
  "shots": 2048,
  "seed": 7,
  "use_correction": false
}
```

`ir` (circuit IR JSON) may be supplied instead of `dsl`. Response:

```json
{
  "run_id": "…",
  "result": {
    "ideal_probabilities": {"00": 0.5, "11": 0.5},
    "predicted_probabilities": {"00": 0.4795, "01": 0.028, "10": 0.0306, "11": 0.4618},
    "predicted_counts": {"00": 982, "...": 0},
    "confidence_intervals": {"00": [0.4578, 0.5011]},
    "reliability_score": 92.25,
    "error_contributions": {"gate_error": 0.0061, "readout": 0.0289, "...": 0},
    "dominant_error_sources": ["readout", "decoherence", "gate_error"],
    "mapping": {"layout": {"0": 3, "1": 2}, "added_swap_count": 0},
    "estimated_duration_us": 1.07,
    "explanations": ["…"]
  },
  "council": {
    "reliability_classification": "high",
    "agent_reports": [{"agent": "HardwareNoiseAgent", "...": 0}],
    "mitigation_strategy": ["…"],
    "beginner_explanation": "…",
    "advanced_explanation": "…"
  }
}
```

Every run is persisted (circuits, prediction_runs, predicted_outputs,
agent_reports tables).

## `POST /predict/report`
Same body as `/predict`; additionally renders a Markdown report, saves it to
`data/reports/<id>.md` and the `generated_reports` table, and returns
`report_id` + `report_markdown`.

## `POST /train/synthetic`
Body: `{"backend": "toy-5q", "num_examples": 60, "shots": 1024,
"model_type": "random_forest", "seed": 0}` → training metrics and the saved
model path. Model types: `ridge`, `random_forest`, `gradient_boosting`.

## `GET /reports` and `GET /reports/{report_id}`
List / fetch generated Markdown reports.

## `GET /examples`
The bundled example circuits as `{name: dsl_source}`.
