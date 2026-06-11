#!/usr/bin/env python3
"""End-to-end demo: predict the Bell circuit on every preset backend."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.circuits.dsl import parse_dsl
from app.hardware.presets import get_profile, list_profiles
from app.storage import db as storage
from app.utils import service

DSL = (Path(__file__).resolve().parents[1] / "examples" / "bell.qf").read_text()


def main() -> None:
    storage.init_db()
    circuit = parse_dsl(DSL)
    for name in list_profiles():
        payload = service.run_prediction(
            circuit,
            get_profile(name),
            shots=2048,
            seed=7,
            dsl_source=DSL,
            generate_report=(name == "toy-5q"),
        )
        r = payload["result"]
        print(f"\n=== {name} ===")
        print("predicted:", {k: round(v, 4) for k, v in r["predicted_probabilities"].items()})
        print("reliability:", r["reliability_score"], "/100,",
              "dominant:", ", ".join(r["dominant_error_sources"]))
        if payload.get("report_id"):
            print("report saved:", payload["report_id"])


if __name__ == "__main__":
    main()
