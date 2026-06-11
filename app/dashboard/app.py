"""Streamlit dashboard for the Quantum Forge Hardware Predictor.

Run with:  streamlit run app/dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import streamlit as st

from app.circuits.dsl import DSLParseError, parse_dsl
from app.hardware.presets import get_profile, list_profiles
from app.storage import db as storage
from app.utils import service

EXAMPLES_DIR = ROOT / "examples"
DOCS_DIR = ROOT / "docs"

st.set_page_config(page_title="Quantum Forge Hardware Predictor", layout="wide")
storage.init_db()

PAGES = [
    "1. Circuit Editor",
    "2. Hardware Profile Selector",
    "3. Output Prediction",
    "4. Noise Breakdown",
    "5. Confidence Intervals",
    "6. Agent Council Report",
    "7. Learning Correction Lab",
    "8. Generated Reports",
    "9. Documentation Viewer",
]

DEFAULT_DSL = (EXAMPLES_DIR / "bell.qf").read_text() if (
    EXAMPLES_DIR / "bell.qf"
).exists() else "circuit bell:\nqubits 2\nh q0\ncx q0 q1\nmeasure all\n"


def _state() -> st.session_state.__class__:
    if "dsl" not in st.session_state:
        st.session_state.dsl = DEFAULT_DSL
    if "backend" not in st.session_state:
        st.session_state.backend = "toy-5q"
    if "last_run" not in st.session_state:
        st.session_state.last_run = None
    return st.session_state


def _histogram(ideal: dict, predicted: dict, title: str):
    keys = sorted(set(ideal) | set(predicted))
    fig, ax = plt.subplots(figsize=(7, 3))
    x = range(len(keys))
    width = 0.4
    ax.bar([i - width / 2 for i in x], [ideal.get(k, 0) for k in keys], width, label="ideal")
    ax.bar(
        [i + width / 2 for i in x],
        [predicted.get(k, 0) for k in keys],
        width,
        label="predicted hardware",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(keys, rotation=45)
    ax.set_ylabel("probability")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def _run_prediction(shots: int, seed: int | None, use_correction: bool, with_report: bool):
    s = _state()
    try:
        circuit = parse_dsl(s.dsl)
        profile = get_profile(s.backend)
        payload = service.run_prediction(
            circuit,
            profile,
            shots=shots,
            seed=seed,
            use_correction=use_correction,
            dsl_source=s.dsl,
            generate_report=with_report,
        )
        s.last_run = payload
        st.success(
            f"Prediction complete on '{s.backend}' — reliability "
            f"{payload['result']['reliability_score']}/100"
        )
    except (DSLParseError, ValueError, KeyError) as exc:
        st.error(str(exc))


def page_circuit_editor():
    s = _state()
    st.header("Circuit Editor")
    examples = {p.stem: p.read_text() for p in sorted(EXAMPLES_DIR.glob("*.qf"))}
    col1, col2 = st.columns([2, 1])
    with col2:
        choice = st.selectbox("Load example", ["(keep current)"] + list(examples))
        if st.button("Load") and choice != "(keep current)":
            s.dsl = examples[choice]
            st.rerun()
    with col1:
        s.dsl = st.text_area("Circuit DSL", s.dsl, height=260)
    try:
        circuit = parse_dsl(s.dsl)
        st.subheader("Parsed circuit")
        st.json(circuit.summary())
        with st.expander("Full IR JSON"):
            st.json(circuit.model_dump())
    except DSLParseError as exc:
        st.error(f"Parse error — {exc}")


def page_profile_selector():
    s = _state()
    st.header("Hardware Profile Selector")
    s.backend = st.selectbox(
        "Backend", list_profiles(), index=list_profiles().index(s.backend)
    )
    profile = get_profile(s.backend)
    st.json(profile.summary())
    with st.expander("Full profile"):
        st.json(profile.model_dump(mode="json"))
    st.subheader("Coupling map")
    st.write(" — ".join(f"({a},{b})" for a, b in profile.coupling_map))


def page_prediction():
    s = _state()
    st.header("Output Prediction")
    st.caption(f"Circuit from editor; backend: **{s.backend}**")
    col1, col2, col3, col4 = st.columns(4)
    shots = col1.number_input("Shots", 16, 100000, 2048)
    seed = col2.number_input("Seed", 0, 10_000, 7)
    use_corr = col3.checkbox("Use trained correction model")
    with_report = col4.checkbox("Generate Markdown report", value=True)
    if st.button("Run prediction", type="primary"):
        _run_prediction(int(shots), int(seed), use_corr, with_report)
    if s.last_run:
        r = s.last_run["result"]
        st.pyplot(
            _histogram(
                r["ideal_probabilities"],
                r["predicted_probabilities"],
                "Ideal vs predicted hardware output",
            )
        )
        st.subheader("Predicted counts")
        st.json(r["predicted_counts"])
        st.metric("Reliability score", f"{r['reliability_score']}/100")
        if r["explanations"]:
            st.subheader("Engine notes")
            for e in r["explanations"]:
                st.write("- " + e)


def page_noise_breakdown():
    s = _state()
    st.header("Noise Breakdown")
    if not s.last_run:
        st.info("Run a prediction first (page 3).")
        return
    r = s.last_run["result"]
    contrib = r["error_contributions"]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(list(contrib.keys()), list(contrib.values()))
    ax.set_xlabel("TVD contribution")
    ax.set_title("Error source contributions")
    fig.tight_layout()
    st.pyplot(fig)
    st.write("**Dominant sources:**", ", ".join(r["dominant_error_sources"]) or "none")
    m = r["mapping"]
    st.subheader("Topology fit")
    st.json(
        {
            "layout": m["layout"],
            "added_swap_count": m["added_swap_count"],
            "added_cnot_count": m["added_cnot_count"],
            "topology_penalty": m["topology_penalty"],
            "hardware_friendly": m["hardware_friendly"],
            "warnings": m["warnings"],
        }
    )


def page_confidence():
    s = _state()
    st.header("Confidence Intervals")
    if not s.last_run:
        st.info("Run a prediction first (page 3).")
        return
    r = s.last_run["result"]
    cis = r["confidence_intervals"]
    preds = r["predicted_probabilities"]
    keys = sorted(cis)
    fig, ax = plt.subplots(figsize=(7, 3))
    centers = [preds.get(k, 0) for k in keys]
    err_low = [max(0, preds.get(k, 0) - cis[k][0]) for k in keys]
    err_high = [max(0, cis[k][1] - preds.get(k, 0)) for k in keys]
    ax.errorbar(range(len(keys)), centers, yerr=[err_low, err_high], fmt="o", capsize=4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=45)
    ax.set_ylabel("probability")
    ax.set_title("95% confidence intervals (drift-widened)")
    fig.tight_layout()
    st.pyplot(fig)
    st.table(
        [
            {"bitstring": k, "predicted": round(preds.get(k, 0), 4),
             "ci_low": round(cis[k][0], 4), "ci_high": round(cis[k][1], 4)}
            for k in keys
        ]
    )


def page_council():
    s = _state()
    st.header("Agent Council Report")
    if not s.last_run:
        st.info("Run a prediction first (page 3).")
        return
    c = s.last_run["council"]
    st.metric(
        "Reliability classification",
        c["reliability_classification"].upper(),
        f"{c['reliability_score']}/100",
    )
    st.subheader("Plain-language summary")
    st.write(c["beginner_explanation"])
    st.subheader("Technical summary")
    st.write(c["advanced_explanation"])
    st.subheader("Mitigation strategy")
    for mstep in c["mitigation_strategy"]:
        st.write("1. " + mstep)
    st.subheader("Individual agents")
    for rep in c["agent_reports"]:
        with st.expander(f"{rep['agent']} (confidence {rep['confidence']:.2f})"):
            st.write(rep["summary"])
            if rep["evidence"]:
                st.write("Evidence:")
                for e in rep["evidence"]:
                    st.write("- " + e)
            if rep["concerns"]:
                st.warning("\n".join(rep["concerns"]))
            if rep["mitigation"]:
                st.info("Mitigation: " + rep["mitigation"])


def page_learning_lab():
    s = _state()
    st.header("Learning Correction Lab")
    st.write(
        "Train a correction model on synthetic 'hardware runs' generated from "
        "a hidden, perturbed version of the selected backend."
    )
    col1, col2, col3 = st.columns(3)
    n = col1.number_input("Synthetic examples", 8, 500, 40)
    shots = col2.number_input("Shots per example", 64, 16384, 512)
    model_type = col3.selectbox("Model", ["random_forest", "ridge", "gradient_boosting"])
    if st.button("Generate data and train", type="primary"):
        with st.spinner("Generating synthetic runs and training..."):
            out = service.train_synthetic(
                backend_name=s.backend,
                num_examples=int(n),
                shots=int(shots),
                model_type=model_type,
            )
        st.success(f"Model saved to {out['model_path']}")
        st.json(out["metrics"])
    existing = service.load_correction_model(s.backend)
    if existing and existing.metrics:
        st.subheader(f"Current model for '{s.backend}'")
        st.json(vars(existing.metrics))
    else:
        st.info(f"No trained correction model for '{s.backend}' yet.")


def page_reports():
    st.header("Generated Reports")
    reports = service.list_reports()
    if not reports:
        st.info("No reports yet — run a prediction with 'Generate Markdown report'.")
        return
    options = {f"{r['title']} ({r['created_at']})": r["report_id"] for r in reports}
    choice = st.selectbox("Report", list(options))
    found = service.get_report(options[choice])
    if found:
        st.download_button(
            "Export Markdown", found["markdown"], file_name=f"{found['report_id']}.md"
        )
        st.markdown(found["markdown"])


def page_docs():
    st.header("Documentation Viewer")
    docs = sorted(DOCS_DIR.glob("*.md")) if DOCS_DIR.exists() else []
    readme = ROOT / "README.md"
    files = ([readme] if readme.exists() else []) + docs
    if not files:
        st.info("No documentation found.")
        return
    choice = st.selectbox("Document", [f.name for f in files])
    chosen = next(f for f in files if f.name == choice)
    st.markdown(chosen.read_text())


def main() -> None:
    _state()
    st.sidebar.title("Quantum Forge")
    st.sidebar.caption("Hardware-level output prediction")
    page = st.sidebar.radio("Page", PAGES)
    dispatch = {
        PAGES[0]: page_circuit_editor,
        PAGES[1]: page_profile_selector,
        PAGES[2]: page_prediction,
        PAGES[3]: page_noise_breakdown,
        PAGES[4]: page_confidence,
        PAGES[5]: page_council,
        PAGES[6]: page_learning_lab,
        PAGES[7]: page_reports,
        PAGES[8]: page_docs,
    }
    dispatch[page]()


main()
