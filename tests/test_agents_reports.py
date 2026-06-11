"""Tests for the agent council and Markdown report generation."""
from app.agents.council import ALL_AGENTS, CouncilEngine
from app.circuits.dsl import parse_dsl
from app.hardware.presets import get_profile
from app.prediction.engine import HardwarePredictionEngine
from app.reports.markdown import generate_prediction_report, save_report


def _run(bell_dsl):
    c = parse_dsl(bell_dsl)
    p = get_profile("toy-5q")
    r = HardwarePredictionEngine().predict(c, p, shots=512, seed=11)
    return c, p, r


def test_council_runs_all_agents(bell_dsl):
    c, p, r = _run(bell_dsl)
    report = CouncilEngine().convene(c, p, r)
    assert len(report.agent_reports) == len(ALL_AGENTS) == 7
    names = {rep.agent for rep in report.agent_reports}
    assert "TeachingAgent" in names and "DriftAgent" in names
    assert report.reliability_classification in {"high", "moderate", "low"}
    assert report.mitigation_strategy
    assert report.beginner_explanation
    assert all(0 <= rep.confidence <= 1 for rep in report.agent_reports)


def test_markdown_report_contains_required_sections(bell_dsl, tmp_path):
    c, p, r = _run(bell_dsl)
    council = CouncilEngine().convene(c, p, r)
    md = generate_prediction_report(c, p, r, council)
    for needle in [
        "# Hardware Prediction Report",
        "## Circuit Summary",
        "## Hardware Profile",
        "## Topology Fit",
        "## Output Distributions",
        "## Error Source Breakdown",
        "## Agent Council Analysis",
        "## Mitigation Strategy",
        "Reliability score",
        "| bitstring |",
    ]:
        assert needle in md, f"missing section: {needle}"

    path = save_report(md, "test-report", directory=tmp_path)
    assert path.exists()
    assert path.read_text() == md
