"""Built-in mock hardware profiles."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.hardware.profile import DriftModel, HardwareProfile


def _linear_chain(n: int) -> list[tuple[int, int]]:
    return [(i, i + 1) for i in range(n - 1)]


def _uniform(n: int, value: float) -> dict[int, float]:
    return {q: value for q in range(n)}


def _edge_errors(edges: list[tuple[int, int]], value: float) -> dict[str, float]:
    return {f"{a}-{b}": value for a, b in edges}


def toy_2q() -> HardwareProfile:
    edges = [(0, 1)]
    return HardwareProfile(
        backend_name="toy-2q",
        num_qubits=2,
        coupling_map=edges,
        single_qubit_error=_uniform(2, 0.0008),
        two_qubit_error=_edge_errors(edges, 0.009),
        readout_error=_uniform(2, 0.015),
        t1_us=_uniform(2, 120.0),
        t2_us=_uniform(2, 90.0),
        description="Minimal two-qubit backend for Bell-state experiments.",
    )


def toy_5q() -> HardwareProfile:
    edges = _linear_chain(5)
    return HardwareProfile(
        backend_name="toy-5q",
        num_qubits=5,
        coupling_map=edges,
        single_qubit_error={0: 0.0006, 1: 0.0009, 2: 0.0012, 3: 0.0008, 4: 0.0015},
        two_qubit_error={"0-1": 0.008, "1-2": 0.012, "2-3": 0.010, "3-4": 0.016},
        readout_error={0: 0.012, 1: 0.020, 2: 0.025, 3: 0.018, 4: 0.030},
        t1_us={0: 140.0, 1: 110.0, 2: 95.0, 3: 125.0, 4: 80.0},
        t2_us={0: 100.0, 1: 85.0, 2: 70.0, 3: 95.0, 4: 60.0},
        description="Five-qubit linear chain with per-qubit error variation.",
    )


def heavy_hex_mock() -> HardwareProfile:
    # A 12-qubit slice inspired by IBM heavy-hex connectivity: a ring with
    # sparse rungs, so most qubits have degree 2 and a few have degree 3.
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (4, 5), (5, 6), (6, 7), (7, 8),
        (8, 9), (9, 10), (10, 11), (11, 0),
        (1, 6), (4, 9),
    ]
    n = 12
    return HardwareProfile(
        backend_name="heavyhex-12q-mock",
        num_qubits=n,
        coupling_map=edges,
        single_qubit_error={q: 0.0004 + 0.0001 * (q % 4) for q in range(n)},
        two_qubit_error={f"{a}-{b}": 0.007 + 0.001 * ((a + b) % 5) for a, b in edges},
        readout_error={q: 0.010 + 0.002 * (q % 3) for q in range(n)},
        t1_us={q: 150.0 - 5.0 * (q % 6) for q in range(n)},
        t2_us={q: 110.0 - 4.0 * (q % 6) for q in range(n)},
        calibration_timestamp=datetime.now(timezone.utc) - timedelta(hours=6),
        description="IBM-style heavy-hex-inspired 12-qubit mock backend.",
    )


def noisy_edu() -> HardwareProfile:
    edges = _linear_chain(4)
    return HardwareProfile(
        backend_name="noisy-edu-4q",
        num_qubits=4,
        coupling_map=edges,
        single_qubit_error=_uniform(4, 0.004),
        two_qubit_error=_edge_errors(edges, 0.045),
        readout_error=_uniform(4, 0.06),
        t1_us=_uniform(4, 35.0),
        t2_us=_uniform(4, 22.0),
        calibration_timestamp=datetime.now(timezone.utc) - timedelta(hours=48),
        drift_model=DriftModel(
            error_growth_per_hour=0.002,
            readout_growth_per_hour=0.001,
            max_extra_uncertainty=0.25,
        ),
        description="Deliberately noisy educational backend with stale calibration.",
    )


def pristine_8q() -> HardwareProfile:
    n = 8
    edges = _linear_chain(n) + [(0, 7)]  # ring
    return HardwareProfile(
        backend_name="pristine-8q",
        num_qubits=n,
        coupling_map=edges,
        single_qubit_error=_uniform(n, 0.0001),
        two_qubit_error=_edge_errors(edges, 0.0015),
        readout_error=_uniform(n, 0.004),
        t1_us=_uniform(n, 400.0),
        t2_us=_uniform(n, 320.0),
        description="High-quality mock backend approaching fault-tolerant-era specs.",
    )


_BUILDERS = {
    "toy-2q": toy_2q,
    "toy-5q": toy_5q,
    "heavyhex-12q-mock": heavy_hex_mock,
    "noisy-edu-4q": noisy_edu,
    "pristine-8q": pristine_8q,
}


def list_profiles() -> list[str]:
    return sorted(_BUILDERS)


def get_profile(name: str) -> HardwareProfile:
    if name not in _BUILDERS:
        raise KeyError(
            f"unknown hardware profile '{name}'; available: {list_profiles()}"
        )
    return _BUILDERS[name]()


def all_profiles() -> list[HardwareProfile]:
    return [build() for build in _BUILDERS.values()]
