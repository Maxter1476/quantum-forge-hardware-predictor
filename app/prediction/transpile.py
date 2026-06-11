"""Hardware-aware transpilation estimator.

Not a full transpiler: it picks a naive layout, checks every two-qubit gate
against the coupling map, and estimates the SWAP overhead (and therefore the
extra depth and error) a real transpiler would incur.
"""
from __future__ import annotations

from collections import deque
from typing import Any

from pydantic import BaseModel, Field

from app.circuits.ir import CircuitIR
from app.hardware.profile import HardwareProfile

SWAP_CNOT_COST = 3  # one SWAP decomposes into three CNOTs


class MappingEstimate(BaseModel):
    """Result of the topology / transpilation analysis."""

    layout: dict[int, int]  # logical qubit -> physical qubit
    native_two_qubit_gates: int
    nonnative_two_qubit_gates: int
    added_swap_count: int
    added_cnot_count: int
    estimated_depth_after_mapping: int
    topology_penalty: float  # 0 (perfect fit) .. 1 (terrible fit)
    estimated_added_gate_error: float
    hardware_friendly: bool
    warnings: list[str] = Field(default_factory=list)
    layout_explanation: str = ""

    def summary(self) -> dict[str, Any]:
        return self.model_dump()


def _distance_matrix(profile: HardwareProfile) -> list[list[int]]:
    """All-pairs shortest path over the coupling map (BFS per qubit)."""
    n = profile.num_qubits
    inf = n + 10
    dist = [[inf] * n for _ in range(n)]
    adj: list[list[int]] = [profile.neighbors(q) for q in range(n)]
    for src in range(n):
        dist[src][src] = 0
        queue = deque([src])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if dist[src][v] > dist[src][u] + 1:
                    dist[src][v] = dist[src][u] + 1
                    queue.append(v)
    return dist


def _interaction_partners(circuit: CircuitIR) -> dict[int, set[int]]:
    partners: dict[int, set[int]] = {q: set() for q in range(circuit.num_qubits)}
    for ins in circuit.gate_instructions:
        if ins.is_two_qubit:
            a, b = ins.qubits
            partners[a].add(b)
            partners[b].add(a)
    return partners


def _choose_layout(circuit: CircuitIR, profile: HardwareProfile) -> dict[int, int]:
    """Naive greedy layout: busiest logical qubit gets the best-connected,
    lowest-error physical qubit; later qubits prefer physical neighbors of
    their already-placed interaction partners."""
    usage = circuit.qubit_usage()
    partners = _interaction_partners(circuit)
    logical_order = sorted(usage, key=lambda q: -usage[q])
    physical_score = {
        p: (
            len(profile.neighbors(p)),
            -profile.single_qubit_error.get(p, 0.0),
            -profile.readout_error.get(p, 0.0),
        )
        for p in range(profile.num_qubits)
    }
    layout: dict[int, int] = {}
    used: set[int] = set()
    for lq in logical_order:
        placed_partners = [layout[p] for p in partners[lq] if p in layout]
        candidates = [p for p in range(profile.num_qubits) if p not in used]
        # Prefer physical qubits adjacent to the most already-placed partners.
        candidates.sort(
            key=lambda p: (
                sum(1 for pp in placed_partners if profile.has_edge(p, pp)),
                physical_score[p],
            ),
            reverse=True,
        )
        layout[lq] = candidates[0]
        used.add(candidates[0])
    return layout


def estimate_mapping(circuit: CircuitIR, profile: HardwareProfile) -> MappingEstimate:
    """Estimate how the circuit maps onto the backend topology."""
    if circuit.num_qubits > profile.num_qubits:
        raise ValueError(
            f"circuit uses {circuit.num_qubits} qubits but backend "
            f"'{profile.backend_name}' only has {profile.num_qubits}"
        )

    layout = _choose_layout(circuit, profile)
    dist = _distance_matrix(profile)

    native = 0
    nonnative = 0
    added_swaps = 0
    warnings: list[str] = []

    for ins in circuit.gate_instructions:
        if not ins.is_two_qubit:
            continue
        pa, pb = layout[ins.qubits[0]], layout[ins.qubits[1]]
        d = dist[pa][pb]
        if d <= 1:
            native += 1
        else:
            nonnative += 1
            # Routing a gate over distance d needs roughly d-1 SWAPs.
            added_swaps += d - 1

    added_cnots = added_swaps * SWAP_CNOT_COST
    two_q_total = native + nonnative
    base_depth = circuit.depth
    # Each SWAP chain adds roughly its CNOT count to the critical path.
    est_depth = base_depth + added_cnots

    penalty = 0.0
    if two_q_total > 0:
        penalty = min(1.0, (nonnative + 0.5 * added_swaps) / two_q_total)

    avg_2q_err = profile.avg_two_qubit_error()
    added_error = 1.0 - (1.0 - avg_2q_err) ** added_cnots

    friendly = penalty < 0.34 and added_swaps <= max(2, two_q_total)
    if nonnative:
        warnings.append(
            f"{nonnative} two-qubit gate(s) act on non-adjacent physical qubits; "
            f"~{added_swaps} SWAP(s) ({added_cnots} extra CNOTs) required"
        )
    if not friendly:
        warnings.append(
            "circuit connectivity is hardware-unfriendly for this topology; "
            "consider re-ordering qubits or choosing a better-connected backend"
        )

    explanation = (
        "Logical qubits were sorted by gate usage and assigned to physical "
        "qubits sorted by connectivity then lowest error: "
        + ", ".join(f"q{lq}->phys{pq}" for lq, pq in sorted(layout.items()))
    )

    return MappingEstimate(
        layout=layout,
        native_two_qubit_gates=native,
        nonnative_two_qubit_gates=nonnative,
        added_swap_count=added_swaps,
        added_cnot_count=added_cnots,
        estimated_depth_after_mapping=est_depth,
        topology_penalty=round(penalty, 4),
        estimated_added_gate_error=round(added_error, 6),
        hardware_friendly=friendly,
        warnings=warnings,
        layout_explanation=explanation,
    )
