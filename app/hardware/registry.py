"""Hardware profile registry: built-in presets plus user-registered profiles.

Custom profiles are validated through :class:`HardwareProfile` and stored as
JSON under ``data/profiles/``. They can mirror real calibration data (e.g.
numbers copied from a provider's calibration page).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.hardware import presets
from app.hardware.profile import HardwareProfile

PROFILES_DIR = Path(__file__).resolve().parents[2] / "data" / "profiles"

_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{1,63}$")


def _custom_path(name: str) -> Path:
    return PROFILES_DIR / f"{name}.json"


def list_custom_profiles() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(p.stem for p in PROFILES_DIR.glob("*.json"))


def list_profiles() -> list[str]:
    """All available backend names, presets first."""
    return presets.list_profiles() + [
        n for n in list_custom_profiles() if n not in presets.list_profiles()
    ]


def get_profile(name: str) -> HardwareProfile:
    """Resolve a backend by name; presets shadow custom files."""
    try:
        return presets.get_profile(name)
    except KeyError:
        pass
    path = _custom_path(name)
    if path.exists():
        return HardwareProfile.model_validate(json.loads(path.read_text()))
    raise KeyError(
        f"unknown hardware profile '{name}'; available: {list_profiles()}"
    )


def register_profile(data: dict[str, Any], overwrite: bool = False) -> HardwareProfile:
    """Validate and persist a custom hardware profile.

    Pydantic does the heavy lifting; extra structural checks ensure the
    per-qubit maps cover the declared register.
    """
    profile = HardwareProfile.model_validate(data)
    name = profile.backend_name
    if not _NAME_RE.match(name):
        raise ValueError(
            "backend_name must be 2-64 chars of letters, digits, '-' or '_'"
        )
    if name in presets.list_profiles():
        raise ValueError(f"'{name}' is a built-in preset and cannot be replaced")

    n = profile.num_qubits
    for label, mapping in [
        ("single_qubit_error", profile.single_qubit_error),
        ("readout_error", profile.readout_error),
        ("t1_us", profile.t1_us),
        ("t2_us", profile.t2_us),
    ]:
        missing = [q for q in range(n) if q not in mapping]
        if missing:
            raise ValueError(f"{label} missing entries for qubits {missing}")
    for a, b in profile.coupling_map:
        if not 0 <= a < n or not 0 <= b < n:
            raise ValueError(f"coupling edge ({a},{b}) references nonexistent qubits")

    path = _custom_path(name)
    if path.exists() and not overwrite:
        raise ValueError(f"profile '{name}' already exists (set overwrite to replace)")
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(profile.model_dump_json(indent=2))
    return profile


def delete_custom_profile(name: str) -> bool:
    path = _custom_path(name)
    if path.exists():
        path.unlink()
        return True
    return False
