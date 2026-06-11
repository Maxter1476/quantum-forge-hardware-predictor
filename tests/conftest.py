"""Shared fixtures: every test gets an isolated in-memory database."""
from __future__ import annotations

import pytest

from app.storage import db as storage


@pytest.fixture(autouse=True)
def _memory_db():
    storage.init_db("sqlite:///:memory:")
    yield


@pytest.fixture
def bell_dsl() -> str:
    return "circuit bell:\nqubits 2\nh q0\ncx q0 q1\nmeasure all\n"
