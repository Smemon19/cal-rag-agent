from __future__ import annotations

import asyncio
import re

import pytest

from rag_agent import run_rag_agent
from verify import verify_answer


def ask(q: str) -> str:
    return asyncio.run(run_rag_agent(q, n_results=5))


def test_slate_underlayment_116mph():
    a = ask("What type of underlayment is approved for use on a slate shingle roof with design wind speed 116 mph?")
    assert ("ASTM D226 Type II" in a) or ("ASTM D4869" in a)
    assert "1507.7" in a


def test_deflection_no_ceiling():
    a = ask("Deflection limits for roof members not supporting a ceiling (all load cases)?")
    assert "L/180" in a
    assert "L/120" in a
    assert "1604.3" in a


def test_machinery_impact():
    a = ask("Do I include additional factors for rooftop mechanical units?")
    assert "+20%" in a
    assert "+50%" in a
    assert "1607.10.2" in a


def test_fall_anchor():
    a = ask("Design load for fall protection anchorages?")
    assert "3100" in a
    assert "1607.10.4" in a


def test_vehicle_barrier():
    a = ask("Design load for vehicle barrier systems?")
    assert "6000" in a
    assert "1607.9" in a


def test_verifier_catches_missing_numbers():
    ctx = (
        "RULE SNIPPET (IBC 2018 §1607.9 – Vehicle barrier systems):\n"
        "- Concentrated load: 6000 lb (Applied horizontally in any direction.)\n"
    )
    bad = "Vehicle barriers must resist vehicle loads per IBC."
    ver = verify_answer(bad, ctx)
    assert ver["ok"] is False
    assert any("6000" in x for x in ver["missing"]["nums"])  # expects missing 6000 lb


def test_verifier_accepts_exact_answer():
    ctx = (
        "RULE SNIPPET (IBC 2018 §1607.9 – Vehicle barrier systems):\n"
        "- Concentrated load: 6000 lb (Applied horizontally in any direction.)\n"
    )
    good = "Vehicle barrier systems: 6,000 lb (≈26.7 kN) (IBC §1607.9)."
    ver = verify_answer(good, ctx)
    assert ver["ok"] is True


