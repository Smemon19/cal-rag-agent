"""Calculation tools for structural queries.

Each function is exposed as a Pydantic AI tool and focuses on small, reliable
computations that the agent can call when needed (deflection limits, unit
conversions, categorization, etc.).
"""

from __future__ import annotations

from typing import Dict, Any
import re

from pint import UnitRegistry


# Create a single UnitRegistry for all computations in this module
ureg = UnitRegistry()
Q_ = ureg.Quantity


def deflection_limit(context: Any, span_ft: float, limit: str) -> float:
    """Compute the maximum allowed deflection in inches given span (ft) and a ratio like 'L/180'.

    Args:
        span_ft: Clear span in feet.
        limit: String ratio in the form 'L/180', 'L/240', 'L/360', 'L/120' (case-insensitive; spaces allowed).

    Returns:
        Maximum deflection in inches as a float.

    Example:
        span_ft=30, limit='L/180' -> 30*12/180 = 2.0 inches.

    Note:
        This is a simple geometric limit check. For context/criteria see IBC Table 1604.3.
    """
    if span_ft < 0:
        raise ValueError("span_ft must be non-negative")
    if not limit:
        raise ValueError("limit is required, e.g., 'L/180'")

    m = re.search(r"(?i)L\s*/\s*(\d+)", limit)
    if not m:
        raise ValueError("limit must be in the form 'L/###', e.g., 'L/180'")
    denom = int(m.group(1))
    if denom <= 0:
        raise ValueError("limit denominator must be positive")

    span_in = (Q_(span_ft, ureg.foot)).to(ureg.inch).magnitude
    max_defl_in = span_in / denom
    return float(max_defl_in)


def vehicle_barrier_reaction(context: Any, load_lbs: float = 6000) -> Dict[str, Any]:
    """Return the standard vehicle barrier design load in both pounds and kN.

    Args:
        load_lbs: Load in pounds-force (default 6000 lb).

    Returns:
        A dict with 'pounds' and 'kN' (kilonewtons) numeric values.
    """
    if load_lbs <= 0:
        raise ValueError("load_lbs must be positive")
    p = Q_(load_lbs, ureg.pound_force)
    kn = p.to(ureg.kilonewton).magnitude
    return {"pounds": float(load_lbs), "kN": float(round(kn, 1))}


def fall_anchor_design_load(context: Any, persons: int = 1) -> Dict[str, Any]:
    """Compute design load for fall protection anchors.

    Args:
        persons: Number of persons attached (>= 1).

    Returns:
        Dict with 'pounds' and 'kN'. Uses 3100 lb per person.
    """
    if persons <= 0:
        raise ValueError("persons must be >= 1")
    total_lb = 3100 * persons
    kn = Q_(total_lb, ureg.pound_force).to(ureg.kilonewton).magnitude
    return {"pounds": float(total_lb), "kN": float(round(kn, 1))}


def wind_speed_category(context: Any, v_mph: float) -> str:
    """Categorize wind speed for underlayment tables.

    Args:
        v_mph: Basic wind speed in mph.

    Returns:
        'V < 140 mph' if v_mph < 140 else 'V ≥ 140 mph'.
    """
    if v_mph < 140:
        return "V < 140 mph"
    return "V ≥ 140 mph"


def machinery_impact_factor(context: Any, machine_type: str) -> Dict[str, Any]:
    """Return impact factor for machinery loads per IBC 1607.10.2.

    Args:
        machine_type: 'light' or 'reciprocating'.

    Returns:
        Dict with 'factor' (e.g., 1.20, 1.50) and 'note' with explanation and citation.

    Reference:
        IBC 1607.10.2 (Machinery) – example impact factors: +20% for light machinery, +50% for reciprocating machinery.
    """
    mt = (machine_type or "").strip().lower()
    if mt not in {"light", "reciprocating"}:
        raise ValueError("machine_type must be 'light' or 'reciprocating'")
    if mt == "light":
        return {
            "factor": 1.20,
            "note": "Light machinery: use 1.20 (20% impact). See IBC 1607.10.2.",
        }
    else:
        return {
            "factor": 1.50,
            "note": "Reciprocating machinery: use 1.50 (50% impact). See IBC 1607.10.2.",
        }


