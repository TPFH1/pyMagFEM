"""Voltage-excited frequency-domain solving helpers."""
from __future__ import annotations

from collections.abc import Mapping

from .frequency_solver import solve_frequency_domain
from .preprocess import AxisymmetricMesh


def solve_voltage_excitation(
    mesh: AxisymmetricMesh,
    frequency: float,
    voltage: complex,
    *,
    winding: str,
    turns: int | None = None,
    relative_permeability: Mapping[str, float | complex] | None = None,
    conductivity: float = 58e6,
    phasor_convention: str = "cosine",
):
    """Solve a single-winding model from an applied terminal voltage.

    The current is obtained self-consistently from the solver's ring-voltage
    formulation. ``turns`` converts terminal voltage to the per-turn ring
    voltage used by the axisymmetric model.
    """
    winding_names = mesh.geometry.get("conductor_windings", [])
    names = [
        name for index, name in enumerate((mesh.conductor_regions or {}).keys())
        if index < len(winding_names) and winding_names[index] == winding
    ]
    if not names:
        raise ValueError(f"unknown winding {winding!r}")
    all_fractions = mesh.geometry.get("conductor_current_fractions", [1.0] * len(winding_names))
    terminal_turns = sum(
        float(all_fractions[index])
        for index, name in enumerate(winding_names)
        if name == winding and index < len(all_fractions)
    )
    turn_count = terminal_turns if turns is None else turns
    if turn_count <= 0:
        raise ValueError("turns must be positive")
    source_voltage = complex(voltage)
    a_phi, densities, voltages = solve_frequency_domain(
        mesh,
        frequency,
        currents={winding: 1.0},
        relative_permeability=relative_permeability,
        conductivity=conductivity,
        phasor_convention=phasor_convention,
        return_voltages=True,
    )
    solved_source = sum(
        voltages.get(name, 0.0j) * float(all_fractions[index])
        for index, name in enumerate(winding_names)
        if index < len(all_fractions) and name == winding
    )
    if abs(solved_source) == 0.0:
        solved_source = sum(
            complex(voltage) for voltage in voltages.values()
        )
    if abs(solved_source) == 0.0:
        raise RuntimeError("voltage-excitation calibration produced zero source voltage")
    current_scale = source_voltage / solved_source
    return a_phi * current_scale, {
        key: density * current_scale for key, density in densities.items()
    }, current_scale
