"""Unified solver for mixed impressed-density and constrained conductors."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .frequency_solver import solve_frequency_domain
from .litz import LitzProperties
from .preprocess import AxisymmetricMesh


def solve_combined(
    mesh: AxisymmetricMesh,
    frequency: float,
    currents: Mapping[str, complex] | complex,
    *,
    relative_permeability: Mapping[str, float] | None = None,
    conductivity: float = 58e6,
    phasor_convention: str = "cosine",
    return_voltages: bool = False,
    current_modes: Mapping[str, str] | str = "total_current",
    fixed_current_density: Mapping[str, object] | None = None,
    litz_properties: Mapping[str, LitzProperties] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Solve a model containing mixed conductor excitation modes.

    ``current_modes`` selects ``"total_current"`` for solid conductors whose
    total current is enforced by a ring-voltage constraint, or
    ``"fixed_current_density"`` for conductors carrying an impressed density.
    The latter values are supplied through ``fixed_current_density`` and may
    be scalar, full-mesh, or conductor-element arrays.
    """
    return solve_frequency_domain(
        mesh,
        frequency=frequency,
        currents=currents,
        relative_permeability=relative_permeability,
        conductivity=conductivity,
        phasor_convention=phasor_convention,
        return_voltages=return_voltages,
        current_modes=current_modes,
        fixed_current_density=fixed_current_density,
        litz_properties=litz_properties,
    )
