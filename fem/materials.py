"""Canonical material constants and permeability conversions."""
from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi
COPPER_CONDUCTIVITY_AT_20C_S_PER_M = 58e6
COPPER_REFERENCE_TEMPERATURE_C = 20.0
COPPER_TEMPERATURE_COEFFICIENT_PER_C = 0.00393


def copper_conductivity_at_temperature_s_per_m(
    temperature_c: float,
    *,
    conductivity_at_reference_s_per_m: float = COPPER_CONDUCTIVITY_AT_20C_S_PER_M,
    reference_temperature_c: float = COPPER_REFERENCE_TEMPERATURE_C,
    temperature_coefficient_per_c: float = COPPER_TEMPERATURE_COEFFICIENT_PER_C,
) -> float:
    """Return copper conductivity at a temperature using a linear temperature coefficient."""
    values = (
        conductivity_at_reference_s_per_m,
        temperature_c,
        reference_temperature_c,
        temperature_coefficient_per_c,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Conductivity and temperature parameters must be finite.")
    if conductivity_at_reference_s_per_m <= 0.0:
        raise ValueError("Reference conductivity must be positive.")

    temperature_factor = 1.0 + temperature_coefficient_per_c * (
        temperature_c - reference_temperature_c
    )
    if temperature_factor <= 0.0:
        raise ValueError("Temperature correction must produce positive conductivity.")
    return conductivity_at_reference_s_per_m / temperature_factor


def relative_to_absolute_permeability(
    relative_permeability: complex | np.ndarray,
) -> float | complex | np.ndarray:
    """Convert relative permeability to absolute permeability in H/m."""
    return MU0 * relative_permeability


def absolute_to_relative_permeability(
    absolute_permeability: complex | np.ndarray,
) -> float | complex | np.ndarray:
    """Convert absolute permeability in H/m to relative permeability."""
    return absolute_permeability / MU0
