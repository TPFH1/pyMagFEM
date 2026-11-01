"""Canonical magnetic-material constants and permeability conversions."""
from __future__ import annotations

import numpy as np

MU0 = 4e-7 * np.pi


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
