"""Nonlinear measured-material helpers for frequency-domain FEM."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .materials import relative_to_absolute_permeability
from .physics import complex_conductivity


@dataclass(frozen=True)
class NonlinearCoreMaterial:
    """Field-dependent relative permeability and core conductivity."""

    relative_permeability: Callable[[np.ndarray], np.ndarray]
    conductivity: complex
    temperature: float
    frequency: float
    b_floor: float = 1e-9

    @property
    def relative_permeability_model(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return local relative complex permeability interpolation."""
        return self.relative_permeability

    def relative_permeability_at(self, flux_density: np.ndarray) -> np.ndarray:
        magnitude = np.maximum(np.asarray(flux_density, dtype=float), self.b_floor)
        return np.asarray(self.relative_permeability(magnitude), dtype=complex)

    def absolute_permeability_at(self, flux_density: np.ndarray) -> np.ndarray:
        """Return local complex absolute permeability in H/m."""
        return np.asarray(
            relative_to_absolute_permeability(self.relative_permeability_at(flux_density)),
            dtype=complex,
        )

def material_from_materialdatabase(
    permeability_data: object,
    permittivity_data: object | None,
    *,
    frequency: float,
    temperature: float,
    b_min: float = 1e-6,
    b_max: float = 2.0,
    b_points: int = 200,
) -> NonlinearCoreMaterial:
    """Adapt materialdatabase fitted N49 data to the FEM material interface."""
    if b_min <= 0.0 or b_max <= b_min or b_points < 2:
        raise ValueError("invalid magnetic-flux interpolation grid")
    b_grid = np.geomspace(b_min, b_max, b_points)
    mu_real, mu_imag = permeability_data.fit_real_and_imaginary_part_at_f_and_T(
        frequency, temperature, b_grid
    )
    # materialdatabase returns relative permeability components; the FEM
    # assembly converts relative permeability to absolute permeability.
    mu_values = np.asarray(mu_real) - 1j * np.asarray(mu_imag)
    real_interp = lambda values: np.interp(values, b_grid, mu_values.real)
    imag_interp = lambda values: np.interp(values, b_grid, mu_values.imag)

    def relative_permeability(values: np.ndarray) -> np.ndarray:
        return real_interp(values) + 1j * imag_interp(values)

    conductivity = 0.0 + 0.0j
    if permittivity_data is not None:
        eps_real, eps_imag = permittivity_data.fit_real_and_imaginary_part_at_f_and_T(
            frequency, temperature
        )
        conductivity = complex_conductivity(
            0.0,
            float(np.asarray(eps_real).reshape(-1)[0]),
            0.0,
            frequency,
        )
        # materialdatabase returns the fitted dielectric loss as eps_imag;
        # convert it with the same negative-imaginary convention.
        epsilon_0 = 8.8541878128e-12
        conductivity = 1j * 2.0 * np.pi * frequency * epsilon_0 * (
            float(np.asarray(eps_real).reshape(-1)[0])
            - 1j * float(np.asarray(eps_imag).reshape(-1)[0])
        )
    return NonlinearCoreMaterial(
        relative_permeability, conductivity, temperature, frequency
    )
