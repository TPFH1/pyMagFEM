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
    relative_peak_mu_real: float = 0.1,
) -> NonlinearCoreMaterial:
    """Adapt fitted materialdatabase data to the nonlinear FEM material interface.

    ``relative_peak_mu_real`` is the fraction of the running peak that marks
    the start of the saturation tail. It must be strictly between zero and
    one and defaults to 0.1 (10%). The tail keeps the cutoff loss angle
    fixed and decays exponentially in magnitude toward one, matching the
    fitted magnitude slope at the cutoff. A cutoff already at unity stays
    there; otherwise the cutoff magnitude must have a finite negative slope.

    The fitted branch retains its lossless real floor of one. The tail
    limits the magnitude rather than the real part.
    """
    if b_min <= 0.0 or b_max <= b_min or b_points < 2:
        raise ValueError("invalid magnetic-flux interpolation grid")
    if not 0.0 < relative_peak_mu_real < 1.0:
        raise ValueError("relative_peak_mu_real must be strictly between 0 and 1")
    b_grid = np.geomspace(b_min, b_max, b_points)
    mu_real, mu_imag = permeability_data.fit_real_and_imaginary_part_at_f_and_T(
        frequency, temperature, b_grid
    )
    # materialdatabase returns relative permeability components; the FEM
    # assembly converts relative permeability to absolute permeability.
    mu_values = np.asarray(mu_real) - 1j * np.asarray(mu_imag)
    real_interp = lambda values: np.interp(values, b_grid, mu_values.real)
    imag_interp = lambda values: np.interp(values, b_grid, mu_values.imag)

    mu_real_samples = mu_values.real
    max_left = np.maximum.accumulate(mu_real_samples)
    saturation_hits = np.flatnonzero(mu_real_samples <= relative_peak_mu_real * max_left)

    def fitted_permeability(values: np.ndarray) -> np.ndarray:
        fitted_real = real_interp(values)
        fitted_imag = np.minimum(imag_interp(values), 0.0)
        return np.where(fitted_real <= 1.0, 1.0 + 0.0j, fitted_real + 1j * fitted_imag)

    saturation_b = None
    cutoff_mu = 1.0 + 0.0j
    cutoff_magnitude = 1.0
    decay_per_t = 0.0
    if saturation_hits.size:
        first = saturation_hits[0]
        saturation_b = b_grid[first]
        threshold = relative_peak_mu_real * max_left[first]
        if first > 0 and mu_real_samples[first - 1] > threshold:
            # The running peak is constant across the descending crossing segment.
            saturation_b = np.interp(
                threshold, mu_real_samples[[first, first - 1]], b_grid[[first, first - 1]],
            )
        cutoff_mu = fitted_permeability(np.asarray(saturation_b)).item()
        cutoff_magnitude = abs(cutoff_mu)
        if not np.isfinite(cutoff_magnitude):
            raise ValueError("saturation cutoff requires a finite permeability magnitude")
        if cutoff_magnitude > 1.0:
            if first == 0:
                raise ValueError("saturation cutoff has no fitted interval to its left")
            mu_slope_per_t = (mu_values[first] - mu_values[first - 1]) / (
                b_grid[first] - b_grid[first - 1]
            )
            # Project the complex slope onto the radial direction in the complex plane.
            magnitude_slope_per_t = np.real(
                np.conj(cutoff_mu) * mu_slope_per_t
            ) / cutoff_magnitude
            decay_per_t = -magnitude_slope_per_t / (cutoff_magnitude - 1.0)
            if not np.isfinite(decay_per_t) or decay_per_t <= 0.0:
                raise ValueError("saturation cutoff requires a finite negative magnitude slope")

    def relative_permeability(values: np.ndarray) -> np.ndarray:
        if saturation_b is None:
            return fitted_permeability(values)
        values = np.asarray(values)
        fitted = fitted_permeability(np.minimum(values, saturation_b))
        if cutoff_magnitude == 1.0:
            return fitted
        delta_b_t = np.maximum(values - saturation_b, 0.0)
        tail_magnitude = 1.0 + (cutoff_magnitude - 1.0) * np.exp(-decay_per_t * delta_b_t)
        tail = (cutoff_mu / cutoff_magnitude) * tail_magnitude
        return np.where(values > saturation_b, tail, fitted)

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
