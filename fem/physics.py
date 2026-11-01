"""Reusable electromagnetic field and loss-density calculations."""
from __future__ import annotations

import numpy as np


def complex_relative_permeability(relative_permeability: float, loss_angle_deg: float = 0.0) -> complex:
    """Return relative permeability using a positive user loss angle.

    The returned phasor uses the physical negative imaginary-angle convention.
    """
    if relative_permeability <= 0.0:
        raise ValueError("relative_permeability must be positive")
    return complex(relative_permeability * np.cos(np.deg2rad(loss_angle_deg)),
                   -relative_permeability * np.sin(np.deg2rad(loss_angle_deg)))


def complex_conductivity(
    conductivity: float,
    relative_permittivity: float,
    loss_angle_deg: float,
    frequency: float,
) -> complex:
    """Return conductivity including dielectric displacement current.

    The user supplies a positive dielectric loss angle. Internally the
    permittivity uses the physical negative-angle convention.
    """
    if conductivity < 0.0 or relative_permittivity < 0.0 or frequency < 0.0:
        raise ValueError("material properties and frequency must be non-negative")
    epsilon_0 = 8.8541878128e-12
    epsilon = epsilon_0 * relative_permittivity * np.exp(-1j * np.deg2rad(loss_angle_deg))
    return complex(conductivity + 1j * 2.0 * np.pi * frequency * epsilon)


def electric_field_from_a_phi(a_phi: np.ndarray, frequency: float) -> np.ndarray:
    """Return the induced azimuthal electric-field phasor from A_phi."""
    if frequency < 0.0:
        raise ValueError("frequency must be non-negative")
    return -1j * 2.0 * np.pi * frequency * np.asarray(a_phi)


def eddy_current_density(electric_field: np.ndarray, conductivity: float) -> np.ndarray:
    """Return induced eddy-current density from an electric field."""
    if conductivity < 0.0:
        raise ValueError("conductivity must be non-negative")
    return conductivity * np.asarray(electric_field)


def electric_loss_density(current_density: np.ndarray, conductivity: float) -> np.ndarray:
    """Return peak-phasor Joule loss density in W/m³."""
    if conductivity <= 0.0:
        raise ValueError("conductivity must be positive")
    return 0.5 * np.abs(current_density) ** 2 / conductivity


def litz_transport_loss_density(current_density: np.ndarray, resistivity: complex) -> np.ndarray:
    """Return homogenized Litz transport loss density in W/m³."""
    return 0.5 * np.real(resistivity) * np.abs(current_density) ** 2


def core_magnetic_loss_density(
    magnetic_field: np.ndarray,
    absolute_permeability: complex,
    frequency: float,
) -> np.ndarray:
    """Return magnetic core-loss density in W/m³."""
    return magnetic_loss_density(magnetic_field, absolute_permeability, frequency)


def core_electric_loss_density(
    electric_field: np.ndarray,
    conductivity: complex,
) -> np.ndarray:
    """Return core electrical loss density from the complex conductivity."""
    if np.real(conductivity) < 0.0:
        raise ValueError("conductivity must have a non-negative real part")
    return 0.5 * np.real(conductivity) * np.abs(electric_field) ** 2


def magnetic_loss_density(
    magnetic_field: np.ndarray,
    absolute_permeability: complex,
    frequency: float,
) -> np.ndarray:
    """Return homogenized magnetic loss density in W/m³."""
    if frequency < 0.0:
        raise ValueError("frequency must be non-negative")
    omega = 2.0 * np.pi * frequency
    return -0.5 * omega * np.imag(absolute_permeability) * np.linalg.norm(
        magnetic_field, axis=-1
    ) ** 2


def core_loss_density(
    magnetic_field: np.ndarray,
    absolute_permeability: complex,
    frequency: float,
) -> np.ndarray:
    """Return total core loss density in W/m³."""
    return core_magnetic_loss_density(magnetic_field, absolute_permeability, frequency)


def total_litz_loss_density(
    current_density: np.ndarray,
    magnetic_field: np.ndarray,
    resistivity: complex,
    absolute_permeability: complex,
    frequency: float,
) -> np.ndarray:
    """Return transport plus proximity loss density in W/m³."""
    return litz_transport_loss_density(current_density, resistivity) + magnetic_loss_density(
        magnetic_field, absolute_permeability, frequency
    )
