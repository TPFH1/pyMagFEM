"""Litz-wire loss estimates based on Meeker's homogenized model."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .materials import MU0
from .preprocess import AxisymmetricMesh


@dataclass(frozen=True)
class LitzProperties:
    """Frequency-dependent effective properties of a Litz-wire region."""

    absolute_permeability: complex
    resistivity: complex
    conductivity: complex
    dimensionless_frequency: float

@dataclass(frozen=True)
class LitzLoss:
    """Peak-phasor Litz loss components in watts."""

    transport: float
    proximity: float

    @property
    def total(self) -> float:
        return self.transport + self.proximity


def meeker_litz_properties(
    frequency: float,
    *,
    strand_diameter: float,
    fill_factor: float,
    conductivity: float = 58e6,
) -> LitzProperties:
    """Return Meeker's effective Litz permeability and resistivity.

    The fitted coefficients are those of equations (9), (10), and (19)-(20)
    in Meeker, ``Continuum representations of skin/proximity effect losses in
    wound coils``. ``fill_factor`` is the copper fraction of the homogenized
    Litz region, and ``strand_diameter`` is the bare copper diameter.
    """
    if frequency < 0.0:
        raise ValueError("frequency must be non-negative")
    if strand_diameter <= 0.0 or conductivity <= 0.0:
        raise ValueError("strand diameter and conductivity must be positive")
    if not 0.0 < fill_factor <= 1.0:
        raise ValueError("fill_factor must be in the interval (0, 1]")
    radius = strand_diameter / 2.0
    omega = 2.0 * np.pi * frequency
    dimensionless_frequency = conductivity * MU0 * radius**2 * omega / 2.0
    c1 = (
        -0.0714373 * fill_factor**3
        + 0.0684158 * fill_factor**2
        + 0.687385 * fill_factor
        + 0.775607
    )
    c2 = 1.5 * fill_factor / c1
    c3 = (
        -0.215718 * fill_factor**3
        + 0.722321 * fill_factor**2
        - 0.00860551 * fill_factor
        + 0.882464
    )
    c4 = 0.425218 - np.log(fill_factor) / 2.0 - c3 / 3.0
    root_mu = np.sqrt(1j * c1 * dimensionless_frequency)
    root_rho = np.sqrt(1j * c3 * dimensionless_frequency)
    absolute_permeability = MU0 * (
        (1.0 - c2) + c2 * np.tanh(root_mu) / root_mu
        if root_mu != 0.0 else 1.0
    )
    resistivity = (
        1.0 / (conductivity * fill_factor)
        * (
            root_rho / np.tanh(root_rho) + 1j * c4 * dimensionless_frequency
            if root_rho != 0.0 else 1.0
        )
    )
    return LitzProperties(
        absolute_permeability=complex(absolute_permeability),
        resistivity=complex(resistivity),
        conductivity=complex(1.0 / resistivity),
        dimensionless_frequency=float(dimensionless_frequency),
    )


def _element_area(mesh: AxisymmetricMesh, element: int) -> tuple[float, float]:
    xy = mesh.points[mesh.triangles[element]]
    area = abs(np.linalg.det(np.array([
        [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
        [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
    ]))) / 2.0
    return area, float(np.mean(xy[:, 0]))


def litz_loss(
    mesh: AxisymmetricMesh,
    current_densities: Mapping[str, np.ndarray],
    magnetic_field: np.ndarray,
    conductor: str,
    frequency: float,
    *,
    strand_diameter: float,
    fill_factor: float,
    conductivity: float = 58e6,
) -> LitzLoss:
    """Estimate transport/skin and proximity losses for one Litz conductor.

    ``current_densities`` and ``magnetic_field`` use peak phasors. The
    proximity term is obtained from ``-omega*imag(mu_eff)*|H|^2/2`` and the
    transport term from ``real(rho_eff)*|J|^2/2``. Axisymmetric volume
    integration includes the ``2*pi*r`` Jacobian.
    """
    if magnetic_field.shape != (len(mesh.triangles), 2):
        raise ValueError("magnetic_field must contain one vector per element")
    if conductor not in mesh.conductor_regions:
        raise ValueError(f"unknown conductor {conductor!r}")
    density = np.asarray(current_densities[conductor])
    if density.shape != (len(mesh.triangles),):
        raise ValueError("current density must contain one value per element")
    properties = meeker_litz_properties(
        frequency,
        strand_diameter=strand_diameter,
        fill_factor=fill_factor,
        conductivity=conductivity,
    )
    omega = 2.0 * np.pi * frequency
    transport = 0.0
    proximity = 0.0
    for element in np.asarray(mesh.conductor_regions[conductor], dtype=int):
        area, radius = _element_area(mesh, int(element))
        volume = 2.0 * np.pi * radius * area
        transport += 0.5 * volume * properties.resistivity.real * abs(density[element]) ** 2
        proximity += -0.5 * volume * omega * properties.absolute_permeability.imag * np.linalg.norm(
            magnetic_field[element]
        ) ** 2
    return LitzLoss(float(transport), float(proximity))
