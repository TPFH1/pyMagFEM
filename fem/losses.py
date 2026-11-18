"""Elementwise loss components used by plots and axisymmetric loss integration."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from .constitutive import core_properties, effective_conductivities
from .litz import LitzProperties
from .nonlinear import NonlinearCoreMaterial
from .physics import (
    core_electric_loss_density,
    electric_field_from_a_phi,
    litz_transport_loss_density,
    magnetic_loss_density,
)
from .preprocess import AxisymmetricMesh


@dataclass(frozen=True)
class LossDensities:
    """Time-average losses in W/m^3, evaluated from peak field phasors.

    Both component arrays have shape ``(n_elements,)`` in mesh triangle order.
    Electrical losses include solid Joule, Litz transport/skin, and core
    conduction/dielectric losses. Magnetic losses include core magnetic and
    Litz proximity losses. Unselected and lossless regions contain zero.
    """

    electrical: np.ndarray
    magnetic: np.ndarray

    @property
    def total(self) -> np.ndarray:
        return self.electrical + self.magnetic


def element_volumes(mesh: AxisymmetricMesh) -> np.ndarray:
    """Return revolved triangle volumes in m^3, in mesh element order.

    Mesh coordinates are ``(r, z)`` in metres. The centroid radius integrates
    the linear ``2*pi*r`` Jacobian exactly over each straight-sided triangle.
    """
    xy = mesh.points[mesh.triangles]
    edges = xy[:, 1:] - xy[:, :1]
    areas = np.abs(edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]) / 2
    return 2.0 * np.pi * xy[:, :, 0].mean(axis=1) * areas


def loss_densities(
    mesh: AxisymmetricMesh,
    frequency: float,
    *,
    a_phi: np.ndarray | None = None,
    current_densities: Mapping[str, np.ndarray] | None = None,
    magnetic_field: np.ndarray | None = None,
    conductivity: float = 58e6,
    conductor_conductivities: Mapping[str, complex] | None = None,
    litz_properties: Mapping[str, LitzProperties] | None = None,
    nonlinear_material: NonlinearCoreMaterial | None = None,
    core_permeability: complex | None = None,
    core_conductivity: complex | None = None,
    conductor_names: Sequence[str] | None = None,
    include_core: bool = True,
) -> LossDensities:
    """Evaluate the loss-density decomposition shared by plots and integration.

    ``frequency`` is in Hz. All fields are peak phasors: nodal ``a_phi`` has
    shape ``(n_nodes,)`` in Wb/m, each ``current_densities`` entry has shape
    ``(n_elements,)`` in A/m^2, and ``magnetic_field`` has shape
    ``(n_elements, 2)`` in A/m. Litz current density is averaged over the
    entire bundle area, not the copper strand area.

    Conductivity is in S/m. Effective ``litz_properties`` take precedence over
    ``conductor_conductivities``, then the scalar ``conductivity`` fallback.
    A ``nonlinear_material`` supplies core permeability at local ``|B|`` in
    tesla and its complex conductivity. Otherwise ``core_permeability``
    (absolute, in H/m) and ``core_conductivity`` override mesh core properties,
    whose loss angles and dielectric contribution are included once.

    ``conductor_names=None`` selects every conductor; an empty sequence skips
    all conductors. ``include_core=False`` skips the core independently.
    Returned arrays cover the full mesh and contain zero in skipped regions
    and lossless air; no plotting masks or volume weights are applied here.

    Raises ``ValueError`` for invalid frequency/conductivity, missing selected
    conductor densities, or missing fields needed for the selected losses.
    Core magnetic and Litz proximity losses require ``magnetic_field``.
    Nonlinear core losses and nonzero AC core electrical losses require
    ``a_phi``. Integrating a component against ``element_volumes`` gives watts.
    """
    if not np.isfinite(frequency) or frequency < 0:
        raise ValueError("frequency must be finite and non-negative")
    electrical = np.zeros(len(mesh.triangles))
    magnetic = np.zeros(len(mesh.triangles))
    regions = mesh.conductor_regions or {}
    names = list(regions) if conductor_names is None else list(conductor_names)
    sigmas = effective_conductivities(
        names, conductivity, conductor_conductivities, litz_properties,
    )
    for name in names:
        if name not in regions or current_densities is None or name not in current_densities:
            raise ValueError(f"missing conductor region or current density for {name}")
        elements = np.asarray(regions[name], dtype=int)
        density = np.asarray(current_densities[name])
        if density.shape != (len(mesh.triangles),):
            raise ValueError(f"current density for {name} must contain one value per element")
        litz = (litz_properties or {}).get(name)
        # Homogenized bundle J requires effective rho, not bare copper resistivity.
        resistivity = litz.resistivity if litz is not None else 1.0 / sigmas[name]
        electrical[elements] = litz_transport_loss_density(density[elements], resistivity)
        if litz is not None:
            if magnetic_field is None:
                raise ValueError("magnetic_field is required for Litz proximity loss")
            magnetic[elements] = magnetic_loss_density(
                magnetic_field[elements], litz.absolute_permeability, frequency,
            )
    core_elements = np.flatnonzero(np.asarray(mesh.regions) == "core")
    material = (mesh.materials or {}).get("core")
    if include_core and core_elements.size and (
        material is not None or nonlinear_material is not None
        or core_permeability is not None or core_conductivity is not None
    ):
        local_b = None
        if nonlinear_material is not None:
            if a_phi is None:
                raise ValueError("a_phi is required for nonlinear core losses")
            # postprocess also integrates these densities; defer its import to avoid a cycle.
            from .postprocess import magnetic_flux_density

            local_b = np.linalg.norm(magnetic_flux_density(mesh, a_phi)[core_elements], axis=1)
        mu, sigma = core_properties(
            material, frequency, nonlinear_material=nonlinear_material,
            flux_density=local_b, permeability=core_permeability, conductivity=core_conductivity,
        )
        if magnetic_field is None:
            raise ValueError("magnetic_field is required for core losses")
        magnetic[core_elements] = magnetic_loss_density(
            magnetic_field[core_elements], mu, frequency,
        )
        # Re(sigma) includes dielectric dissipation even when DC conductivity is zero.
        if np.real(sigma) != 0.0 and frequency != 0.0:
            if a_phi is None:
                raise ValueError("a_phi is required for core electrical losses")
            e_field = electric_field_from_a_phi(
                np.mean(a_phi[mesh.triangles[core_elements]], axis=1), frequency,
            )
            electrical[core_elements] = core_electric_loss_density(e_field, sigma)
    return LossDensities(electrical, magnetic)
