"""Optional matplotlib plotting helpers for axisymmetric FEM results."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum
from typing import Literal

import numpy as np
from matplotlib.colors import LogNorm

from .litz import LitzProperties
from .physics import (
    complex_conductivity,
    complex_relative_permeability,
    core_electric_loss_density,
    core_magnetic_loss_density,
    eddy_current_density,
    electric_field_from_a_phi,
    electric_loss_density,
    magnetic_loss_density,
    total_litz_loss_density,
)
from .postprocess import magnetic_field_intensity, magnetic_flux_density
from .preprocess import AxisymmetricMesh

MU0 = 4e-7 * np.pi


class PhysicalField(str, Enum):
    """Physical fields supported by :func:`plot_physical_field`."""

    B = "b"
    H = "h"
    J = "j"
    E = "e"
    MU = "mu"
    MU_REAL = "mu_real"
    MU_IMAG = "mu_imag"
    LOSS_DENSITY = "loss_density"
    MAG_LOSS_DENSITY = "mag_loss_density"
    ELECTRIC_LOSS_DENSITY = "electric_loss_density"


def plot_mesh(
    mesh: AxisymmetricMesh,
    *,
    ax=None,
    length_unit: str = "mm",
    color_by_region: bool = True,
):
    """Plot the triangular meridian mesh used by the FEM solver."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except ImportError as exc:
        raise RuntimeError("plot_mesh requires matplotlib") from exc
    if ax is None:
        _, ax = plt.subplots()
    scale, unit = _length_scale(length_unit)
    polygons = scale * mesh.points[mesh.triangles]
    if color_by_region:
        palette = {"core": "#d9d9d9", "air": "#cfe8ff", "winding": "#ffd9a8"}
        colors = [palette.get(str(region), "#ffffff") for region in mesh.regions]
    else:
        colors = "none"
    collection = PolyCollection(
        polygons,
        facecolors=colors,
        edgecolors="#4d4d4d",
        linewidths=0.25,
    )
    ax.add_collection(collection)
    _plot_domain_limits(mesh, ax, scale)
    ax.set_aspect("equal")
    ax.set_xlabel(f"r [{unit}]")
    ax.set_ylabel(f"z [{unit}]")
    ax.set_title("FEM mesh")
    return ax


def _length_scale(length_unit: str) -> tuple[float, str]:
    scales = {"m": 1.0, "mm": 1e3}
    try:
        return scales[length_unit], length_unit
    except KeyError as exc:
        raise ValueError("length_unit must be 'm' or 'mm'") from exc


def _plot_domain_limits(mesh: AxisymmetricMesh, ax, scale: float) -> None:
    limits = scale * mesh.points
    ax.set_xlim(float(limits[:, 0].min()), float(limits[:, 0].max()))
    ax.set_ylim(float(limits[:, 1].min()), float(limits[:, 1].max()))


def plot_physical_field(
    mesh: AxisymmetricMesh,
    field: PhysicalField | str,
    *,
    a_phi: np.ndarray | None = None,
    current_densities: Mapping[str, np.ndarray] | None = None,
    magnetic_field: np.ndarray | None = None,
    relative_permeability_model=None,
    nonlinear_material=None,
    litz_properties: LitzProperties | None = None,
    frequency: float = 0.0,
    domains: str | Sequence[str] | None = None,
    ax=None,
    levels: int = 20,
    color_scale: Literal["linear", "log"] = "linear",
    length_unit: str = "mm",
):
    """Plot a scalar physical field while preserving the full mesh extent.

    ``domains`` may contain ``"air"``, ``"winding"``, and ``"core"``.
    Omitted domains show the complete model. Loss-density fields use peak
    phasors and return W/m³.
    ``color_scale="linear"`` (default) uses linear colour normalization.
    ``color_scale="log"`` uses logarithmic colour normalization, masking
    non-positive values and requiring at least one positive value. This
    affects field colours and the colourbar, not the spatial axis scales.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
    except ImportError as exc:
        raise RuntimeError("plot_physical_field requires matplotlib") from exc
    selected = {domains} if isinstance(domains, str) else set(domains or ("air", "winding", "core"))
    if not selected.issubset({"air", "winding", "core"}):
        raise ValueError("domains must contain only 'air', 'winding', or 'core'")
    field = PhysicalField(field)
    if color_scale not in {"linear", "log"}:
        raise ValueError("color_scale must be 'linear' or 'log'")
    if ax is None:
        _, ax = plt.subplots()
    scale, unit = _length_scale(length_unit)
    values = np.full(len(mesh.triangles), np.nan, dtype=float)
    conductivity = 58e6
    if mesh.materials and "copper" in mesh.materials:
        conductivity = float(getattr(mesh.materials["copper"], "conductivity", conductivity) or conductivity)
    if field is PhysicalField.B:
        if a_phi is None: raise ValueError("a_phi is required for B")
        vectors = magnetic_flux_density(mesh, a_phi)
        values[:] = np.linalg.norm(vectors, axis=1)
        label = "|B| [T]"
    elif field is PhysicalField.H:
        if magnetic_field is None:
            if a_phi is None: raise ValueError("a_phi or magnetic_field is required for H")
            magnetic_field = magnetic_field_intensity(mesh, a_phi)
        values[:] = np.linalg.norm(magnetic_field, axis=1)
        label = "|H| [A/m]"
    elif field is PhysicalField.J:
        if current_densities is None: raise ValueError("current_densities is required for J")
        for name, elements in (mesh.conductor_regions or {}).items(): values[elements] = np.abs(current_densities[name][elements])
        if a_phi is not None:
            core_material = (mesh.materials or {}).get("core")
            core_elements = np.flatnonzero(np.asarray(mesh.regions) == "core")
            if core_material is not None and core_elements.size:
                electric_field = electric_field_from_a_phi(
                    a_phi[mesh.triangles[core_elements]].mean(axis=1), frequency
                )
                values[core_elements] = np.abs(eddy_current_density(
                    electric_field, float(getattr(core_material, "conductivity", 0.0))
                ))
        label = "|J| [A/m²]"
    elif field is PhysicalField.E:
        if a_phi is None: raise ValueError("a_phi is required for E")
        values[:] = np.abs(electric_field_from_a_phi(
            np.mean(a_phi[mesh.triangles], axis=1), frequency
        ))
        label = "|E| [V/m]"
    elif field is PhysicalField.MU:
        core_material = (mesh.materials or {}).get("core")
        if magnetic_field is None:
            raise ValueError("magnetic_field is required for MU")
        if a_phi is None:
            raise ValueError("a_phi is required for local B/H permeability")
        flux_density = magnetic_flux_density(mesh, a_phi)
        b_magnitude = np.linalg.norm(flux_density, axis=1)
        h_magnitude = np.linalg.norm(magnetic_field, axis=1)
        core_elements = np.flatnonzero(np.asarray(mesh.regions) == "core")
        values[:] = np.nan
        if relative_permeability_model is not None:
            values[core_elements] = np.real(
                relative_permeability_model(
                    np.maximum(b_magnitude[core_elements], 1e-12)
                )
            )
        else:
            values[core_elements] = b_magnitude[core_elements] / np.maximum(
                MU0 * h_magnitude[core_elements], 1e-30
            )
        label = "|B|/(μ₀|H|) [-]"
    elif field in {PhysicalField.MU_REAL, PhysicalField.MU_IMAG}:
        if magnetic_field is None:
            raise ValueError("magnetic_field is required for permeability")
        b_values = np.linalg.norm(magnetic_field, axis=1)
        core_material = (mesh.materials or {}).get("core")
        if (
            core_material is not None
            and hasattr(core_material, "relative_permeability_model")
        ):
            relative_permeability = np.asarray(
                core_material.relative_permeability_model(b_values), dtype=complex
            )
            values[:] = (
                relative_permeability.real
                if field is PhysicalField.MU_REAL
                else relative_permeability.imag
            )
        else:
            relative = float(getattr(core_material, "relative_permeability", 1.0)) if core_material is not None else 1.0
            values[:] = relative if field is PhysicalField.MU_REAL else 0.0
        label = "μr real [-]" if field is PhysicalField.MU_REAL else "μr imag [-]"
    else:
        if current_densities is None: raise ValueError("current_densities is required for loss density")
        if field is PhysicalField.ELECTRIC_LOSS_DENSITY:
            for name, elements in (mesh.conductor_regions or {}).items():
                values[elements] = electric_loss_density(
                    current_densities[name][elements], conductivity
                )
            label = "electric loss density [W/m³]"
        elif field is PhysicalField.LOSS_DENSITY:
            if magnetic_field is None:
                raise ValueError("magnetic_field is required for loss density")
            for name, elements in (mesh.conductor_regions or {}).items():
                if litz_properties is None:
                    values[elements] = electric_loss_density(
                        current_densities[name][elements], conductivity
                    )
                else:
                    values[elements] = total_litz_loss_density(
                        current_densities[name][elements], magnetic_field[elements],
                        litz_properties.resistivity,
                        litz_properties.absolute_permeability,
                        frequency,
                    )
            core_elements = np.flatnonzero(np.asarray(mesh.regions) == "core")
            core_material = (mesh.materials or {}).get("core")
            if core_elements.size and (core_material is not None or nonlinear_material is not None):
                if nonlinear_material is not None and a_phi is not None:
                    local_b = np.linalg.norm(
                        magnetic_flux_density(mesh, a_phi), axis=1
                    )
                    core_mu = nonlinear_material.absolute_permeability_at(local_b)
                    core_sigma = nonlinear_material.conductivity
                else:
                    if core_material is None:
                        raise ValueError("core material properties are required for loss density")
                    core_mu = MU0 * complex_relative_permeability(
                        float(getattr(core_material, "relative_permeability", 1.0)),
                        float(getattr(core_material, "permeability_loss_angle_deg", 0.0)),
                    )
                    core_sigma = complex_conductivity(
                        float(getattr(core_material, "conductivity", 0.0)),
                        float(getattr(core_material, "relative_permittivity", 0.0)),
                        float(getattr(core_material, "permittivity_loss_angle_deg", 0.0)),
                        frequency,
                    )
                if np.ndim(core_mu):
                    core_mu_values = core_mu[core_elements]
                else:
                    core_mu_values = core_mu
                values[core_elements] = core_magnetic_loss_density(
                    magnetic_field[core_elements], core_mu_values, frequency
                )
                if a_phi is not None:
                    electric_field = -1j * 2.0 * np.pi * frequency * (
                        a_phi[mesh.triangles[core_elements]].mean(axis=1)
                    )
                    values[core_elements] += core_electric_loss_density(
                        electric_field, core_sigma
                    )
            label = "loss density [W/m³]"
        else:
            if litz_properties is not None and magnetic_field is not None:
                for name, elements in (mesh.conductor_regions or {}).items():
                    values[elements] = magnetic_loss_density(
                        magnetic_field[elements],
                        litz_properties.absolute_permeability,
                        frequency,
                    )
            label = "magnetic loss density [W/m³]"
    mask = np.array([str(region) in selected for region in mesh.regions])
    values[~mask] = np.nan
    finite_values = values[np.isfinite(values)]
    norm = None
    if color_scale == "log":
        positive_values = finite_values[finite_values > 0.0]
        if positive_values.size == 0:
            raise ValueError("log color scale requires at least one positive value")
        values[values <= 0.0] = np.nan
        norm = LogNorm(vmin=float(positive_values.min()), vmax=float(positive_values.max()))
    collection = PolyCollection(
        scale * mesh.points[mesh.triangles],
        array=values,
        cmap="viridis",
        norm=norm,
        edgecolors="none",
        linewidths=0,
        antialiaseds=False,
    )
    collection.set_edgecolor("none")
    ax.add_collection(collection)
    ax.set_aspect("equal")
    ax.set_xlabel(f"r [{unit}]")
    ax.set_ylabel(f"z [{unit}]")
    _plot_domain_limits(mesh, ax, scale)
    ax.figure.colorbar(collection, ax=ax, label=label)
    return ax
