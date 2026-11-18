"""Resolve the constitutive properties shared by solves and post-processing."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from model import FEMModel

from .litz import LitzProperties, meeker_litz_properties
from .materials import COPPER_CONDUCTIVITY_AT_20C_S_PER_M, MU0
from .nonlinear import NonlinearCoreMaterial
from .physics import complex_conductivity, complex_relative_permeability


def conductor_properties(
    model: FEMModel, frequency: float,
) -> tuple[dict[str, complex], dict[str, LitzProperties]]:
    """Return per-placement conductivity and effective Litz properties at Hz.

    Keys follow mesh naming (``conductor_0``, ...) in winding/turn order.
    Each turn's conductor assignment takes precedence over its winding's.
    Solid conductivity comes from the assigned material in S/m; Litz entries
    use Meeker's complex bundle conductivity, derived from that material's
    bulk conductivity, strand diameter, and bundle copper fill factor.

    Legacy models without conductor definitions or assigned material data
    retain the reference-copper fallback. Missing referenced definitions,
    invalid conductivity, and incomplete Litz geometry raise ``ValueError``.
    """
    materials = {material.name: material for material in model.materials}
    definitions = {conductor.name: conductor for conductor in model.conductors}
    conductivities = {}
    litz = {}
    for winding in model.windings:
        for turn in winding.turns:
            name = f"conductor_{len(conductivities)}"
            definition = definitions.get(turn.conductor or winding.conductor)
            if definition is None and definitions:
                raise ValueError(f"missing conductor definition for {name}")
            material = materials.get(definition.material if definition else "copper")
            sigma = (
                material.conductivity if material is not None
                else COPPER_CONDUCTIVITY_AT_20C_S_PER_M
            )
            if not np.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(f"conductivity for {name} must be finite and positive")
            conductivities[name] = complex(sigma)
            if definition is not None and definition.construction == "litz":
                if (
                    definition.radius is None or definition.radius <= 0.0
                    or definition.strands is None or definition.strands <= 0
                    or definition.strand_diameter is None or definition.strand_diameter <= 0.0
                ):
                    raise ValueError("Litz conductors require radius, strand count, and strand diameter")
                fill_factor = definition.strands * (
                    definition.strand_diameter / (2.0 * definition.radius)
                ) ** 2
                # Meeker takes bulk strand conductivity; its output includes the fill factor.
                properties = meeker_litz_properties(
                    frequency, strand_diameter=definition.strand_diameter,
                    fill_factor=fill_factor, conductivity=sigma,
                )
                litz[name] = properties
                conductivities[name] = properties.conductivity
    return conductivities, litz


def core_properties(
    material: object | None,
    frequency: float,
    *,
    nonlinear_material: NonlinearCoreMaterial | None = None,
    flux_density: np.ndarray | None = None,
    permeability: complex | None = None,
    conductivity: complex | None = None,
) -> tuple[complex | np.ndarray, complex]:
    """Resolve absolute core permeability in H/m and conductivity in S/m.

    A nonlinear material takes precedence over all linear inputs and requires
    ``flux_density`` magnitudes in tesla. Its permeability follows that array's
    shape; linear permeability is scalar. Otherwise explicit complex overrides
    take precedence over material attributes evaluated at ``frequency`` in Hz.

    Linear material loss angles are applied once, and complex conductivity
    includes displacement current and dielectric dissipation. With no material,
    unspecified properties describe a lossless, nonconductive vacuum.
    """
    if nonlinear_material is not None:
        if flux_density is None:
            raise ValueError("flux density is required for nonlinear core properties")
        return (
            nonlinear_material.absolute_permeability_at(flux_density),
            nonlinear_material.conductivity,
        )
    mu = permeability if permeability is not None else MU0 * complex_relative_permeability(
        float(getattr(material, "relative_permeability", 1.0)),
        float(getattr(material, "permeability_loss_angle_deg", 0.0)),
    )
    sigma = conductivity if conductivity is not None else complex_conductivity(
        float(getattr(material, "conductivity", 0.0)),
        float(getattr(material, "relative_permittivity", 0.0)),
        float(getattr(material, "permittivity_loss_angle_deg", 0.0)),
        frequency,
    )
    return mu, sigma


def effective_conductivities(
    names: list[str],
    conductivity: float,
    conductor_conductivities: Mapping[str, complex] | None = None,
    litz_properties: Mapping[str, LitzProperties] | None = None,
) -> dict[str, complex]:
    """Return complex conductivity in S/m for the requested conductor names.

    Precedence is effective Litz conductivity, per-conductor override, then
    scalar bulk fallback. This keeps assembly, voltage reconstruction, and
    loss evaluation consistent. Every selected value must be finite with a
    positive real part; invalid values raise ``ValueError``.
    """
    result = {}
    for name in names:
        litz = (litz_properties or {}).get(name)
        sigma = (
            litz.conductivity if litz is not None
            else (conductor_conductivities or {}).get(name, conductivity)
        )
        if not np.isfinite(sigma) or np.real(sigma) <= 0.0:
            raise ValueError(f"conductivity for {name} must have a finite positive real part")
        result[name] = complex(sigma)
    return result
