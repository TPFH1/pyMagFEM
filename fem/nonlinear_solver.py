"""Nonlinear fixed-point wrapper for field-dependent core materials."""
from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np

from .frequency_solver import solve_frequency_domain
from .nonlinear import NonlinearCoreMaterial
from .postprocess import magnetic_flux_density
from .preprocess import AxisymmetricMesh


def solve_nonlinear_frequency_domain(
    mesh: AxisymmetricMesh,
    frequency: float,
    currents: Mapping[str, complex] | complex,
    *,
    nonlinear_materials: Mapping[str, NonlinearCoreMaterial],
    max_iterations: int = 15,
    tolerance: float = 2e-2,
    relaxation: float = 0.85,
    progress: bool = True,
    **kwargs,
):
    """Iterate permeability at local flux density until the field converges.

    Each nonlinear material also supplies its complex region conductivity
    (including dielectric response), overriding static mesh data and matching
    post-processing. Conductivity is held fixed across iterations while local
    permeability follows the relaxed B field. Remaining ``kwargs``, including
    per-conductor properties, are forwarded to ``solve_frequency_domain``.
    """
    if (
        max_iterations <= 0
        or not np.isfinite(tolerance)
        or tolerance <= 0.0
        or not 0.0 < relaxation <= 1.0
    ):
        raise ValueError("invalid nonlinear iteration settings")
    previous = np.zeros((len(mesh.triangles), 2), dtype=float)
    convergence_floor = max(
        (material.b_floor for material in nonlinear_materials.values()),
        default=1e-9,
    )
    result = None
    assembly_cache: dict[str, object] = {}
    start_time = time.perf_counter()
    if progress:
        print(f"Nonlinear solve: starting ({max_iterations} max iterations)")
    # Keep electrical and magnetic properties from the same material model in every solve.
    material_conductivities = dict(kwargs.pop("material_conductivities", {}) or {})
    material_conductivities.update({
        region: material.conductivity for region, material in nonlinear_materials.items()
    })
    for iteration in range(max_iterations):
        iteration_start = time.perf_counter()
        relative_permeability = dict(
            kwargs.pop("relative_permeability", {}) or {}
        )
        for region, material in nonlinear_materials.items():
            region_values = material.relative_permeability_at(
                np.linalg.norm(previous, axis=1)
            )
            relative_permeability[region] = region_values
        result = solve_frequency_domain(
            mesh,
            frequency,
            currents,
            relative_permeability=relative_permeability,
            material_conductivities=material_conductivities,
            _assembly_cache=assembly_cache,
            **kwargs,
        )
        a_phi = result[0]
        # This is the only post-processing needed during the fixed-point loop.
        # magnetic_flux_density reuses cached mesh geometry and does not compute
        # H, losses, or visualization fields until the final solution is ready.
        current = magnetic_flux_density(mesh, a_phi)
        updated = relaxation * current + (1.0 - relaxation) * previous
        error = np.linalg.norm(updated - previous) / max(
            np.linalg.norm(updated), convergence_floor
        )
        previous = updated
        if progress:
            elapsed = time.perf_counter() - start_time
            iteration_time = time.perf_counter() - iteration_start
            print(
                f"Nonlinear solve: iteration {iteration + 1}/{max_iterations}, "
                f"relative error={error:.3e}, elapsed={elapsed:.2f}s, "
                f"iteration={iteration_time:.2f}s"
            )
        if error < tolerance:
            if progress:
                print(f"Nonlinear solve converged in {iteration + 1} iterations")
            return result
    raise RuntimeError(f"nonlinear frequency solve did not converge after {max_iterations} iterations")
