"""Linear P1 FEM solver for the axisymmetric A_phi magnetostatic equation."""
from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from model import load

from .preprocess import AxisymmetricMesh, build_mesh


def conductor_area(mesh: AxisymmetricMesh, conductor_index: int = 0) -> float:
    if conductor_index < 0 or conductor_index >= len(mesh.geometry["p_cond_center_1"]):
        raise IndexError("conductor index out of range")
    conductor_name = f"conductor_{conductor_index}"
    if conductor_name in (mesh.conductor_regions or {}):
        area = 0.0
        for element_index in mesh.conductor_regions[conductor_name]:
            nodes = mesh.triangles[int(element_index)]
            xy = mesh.points[nodes]
            area += abs(np.linalg.det(np.array([
                [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
            ]))) / 2.0
        if area > 0.0:
            return float(area)
    shape = mesh.geometry.get("conductor_shapes", ["round"] *
                              len(mesh.geometry["p_cond_center_1"]))[conductor_index]
    if shape == "rectangular":
        widths = mesh.geometry["conductor_widths"]
        heights = mesh.geometry["conductor_heights"]
        return float(widths[conductor_index]) * float(heights[conductor_index])
    radii = mesh.geometry.get("conductor_radii", [mesh.geometry["radius_cond_1"]] *
                              len(mesh.geometry["p_cond_center_1"]))
    return float(np.pi * float(radii[conductor_index]) ** 2)


def conductor_current(mesh: AxisymmetricMesh, current_density: Mapping[str, float], conductor_index: int = 0) -> float:
    """Return the prescribed circuit current represented by a conductor J map."""
    return float(current_density.get(f"conductor_{conductor_index}", 0.0)) * conductor_area(mesh, conductor_index)


def current_density_for_conductors(mesh: AxisymmetricMesh, circuit_currents: Mapping[str, float],
                                   signs: list[float] | None = None) -> dict[str, float]:
    """Map one series-circuit current to uniform per-turn J so each conductor carries I.

    ``signs`` controls the winding polarity; a series winding normally uses all
    ``+1`` values, while an opposition-connected turn can use ``-1``. This is
    the magnetostatic solid-conductor approximation; skin/proximity effects are
    intentionally not modeled.
    """
    signs = signs or [1.0] * len(mesh.geometry["p_cond_center_1"])
    if len(signs) != len(mesh.geometry["p_cond_center_1"]):
        raise ValueError("one current sign is required per conductor")
    current = float(next(iter(circuit_currents.values()), 0.0))
    return {f"conductor_{i}": current * float(signs[i]) / conductor_area(mesh, i)
            for i in range(len(signs))}

MU0 = 4e-7 * np.pi


_TRIANGLE_QUADRATURE = (
    (-27.0 / 48.0, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
    (25.0 / 48.0, (0.6, 0.2, 0.2)),
    (25.0 / 48.0, (0.2, 0.6, 0.2)),
    (25.0 / 48.0, (0.2, 0.2, 0.6)),
)


def solve_axisymmetric(mesh: AxisymmetricMesh, relative_permeability: Mapping[str, float] | None = None,
                       current_density: float | Mapping[str, object] = 0.0,
                       dirichlet: np.ndarray | None = None) -> np.ndarray:
    """Solve the axisymmetric ``A_phi`` equation using linear triangles.

    For the azimuthal vector potential, the weak form is
    ``integral [(r/mu) grad(A) dot grad(v) + A*v/(mu*r)] = integral [r*J*v]``.
    ``relative_permeability`` contains relative permeabilities, e.g.
    ``{"core": 1500}``.
    Nodes on ``r=0`` and the outer ferrite boundary are fixed to zero by
    default.  The returned nodal values are the physical ``A_phi`` values.
    """
    # Material inputs are relative permeabilities, matching the JSON/example
    # convention.  Convert them to absolute permeability for the weak form.
    mu_r = {"core": 1000.0, "air": 1.0, "winding": 1.0}
    if relative_permeability:
        mu_r.update(relative_permeability)
    mu = {region: MU0 * float(value) for region, value in mu_r.items()}
    n = len(mesh.points)
    matrix = lil_matrix((n, n), dtype=float)
    rhs = np.zeros(n)
    if isinstance(current_density, Mapping):
        currents = {"core": 0.0, "air": 0.0, "winding": 0.0}
        currents.update(current_density)
    else:
        currents = {"core": 0.0, "air": 0.0, "winding": float(current_density)}
    conductor_density = current_density if isinstance(current_density, Mapping) else {}
    for element_index, (nodes, region) in enumerate(zip(mesh.triangles, mesh.regions)):
        xy = mesh.points[nodes]
        det = np.linalg.det(np.array([[xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                                      [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]]]))
        area = abs(det) / 2
        if area == 0:
            continue
        gradients = np.array([[xy[1, 1] - xy[2, 1], xy[2, 1] - xy[0, 1], xy[0, 1] - xy[1, 1]],
                              [xy[2, 0] - xy[1, 0], xy[0, 0] - xy[2, 0], xy[1, 0] - xy[0, 0]]]) / det
        radius = max(float(xy[:, 0].mean()), 1e-12)
        material_mu = mu[str(region)]
        # Preserve the working formulation used before the frequency-domain
        # transition: solve for u = r*A_phi. The transformed basis is r*N_i.
        radial_basis = np.full(3, 1.0 / 3.0)
        grad_r_basis = radius * gradients + np.outer(
            np.array([1.0, 0.0]), radial_basis
        )
        ke = area * (grad_r_basis.T @ grad_r_basis) / (material_mu * radius)
        density = currents[str(region)]
        if mesh.conductor_regions:
            for conductor_name, element_indices in mesh.conductor_regions.items():
                if element_index in element_indices and conductor_name in conductor_density:
                    prescribed = np.asarray(conductor_density[conductor_name])
                    if prescribed.ndim == 0:
                        density = float(prescribed)
                    elif prescribed.shape == (len(mesh.triangles),):
                        density = float(prescribed[element_index])
                    elif prescribed.shape == (len(element_indices),):
                        local_index = np.flatnonzero(
                            np.asarray(element_indices, dtype=int) == element_index
                        )
                        density = float(prescribed[local_index[0]])
                    else:
                        raise ValueError(
                            f"current density for {conductor_name!r} must be scalar "
                            "or element-sized"
                        )
                    break
        fe = area * radius * density / 3.0 * np.ones(3)
        for i, ni in enumerate(nodes):
            rhs[ni] += fe[i]
            for j, nj in enumerate(nodes):
                matrix[ni, nj] += ke[i, j]
    fixed = np.unique(np.concatenate((mesh.boundaries.get("r0", []), mesh.boundaries.get("outer", []))))
    if dirichlet is not None:
        fixed = np.unique(np.concatenate((fixed, np.asarray(dirichlet, dtype=int))))
    free = np.ones(n, dtype=bool)
    free[fixed] = False
    values = np.zeros(n)
    reduced = matrix.tocsr()[free][:, free]
    if reduced.shape[0] == 0:
        return values
    values[free] = spsolve(reduced, rhs[free])
    if not np.isfinite(values).all():
        raise RuntimeError("FEM solve produced non-finite A_phi values; check mesh connectivity and boundary conditions")
    return values


def solve_json(path: str | Path, max_size: float | None = None, **kwargs) -> tuple[AxisymmetricMesh, np.ndarray]:
    """Build and solve a model described by a geometry JSON file."""
    mesh = build_mesh(load(path), max_size=max_size)
    return mesh, solve_axisymmetric(mesh, **kwargs)



def main() -> None:
    """CLI: ``python -m fem.solver geometry.json --max-size 0.001``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("--max-size", type=float)
    parser.add_argument("--current-density", type=float, default=0.0)
    parser.add_argument("--output", type=Path, help="Optional NumPy output path")
    args = parser.parse_args()
    mesh, solution = solve_json(args.geometry, args.max_size, current_density=args.current_density)
    if args.output:
        np.save(args.output, solution)
    else:
        print(json.dumps({"nodes": len(mesh.points), "triangles": len(mesh.triangles),
                          "min": float(solution.min()), "max": float(solution.max())}))


if __name__ == "__main__":
    main()
