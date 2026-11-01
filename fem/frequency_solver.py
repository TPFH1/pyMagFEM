"""Frequency-domain axisymmetric eddy-current FEM for solid conductors.

The cosine phasor convention is used: ``Re{A exp(j omega t)}``.  A conductor
is represented as an axisymmetric ring voltage source: ``E_phi = V/(2*pi*r)``.
Thus ``J_phi = sigma*(V/(2*pi*r) - j*omega*A_phi)`` (with the sign adjusted
for the selected phasor convention).  The unknown ring voltage is determined
by ``integral(J_phi dS) = I`` over the meridian conductor section.  This keeps
the source physical at low frequency and retains skin-effect redistribution
through the induced A term.
"""
from __future__ import annotations

import time
from collections.abc import Mapping

import numpy as np
from scipy.sparse import bmat, coo_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from .litz import LitzProperties
from .nonlinear import NonlinearCoreMaterial
from .physics import complex_conductivity, complex_relative_permeability
from .preprocess import AxisymmetricMesh
from .solver import MU0, solve_axisymmetric

_TRIANGLE_QUADRATURE = (
    (-27.0 / 48.0, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)),
    (25.0 / 48.0, (0.6, 0.2, 0.2)),
    (25.0 / 48.0, (0.2, 0.6, 0.2)),
    (25.0 / 48.0, (0.2, 0.2, 0.6)),
)


def compare_domainc_blocks(
    mesh: AxisymmetricMesh,
    conductivity: float = 58e6,
    relative_permeability: Mapping[str, float] | None = None,
) -> dict[str, dict[str, np.ndarray | float]]:
    """Compare the ring-voltage blocks with a ``DomainC``-style reduction.

    GetDP's ``BF_RegionZ`` gives one constant electric coefficient per
    conductor region.  The current solver instead uses a physical ring
    voltage, whose electric field is ``V / (2*pi*r)``.  Both formulations
    share the magnetic block and the induced-current mass block, but their
    source, current-constraint, and voltage blocks have different radial
    weights.  This diagnostic assembles those blocks on the same mesh and
    reports Frobenius-norm differences; it does not alter the solver.

    The returned matrices use the same normalization as the saddle-point
    system assembled by :func:`solve_frequency_domain`: the axisymmetric
    ``2*pi`` Jacobian factor is omitted consistently from all blocks.
    """
    if conductivity < 0:
        raise ValueError("conductivity must be non-negative")
    mu_r = {"core": 1000.0, "air": 1.0, "winding": 1.0}
    if relative_permeability:
        mu_r.update(relative_permeability)
    names = list((mesh.conductor_regions or {}).keys())
    if not names:
        raise ValueError("mesh.conductor_regions is required")
    n = len(mesh.points)
    conductor_elements = {
        name: set(np.asarray(indices, dtype=int).tolist())
        for name, indices in mesh.conductor_regions.items()
    }
    conductive_regions = set().union(*conductor_elements.values())
    K = lil_matrix((n, n), dtype=float)
    M = lil_matrix((n, n), dtype=float)
    ring_B = lil_matrix((n, len(names)), dtype=float)
    ring_D = lil_matrix((len(names), n), dtype=float)
    ring_H = np.zeros((len(names), len(names)), dtype=float)
    domainc_B = lil_matrix((n, len(names)), dtype=float)
    domainc_D = lil_matrix((len(names), n), dtype=float)
    domainc_H = np.zeros((len(names), len(names)), dtype=float)

    for ei, (nodes, region) in enumerate(zip(mesh.triangles, mesh.regions)):
        xy = mesh.points[nodes]
        det = np.linalg.det(np.array([
            [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
            [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
        ]))
        area = abs(det) / 2.0
        if area == 0.0:
            continue
        gradients = np.array([
            [xy[1, 1] - xy[2, 1], xy[2, 1] - xy[0, 1], xy[0, 1] - xy[1, 1]],
            [xy[2, 0] - xy[1, 0], xy[0, 0] - xy[2, 0], xy[1, 0] - xy[0, 0]],
        ]) / det
        radius = max(float(xy[:, 0].mean()), 1e-12)
        material_mu = MU0 * float(mu_r.get(str(region), 1.0))
        radial_basis = np.full(3, 1.0 / 3.0)
        grad_r_basis = radius * gradients + np.outer(
            np.array([1.0, 0.0]), radial_basis
        )
        ke = area * (grad_r_basis.T @ grad_r_basis) / (material_mu * radius)
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                K[ni, nj] += ke[i, j]
        if ei not in conductive_regions:
            continue

        radii = xy[:, 0]
        me_r = np.empty((3, 3), dtype=float)
        for i in range(3):
            for j in range(3):
                if i == j:
                    me_r[i, j] = area * (
                        radii[i] / 10.0
                        + (radii.sum() - radii[i]) / 30.0
                    )
                else:
                    k = 3 - i - j
                    me_r[i, j] = area * (
                        (radii[i] + radii[j]) / 30.0
                        + radii[k] / 60.0
                    )
        me = float(conductivity) * me_r
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                M[ni, nj] += me[i, j]
        for ci, name in enumerate(names):
            if ei not in conductor_elements[name]:
                continue
            for node in nodes:
                ring_B[node, ci] += area * float(conductivity) / (6.0 * np.pi)
                ring_D[ci, node] += area / 3.0
            inverse_radius = 0.0
            for weight, barycentric in _TRIANGLE_QUADRATURE:
                radius_q = max(float(np.dot(barycentric, radii)), 1e-12)
                inverse_radius += weight / radius_q
            ring_H[ci, ci] += (
                float(conductivity) * area * inverse_radius / (2.0 * np.pi)
            )
            for node in nodes:
                domainc_B[node, ci] += (
                    float(conductivity) * area * radius / (6.0 * np.pi)
                )
                domainc_D[ci, node] += area / 3.0
            domainc_H[ci, ci] += float(conductivity) * area / (2.0 * np.pi)

    fixed = np.unique(mesh.boundaries.get("outer", []))
    free = np.ones(n, dtype=bool)
    free[fixed] = False
    ring = {
        "K": K.tocsr()[free][:, free],
        "M": M.tocsr()[free][:, free],
        "B": ring_B.tocsr()[free, :],
        "D": ring_D.tocsr()[:, free],
        "H": ring_H,
    }
    domainc = {
        "K": ring["K"].copy(),
        "M": ring["M"].copy(),
        "B": domainc_B.tocsr()[free, :],
        "D": domainc_D.tocsr()[:, free],
        "H": domainc_H,
    }
    relative_difference = {}
    for name in ("K", "M", "B", "D", "H"):
        reference = np.asarray(ring[name].toarray() if hasattr(ring[name], "toarray")
                               else ring[name])
        candidate = np.asarray(domainc[name].toarray() if hasattr(domainc[name], "toarray")
                               else domainc[name])
        denominator = np.linalg.norm(reference)
        relative_difference[name] = float(np.linalg.norm(candidate - reference)
                                          / denominator) if denominator else 0.0
    return {"ring": ring, "domainc": domainc, "relative_difference": relative_difference}


def _conductor_area(mesh: AxisymmetricMesh, indices: np.ndarray) -> float:
    area = 0.0
    for ei in np.asarray(indices, dtype=int):
        nodes = mesh.triangles[ei]
        xy = mesh.points[nodes]
        area += abs(np.linalg.det(np.array([[xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                                            [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]]]))) / 2.0
    if area <= 0.0:
        raise ValueError("conductor has no positive area")
    return area


def _element_geometry(mesh: AxisymmetricMesh) -> list[tuple[np.ndarray, float, np.ndarray, float, np.ndarray]]:
    """Return mesh-only element terms reused by repeated frequency solves."""
    cached = mesh._cache.get("frequency_element_geometry")
    if cached is not None:
        return cached  # type: ignore[return-value]
    geometry = []
    for nodes in mesh.triangles:
        xy = mesh.points[nodes]
        det = np.linalg.det(np.array([
            [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
            [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
        ]))
        area = abs(det) / 2.0
        gradients = np.array([
            [xy[1, 1] - xy[2, 1], xy[2, 1] - xy[0, 1], xy[0, 1] - xy[1, 1]],
            [xy[2, 0] - xy[1, 0], xy[0, 0] - xy[2, 0], xy[1, 0] - xy[0, 0]],
        ]) / det if det else np.zeros((2, 3))
        radius = max(float(xy[:, 0].mean()), 1e-12)
        barycentric = np.full(3, 1.0 / 3.0)
        grad_r_basis = radius * gradients + np.outer(
            np.array([1.0, 0.0]), barycentric
        )
        geometry.append((nodes, area, gradients, radius, grad_r_basis))
    mesh._cache["frequency_element_geometry"] = geometry
    return geometry


def _target_currents(
    mesh: AxisymmetricMesh,
    currents: Mapping[str, complex] | complex,
    names: list[str],
) -> dict[str, complex]:
    """Expand winding/circuit current commands to individual turn currents."""
    signs = mesh.geometry.get("conductor_signs", [1.0] * len(names))
    if not isinstance(currents, Mapping):
        return {
            name: complex(currents) * float(signs[index])
            for index, name in enumerate(names)
        }
    winding_names = mesh.geometry.get("conductor_windings", [])
    circuit_names = mesh.geometry.get("conductor_circuits", [])
    target: dict[str, complex] = {}
    for index, name in enumerate(names):
        value = currents.get(name)
        if value is None and index < len(winding_names):
            value = currents.get(winding_names[index])
        if value is None and index < len(circuit_names):
            value = currents.get(circuit_names[index])
        target[name] = complex(0.0 if value is None else value)
    return target


def _conductor_values(
    mesh: AxisymmetricMesh,
    values: Mapping[str, object] | None,
    names: list[str],
    *,
    default: object = None,
) -> dict[str, object]:
    """Expand conductor, winding, and circuit keyed values to conductors."""
    values = values or {}
    winding_names = mesh.geometry.get("conductor_windings", [])
    circuit_names = mesh.geometry.get("conductor_circuits", [])
    result: dict[str, object] = {}
    for index, name in enumerate(names):
        value = values.get(name)
        if value is None and index < len(winding_names):
            value = values.get(winding_names[index])
        if value is None and index < len(circuit_names):
            value = values.get(circuit_names[index])
        result[name] = default if value is None else value
    return result


def _mode_map(
    mesh: AxisymmetricMesh,
    current_modes: Mapping[str, str] | str,
    names: list[str],
) -> dict[str, str]:
    if isinstance(current_modes, str):
        modes = {name: current_modes for name in names}
    else:
        modes = _conductor_values(mesh, current_modes, names, default="total_current")
    valid = {"total_current", "fixed_current_density"}
    invalid = {name: mode for name, mode in modes.items() if mode not in valid}
    if invalid:
        raise ValueError(
            f"unsupported current mode(s): {invalid}; "
            f"choose from {sorted(valid)}"
        )
    return modes


def solve_frequency_domain(
    mesh: AxisymmetricMesh,
    frequency: float,
    currents: Mapping[str, complex] | complex,
    relative_permeability: Mapping[str, float | complex] | None = None,
    conductivity: float = 58e6,
    phasor_convention: str = "cosine",
    return_voltages: bool = False,
    current_modes: Mapping[str, str] | str = "total_current",
    fixed_current_density: Mapping[str, object] | None = None,
    litz_properties: Mapping[str, LitzProperties] | None = None,
    nonlinear_materials: Mapping[str, NonlinearCoreMaterial] | None = None,
    material_conductivities: Mapping[str, complex] | None = None,
    _assembly_cache: dict[str, object] | None = None,
    timing: dict[str, float] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]] | tuple[
    np.ndarray, dict[str, np.ndarray], dict[str, complex]
]:
    """Solve complex ``A_phi`` and return element current densities by conductor.

    ``currents`` maps constrained conductors to total amperes. ``current_modes``
    selects ``"total_current"`` or ``"fixed_current_density"`` per conductor.
    Fixed-density conductors use ``fixed_current_density`` and do not receive a
    ring-voltage unknown. Conductivity and skin effect are assembled only for
    total-current conductors.
    """
    if frequency < 0 or conductivity < 0:
        raise ValueError("frequency and conductivity must be non-negative")
    if return_voltages and conductivity == 0:
        raise ValueError("conductivity must be positive when returning conductor voltages")
    if phasor_convention not in {"cosine", "sine"}:
        raise ValueError("phasor_convention must be 'cosine' or 'sine'")
    omega = 2.0 * np.pi * float(frequency)
    # For Re{u exp(+j omega t)}, d/dt becomes +j*omega.  The sine
    # convention used by the public API reverses that sign.
    sign = 1j if phasor_convention == "cosine" else -1j
    n = len(mesh.points)
    names = list((mesh.conductor_regions or {}).keys())
    if not names:
        raise ValueError("mesh.conductor_regions is required")
    target = _target_currents(mesh, currents, names)
    modes = _mode_map(mesh, current_modes, names)
    litz_properties_by_conductor = _conductor_values(
        mesh, litz_properties, names
    )
    for name, properties in litz_properties_by_conductor.items():
        if properties is not None and not isinstance(properties, LitzProperties):
            raise TypeError(f"Litz properties for '{name}' must be LitzProperties")
    fixed_density = _conductor_values(mesh, fixed_current_density, names)
    constrained_names = [name for name in names if modes[name] == "total_current"]
    constrained_indices = {name: index for index, name in enumerate(constrained_names)}
    name_indices = {name: index for index, name in enumerate(names)}
    for name in names:
        if modes[name] == "fixed_current_density" and fixed_density[name] is None:
            raise ValueError(f"fixed current density is required for conductor '{name}'")
    # At DC use the same uniform impressed-current problem as the
    # magnetostatic solver.  A ring-voltage model has E_phi proportional to
    # 1/r and therefore does not have the required uniform-current limit.
    if nonlinear_materials:
        from .nonlinear_solver import solve_nonlinear_frequency_domain
        return solve_nonlinear_frequency_domain(
            mesh,
            frequency,
            currents,
            relative_permeability=relative_permeability,
            conductivity=conductivity,
            phasor_convention=phasor_convention,
            return_voltages=return_voltages,
            current_modes=current_modes,
            fixed_current_density=fixed_current_density,
            litz_properties=litz_properties,
            nonlinear_materials=nonlinear_materials,
            material_conductivities=material_conductivities,
        )
    if omega == 0.0:
        areas = {name: _conductor_area(mesh, indices) for name, indices in mesh.conductor_regions.items()}
        source = {
            name: target[name] / areas[name]
            for name in constrained_names
        }
        source.update({
            name: fixed_density[name]
            for name in names
            if modes[name] == "fixed_current_density"
        })
        source["winding"] = 0.0
        values = solve_axisymmetric(mesh, relative_permeability=relative_permeability,
                                    current_density={
                                        key: np.asarray(value).real
                                        if np.asarray(value).ndim
                                        else float(np.real(value))
                                        for key, value in source.items()
                                    })
        densities = {name: np.zeros(len(mesh.triangles), dtype=complex) for name in names}
        for name, indices in mesh.conductor_regions.items():
            indices = np.asarray(indices, dtype=int)
            prescribed = np.asarray(source[name])
            densities[name][indices] = (
                prescribed[indices] if prescribed.shape == (len(mesh.triangles),)
                else prescribed
            )
        if return_voltages:
            voltages = {
                name: target[name] * 2.0 * np.pi / (float(conductivity) * sum(
                    _element_area_over_radius(mesh, element)
                    for element in indices
                ))
                for name, indices in mesh.conductor_regions.items()
                if modes[name] == "total_current"
            }
            return values.astype(complex), densities, voltages
        return values.astype(complex), densities
    mu_r = {"core": 1000.0, "air": 1.0, "winding": 1.0}
    conductivity_by_region = {"core": 0.0, "air": 0.0, "winding": conductivity}
    if material_conductivities:
        conductivity_by_region.update(material_conductivities)
    if mesh.materials:
        mu_r.update({
            name: complex_relative_permeability(
                float(getattr(material, "relative_permeability", 1.0)),
                float(getattr(material, "permeability_loss_angle_deg", 0.0)),
            )
            for name, material in mesh.materials.items()
        })
        conductivity_by_region.update({
            name: complex_conductivity(
                float(getattr(material, "conductivity", 0.0)),
                float(getattr(material, "relative_permittivity", 0.0)),
                float(getattr(material, "permittivity_loss_angle_deg", 0.0)),
                frequency,
            )
            for name, material in mesh.materials.items()
        })
    if relative_permeability:
        mu_r.update(relative_permeability)
    if timing is not None:
        timing.clear()
    assembly_start = time.perf_counter()
    # K is the magnetic curl-curl matrix for physical A_phi.
    # M is the conductive mass matrix for the induced current term.
    # B maps each unknown ring voltage into the magnetic weak-form RHS.
    # D maps nodal A_phi into the cross-sectional current constraint.
    k_rows: list[int] = []
    k_cols: list[int] = []
    k_data: list[complex] = []
    cached_fixed = _assembly_cache.get("fixed_blocks") if _assembly_cache else None
    if cached_fixed is None:
        m_rows: list[int] = []
        m_cols: list[int] = []
        m_data: list[complex] = []
        b_rows: list[int] = []
        b_cols: list[int] = []
        b_data: list[complex] = []
        d_rows: list[int] = []
        d_cols: list[int] = []
        d_data: list[complex] = []
        areas = np.zeros(len(constrained_names), dtype=float)
        inv_r_integrals = np.zeros(len(constrained_names), dtype=float)
        elem_j = np.zeros((len(mesh.triangles), len(names)), dtype=complex)
        fixed_rhs = np.zeros(n, dtype=complex)
    else:
        M, B, D, areas, inv_r_integrals, fixed_rhs = cached_fixed
        elem_j = np.array(_assembly_cache["fixed_elem_j"], copy=True)
    element_conductor = np.full(len(mesh.triangles), -1, dtype=int)
    element_constrained = np.full(len(mesh.triangles), -1, dtype=int)
    element_positions: dict[str, dict[int, int]] = {}
    for name_index, name in enumerate(names):
        indices = np.asarray(mesh.conductor_regions[name], dtype=int)
        element_positions[name] = {int(ei): position for position, ei in enumerate(indices)}
        unassigned = element_conductor[indices] < 0
        element_conductor[indices[unassigned]] = name_index
        if modes[name] == "total_current":
            constrained_index = constrained_indices[name]
            element_constrained[indices[unassigned]] = constrained_index
    for ei, ((nodes, area, gradients, r, grad_r_basis), region) in enumerate(
        zip(_element_geometry(mesh), mesh.regions)
    ):
        if area == 0:
            continue
        xy = mesh.points[nodes]
        rb = np.full(3, 1.0 / 3.0)
        conductor_index = element_conductor[ei]
        conductor_name = names[conductor_index] if conductor_index >= 0 else None
        litz = (
            litz_properties_by_conductor.get(conductor_name)
            if conductor_name else None
        )
        region_mu = mu_r.get(str(region), 1.0)
        if isinstance(region_mu, np.ndarray):
            material_mu = MU0 * complex(region_mu[ei])
        else:
            material_mu = MU0 * complex(region_mu)
        if litz is not None and modes.get(conductor_name) == "total_current":
            material_mu = litz.absolute_permeability
        # Match the working magnetostatic formulation: the cylindrical
        # operator is evaluated through the transformed basis r*A_phi.
        ke = area * (grad_r_basis.T @ grad_r_basis) / (material_mu * r)
        for i, ni in enumerate(nodes):
            for j, nj in enumerate(nodes):
                k_rows.append(int(ni))
                k_cols.append(int(nj))
                k_data.append(complex(ke[i, j]))
        # Conductivity exists only in conductor elements.  In particular, core
        # and air must not contribute an eddy-current mass term.
        if abs(conductivity_by_region.get(str(region), 0.0)) > 0.0 and (
            str(region) == "core"
            or element_constrained[ei] >= 0
        ):
            # Assemble the exact integral of r*N_i*N_j.  Centroid quadrature
            # is exact for r alone, but not for this cubic integrand because
            # r is linear and N_i*N_j is quadratic.
            radii = xy[:, 0]
            me_r = np.empty((3, 3), dtype=float)
            for i in range(3):
                for j in range(3):
                    if i == j:
                        me_r[i, j] = area * (
                            radii[i] / 10.0
                            + (radii.sum() - radii[i]) / 30.0
                        )
                    else:
                        k = 3 - i - j
                        me_r[i, j] = area * (
                            (radii[i] + radii[j]) / 30.0
                            + radii[k] / 60.0
                        )
            element_conductivity = (
                litz.conductivity if litz is not None
                else conductivity_by_region.get(str(region), conductivity)
            )
            me = me_r * element_conductivity
        else:
            me = None
        if cached_fixed is None and me is not None:
            for i, ni in enumerate(nodes):
                for j, nj in enumerate(nodes):
                    m_rows.append(int(ni))
                    m_cols.append(int(nj))
                    m_data.append(complex(me[i, j]))
        if cached_fixed is not None:
            continue
        if conductor_index >= 0:
            name = names[conductor_index]
            if modes[name] == "fixed_current_density":
                prescribed = np.asarray(fixed_density[name])
                indices = np.asarray(mesh.conductor_regions[name], dtype=int)
                if prescribed.ndim == 0:
                    density = complex(prescribed)
                elif prescribed.shape == (len(mesh.triangles),):
                    density = complex(prescribed[ei])
                elif prescribed.shape == (len(indices),):
                    density = complex(prescribed[element_positions[name][ei]])
                else:
                    raise ValueError(
                        f"fixed current density for {name!r} must be scalar "
                        "or element-sized"
                    )
                fixed_rhs[nodes] += area * r * density / 3.0
                elem_j[ei, name_indices[name]] = density
            else:
                ci = constrained_indices[name]
                # Ring voltage source: E_source=V/(2*pi*r).  In the magnetic
                # equation, r*E_source cancels the 1/r.  In the current
                # constraint, the 1/r remains and is integrated separately.
                element_conductivity = (
                    litz.conductivity if litz is not None else conductivity
                )
                b = area * rb * element_conductivity / (2.0 * np.pi)
                areas[ci] += area
                inverse_radius = 0.0
                for weight, barycentric in _TRIANGLE_QUADRATURE:
                    radius_q = max(float(np.dot(barycentric, xy[:, 0])), 1e-12)
                    inverse_radius += weight / radius_q
                inv_r_integrals[ci] += area * inverse_radius
                for i, ni in enumerate(nodes):
                    b_rows.append(int(ni))
                    b_cols.append(ci)
                    b_data.append(complex(b[i]))
                    # The current constraint integrates J over the meridian
                    # cross-section, so the induced A term has no radial
                    # weight: integral(J dS) = H*V - sign*omega*sigma*D*A.
                    d_rows.append(ci)
                    d_cols.append(int(ni))
                    d_data.append(complex(area * rb[i]))
    K = coo_matrix(
        (k_data, (k_rows, k_cols)), shape=(n, n), dtype=complex
    ).tocsr()
    if cached_fixed is None:
        M = coo_matrix((m_data, (m_rows, m_cols)), shape=(n, n), dtype=complex).tocsr()
        B = coo_matrix(
            (b_data, (b_rows, b_cols)),
            shape=(n, len(constrained_names)),
            dtype=complex,
        ).tocsr()
        D = coo_matrix(
            (d_data, (d_rows, d_cols)),
            shape=(len(constrained_names), n),
            dtype=complex,
        ).tocsr()
    if _assembly_cache is not None and cached_fixed is None:
        _assembly_cache["fixed_blocks"] = (
            M,
            B,
            D,
            areas,
            inv_r_integrals,
            fixed_rhs,
        )
        _assembly_cache["fixed_elem_j"] = np.array(elem_j, copy=True)
    if timing is not None:
        timing["matrix assembly"] = time.perf_counter() - assembly_start
    solve_start = time.perf_counter()
    # A_phi is prescribed as the complex Dirichlet value 0+0j on the
    # complete outer core perimeter (top, right, and bottom). The axis is
    # left unconstrained here; regularity is represented by the cylindrical
    # weak form rather than by an additional Dirichlet condition.
    fixed = np.unique(mesh.boundaries.get("outer", []))
    free = np.ones(n, dtype=bool); free[fixed] = False
    # Substituting J=sigma*(V/(2*pi*r)-sign*omega*A) into
    # integral(r*J*v) moves the induced-current term to the left:
    # (K + sign*omega*M) A - B V = 0.
    A = (K + sign * omega * M).tocsr()[free][:, free]
    Br = B[free, :]
    for ci, area in enumerate(areas):
        if area <= 0.0 or inv_r_integrals[ci] <= 0.0:
            raise ValueError(f"conductor '{constrained_names[ci]}' has no elements")
    Br = Br.tocsr()
    Dr = D.tocsr()[:, free]
    # V is the unknown ring voltage.  The prescribed-current equation is
    # I = H*V - sign*omega*sigma*D*A.
    H = lil_matrix((len(constrained_names), len(constrained_names)), dtype=complex)
    constrained_conductivities = np.empty(len(constrained_names), dtype=complex)
    for i in range(len(constrained_names)):
        name = constrained_names[i]
        litz = litz_properties_by_conductor.get(name)
        element_conductivity = litz.conductivity if litz is not None else conductivity
        constrained_conductivities[i] = element_conductivity
        H[i, i] = element_conductivity * inv_r_integrals[i] / (2.0 * np.pi)
    # The two block rows are the magnetic equation and the current
    # constraint.  The right-hand side contains no impressed magnetic source:
    # all excitation enters through the prescribed conductor currents.
    if constrained_names:
        system = bmat(
            [
                [A, -Br],
                [
                    -sign * omega * Dr.multiply(constrained_conductivities[:, None]),
                    H.tocsr(),
                ],
            ],
            format="csr",
        )
        rhs = np.r_[
            fixed_rhs[free],
            [target[name] for name in constrained_names],
        ]
    else:
        system = A
        rhs = fixed_rhs[free]
    solution = spsolve(system, rhs)
    if timing is not None:
        timing["sparse system construction and solve"] = (
            time.perf_counter() - solve_start
        )
    values = np.zeros(n, dtype=complex); values[free] = solution[:free.sum()]
    if not np.isfinite(values).all():
        raise RuntimeError("frequency-domain FEM solve produced non-finite A_phi")
    voltages = solution[free.sum():] if constrained_names else np.zeros(0, dtype=complex)
    reconstruction_start = time.perf_counter()
    # Reconstruct the element-centre current density from the solved A_phi
    # and ring voltage.  This is the same constitutive law used in assembly.
    for ei in np.flatnonzero(element_constrained >= 0):
        nodes = mesh.triangles[ei]
        av = np.mean(values[nodes])
        xy = mesh.points[nodes]
        inverse_radius = 0.0
        for weight, barycentric in _TRIANGLE_QUADRATURE:
            radius_q = max(float(np.dot(barycentric, xy[:, 0])), 1e-12)
            inverse_radius += weight / radius_q
        ci = element_constrained[ei]
        name = constrained_names[ci]
        litz = litz_properties_by_conductor.get(name)
        element_conductivity = litz.conductivity if litz is not None else conductivity
        elem_j[ei, name_indices[name]] = (
            element_conductivity * voltages[ci] * inverse_radius / (2.0 * np.pi)
            - sign * omega * element_conductivity * av
        )
    densities = {name: elem_j[:, i] for i, name in enumerate(names)}
    if timing is not None:
        timing["current-density reconstruction"] = (
            time.perf_counter() - reconstruction_start
        )
    if return_voltages:
        return values, densities, {
            name: voltages[i] for i, name in enumerate(constrained_names)
        }
    return values, densities


def solve_combined(*args, **kwargs):
    """Solve mixed fixed-density and total-current conductor models."""
    return solve_frequency_domain(*args, **kwargs)


def _element_area_over_radius(mesh: AxisymmetricMesh, element: int) -> float:
    """Return the meridian area divided by the element's mean radius."""
    nodes = mesh.triangles[element]
    xy = mesh.points[nodes]
    area = abs(np.linalg.det(np.array([
        [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
        [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
    ]))) / 2.0
    radius = max(float(xy[:, 0].mean()), 1e-12)
    return area / radius


def integrated_current(mesh: AxisymmetricMesh, density: np.ndarray, conductor: str) -> complex:
    """Integrate an element-centre current density over a conductor cross-section."""
    total = 0j
    for ei in mesh.conductor_regions.get(conductor, []):
        nodes = mesh.triangles[ei]; xy = mesh.points[nodes]
        area = abs(np.linalg.det(np.array([[xy[1, 0]-xy[0, 0], xy[2, 0]-xy[0, 0]], [xy[1, 1]-xy[0, 1], xy[2, 1]-xy[0, 1]]]))) / 2
        total += area * density[ei]
    return total


def average_abs_current_density(mesh: AxisymmetricMesh, density: np.ndarray, conductor: str) -> float:
    """Return area-weighted mean ``|J_phi|`` over a conductor cross-section."""
    area_total = 0.0
    weighted = 0.0
    for ei in mesh.conductor_regions.get(conductor, []):
        nodes = mesh.triangles[ei]; xy = mesh.points[nodes]
        area = abs(np.linalg.det(np.array([[xy[1, 0]-xy[0, 0], xy[2, 0]-xy[0, 0]],[xy[1, 1]-xy[0, 1], xy[2, 1]-xy[0, 1]]]))) / 2
        area_total += area
        weighted += area * abs(density[ei])
    if area_total <= 0.0:
        raise ValueError(f"conductor '{conductor}' has no elements")
    return float(weighted / area_total)


def copper_loss(mesh: AxisymmetricMesh, densities: Mapping[str, np.ndarray], conductivity: float = 58e6) -> float:
    """Return time-average copper loss for peak phasors: ``0.5 integral |J|^2/sigma``."""
    return float(sum(0.5 * sum((2*np.pi * abs(np.linalg.det(np.array([[mesh.points[mesh.triangles[e,1],0]-mesh.points[mesh.triangles[e,0],0], mesh.points[mesh.triangles[e,2],0]-mesh.points[mesh.triangles[e,0],0]],[mesh.points[mesh.triangles[e,1],1]-mesh.points[mesh.triangles[e,0],1], mesh.points[mesh.triangles[e,2],1]-mesh.points[mesh.triangles[e,0],1]]]))) / 2) * np.mean(mesh.points[mesh.triangles[e],0]) * abs(j[e])**2 / conductivity for e in mesh.conductor_regions.get(name, [])) for name, j in densities.items()))
