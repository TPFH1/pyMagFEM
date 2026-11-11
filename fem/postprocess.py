"""Post-processing for axisymmetric magnetostatic FEM solutions."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from model import FEMModel

    from .simulation import FrequencyDomainResult

from .litz import LitzLoss, litz_loss
from .nonlinear import NonlinearCoreMaterial
from .physics import complex_conductivity, core_electric_loss_density, core_magnetic_loss_density

MU0 = 4e-7 * np.pi
from .preprocess import AxisymmetricMesh


@dataclass(frozen=True)
class WindingLosses:
    """Per-turn and total loss for one homogenized Litz winding."""

    per_turn: tuple[LitzLoss, ...]

    @property
    def total(self) -> float:
        """Return the total winding loss in watts."""
        return float(sum(loss.total for loss in self.per_turn))

    @property
    def transport(self) -> float:
        """Return the total transport/skin loss in watts."""
        return float(sum(loss.transport for loss in self.per_turn))

    @property
    def proximity(self) -> float:
        """Return the total proximity loss in watts."""
        return float(sum(loss.proximity for loss in self.per_turn))


def core_losses(
    mesh: AxisymmetricMesh,
    magnetic_field: np.ndarray,
    frequency: float,
    *,
    relative_permeability: float = 1000.0,
    loss_angle_deg: float = 0.0,
    a_phi: np.ndarray | None = None,
    conductivity: float = 0.0,
    relative_permittivity: float = 0.0,
    permittivity_loss_angle_deg: float = 0.0,
    nonlinear_material: NonlinearCoreMaterial | None = None,
) -> tuple[float, float, float]:
    """Return magnetic, electrical, and total core loss in watts."""
    from .physics import complex_relative_permeability

    core_indices = np.flatnonzero(np.asarray(mesh.regions) == "core")
    magnetic_values = np.zeros(len(mesh.triangles), dtype=float)
    if nonlinear_material is not None:
        b_values = (
            np.linalg.norm(magnetic_flux_density(mesh, a_phi), axis=1)
            if a_phi is not None
            else np.zeros(len(mesh.triangles))
        )
        local_mu = nonlinear_material.relative_permeability_at(b_values)
        magnetic_values[core_indices] = core_magnetic_loss_density(
            magnetic_field[core_indices], MU0 * local_mu[core_indices], frequency
        )
    else:
        magnetic_values[core_indices] = core_magnetic_loss_density(
            magnetic_field[core_indices],
            MU0 * complex_relative_permeability(relative_permeability, loss_angle_deg),
            frequency,
        )
    electric_values = np.zeros(len(mesh.triangles), dtype=float)
    if a_phi is not None and (nonlinear_material is not None or conductivity > 0.0):
        conductivity_complex = (
            nonlinear_material.conductivity
            if nonlinear_material is not None
            else complex_conductivity(
                conductivity,
                relative_permittivity,
                permittivity_loss_angle_deg,
                frequency,
            )
        )
        element_a_phi = np.mean(a_phi[mesh.triangles[core_indices]], axis=1)
        electric_field = -1j * 2.0 * np.pi * frequency * element_a_phi
        electric_values[core_indices] = core_electric_loss_density(
            electric_field, conductivity_complex
        )
    magnetic_total = 0.0
    electric_total = 0.0
    for element in np.flatnonzero(np.asarray(mesh.regions) == "core"):
        nodes = mesh.triangles[element]
        xy = mesh.points[nodes]
        area = (
            abs(
                np.linalg.det(
                    np.array(
                        [
                            [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                            [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
                        ]
                    )
                )
            )
            / 2.0
        )
        volume_factor = 2.0 * np.pi * float(np.mean(xy[:, 0])) * area
        magnetic_total += volume_factor * magnetic_values[element]
        electric_total += volume_factor * electric_values[element]
    return float(magnetic_total), float(electric_total), float(magnetic_total + electric_total)


def core_loss(*args, **kwargs) -> float:
    """Backward-compatible total core-loss wrapper."""
    return core_losses(*args, **kwargs)[2]


def winding_copper_losses(
    mesh: AxisymmetricMesh,
    current_densities: Mapping[str, np.ndarray],
    *,
    conductivity: float = 58e6,
    conductor_names: list[str] | None = None,
) -> WindingLosses:
    """Return classical solid-conductor loss for each turn and the winding."""
    if conductivity <= 0.0:
        raise ValueError("conductivity must be positive")
    names = conductor_names or [
        f"conductor_{index}" for index in range(len(mesh.conductor_regions or {}))
    ]
    losses = []
    for name in names:
        density = np.asarray(current_densities[name])
        loss = 0.0
        for element in np.asarray(mesh.conductor_regions[name], dtype=int):
            nodes = mesh.triangles[element]
            xy = mesh.points[nodes]
            area = (
                abs(
                    np.linalg.det(
                        np.array(
                            [
                                [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                                [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
                            ]
                        )
                    )
                )
                / 2.0
            )
            radius = float(np.mean(xy[:, 0]))
            loss += 0.5 * 2.0 * np.pi * radius * area * abs(density[element]) ** 2 / conductivity
        losses.append(LitzLoss(float(loss), 0.0))
    return WindingLosses(tuple(losses))


def analytical_dc_winding_resistances(
    component: FEMModel,
) -> dict[str, float]:
    """Estimate terminal DC resistance for each Litz winding in ohms.

    The estimate sums each physical bundle-turn circumference using its strand
    copper area and component material conductivity, then applies ideal equal
    sharing across ``parallel_paths``.
    """
    if not component.windings:
        return {}

    conductors_by_name = {conductor.name: conductor for conductor in component.conductors}
    materials_by_name = {material.name: material for material in component.materials}
    resistances_ohm = {}
    for winding in component.windings:
        terminal_resistance_ohm = 0.0
        for turn in winding.turns:
            conductor_name = turn.conductor or winding.conductor
            conductor = conductors_by_name.get(conductor_name)
            if (
                conductor is None
                or conductor.construction != "litz"
                or conductor.strands is None
                or conductor.strand_diameter is None
            ):
                raise ValueError(
                    f"Winding {winding.name!r} requires defined Litz strand data."
                )
            material = materials_by_name.get(conductor.material)
            if (
                material is None
                or not np.isfinite(material.conductivity)
                or material.conductivity <= 0.0
            ):
                raise ValueError(
                    f"Conductor {conductor.name!r} requires a material with finite, "
                    "positive conductivity."
                )
            turn_radius_m = float(turn.center[0])
            if not np.isfinite(turn_radius_m) or turn_radius_m <= 0.0:
                raise ValueError(f"Winding {winding.name!r} has an invalid turn radius.")
            if conductor.strands < 1 or conductor.strand_diameter <= 0.0:
                raise ValueError(f"Conductor {conductor.name!r} has invalid Litz strand data.")

            strand_area_m2 = np.pi * conductor.strand_diameter**2 / 4.0
            copper_area_m2 = conductor.strands * strand_area_m2
            turn_resistance_ohm = (
                2.0 * np.pi * turn_radius_m
                / (material.conductivity * copper_area_m2)
            )
            current_fraction = (
                turn.current_fraction
                if turn.current_fraction is not None
                else 1.0 / winding.parallel_paths
            )
            terminal_resistance_ohm += turn_resistance_ohm * current_fraction**2
        resistances_ohm[winding.name] = terminal_resistance_ohm
    return resistances_ohm


def winding_litz_losses(
    mesh: AxisymmetricMesh,
    current_densities: Mapping[str, np.ndarray],
    magnetic_field: np.ndarray,
    frequency: float,
    *,
    conductor_names: list[str] | None = None,
    strand_diameter: float,
    fill_factor: float,
    conductivity: float = 58e6,
) -> WindingLosses:
    """Return Meeker Litz losses for each turn and the complete winding.

    ``conductor_names`` determines the turn order. When omitted, all mesh
    conductors are evaluated in mesh order. The returned object exposes
    ``per_turn`` and aggregate ``total``, ``transport``, and ``proximity``
    values in watts.
    """
    names = conductor_names or [
        f"conductor_{index}" for index in range(len(mesh.conductor_regions or {}))
    ]
    losses = tuple(
        litz_loss(
            mesh,
            current_densities,
            magnetic_field,
            name,
            frequency,
            strand_diameter=strand_diameter,
            fill_factor=fill_factor,
            conductivity=conductivity,
        )
        for name in names
    )
    return WindingLosses(losses)


def magnetic_field_intensity(
    mesh: AxisymmetricMesh,
    a_phi: np.ndarray,
    relative_permeability: Mapping[str, float | complex] | None = None,
    nonlinear_material=None,
) -> np.ndarray:
    """Return element-centre magnetic field intensity ``(H_r, H_z)`` in A/m."""
    relative_permeability_by_region = {
        "core": 1000.0,
        "air": 1.0,
        "winding": 1.0,
    }
    if relative_permeability:
        relative_permeability_by_region.update(relative_permeability)
    flux_density = magnetic_flux_density(mesh, a_phi)
    result = np.empty(flux_density.shape, dtype=np.result_type(flux_density, np.complex128))
    for index, region in enumerate(mesh.regions):
        if str(region) == "core" and nonlinear_material is not None:
            mu = nonlinear_material.absolute_permeability_at(
                np.array([np.linalg.norm(flux_density[index])])
            )[0]
        else:
            mu = MU0 * complex(relative_permeability_by_region.get(str(region), 1.0))
        result[index] = flux_density[index] / mu
    return result


def magnetic_flux_density(mesh: AxisymmetricMesh, a_phi: np.ndarray) -> np.ndarray:
    """Return element-centre ``(B_r, B_z)`` values in tesla."""
    a_phi = np.asarray(a_phi)
    if a_phi.shape != (len(mesh.points),):
        raise ValueError("a_phi must contain one value per mesh node")
    result_dtype = np.result_type(a_phi.dtype, np.float64)
    result = np.zeros((len(mesh.triangles), 2), dtype=result_dtype)
    geometry = mesh._cache.get("flux_element_geometry")
    if geometry is None:
        geometry = []
        for nodes in mesh.triangles:
            xy = mesh.points[nodes]
            det = np.linalg.det(
                np.array(
                    [
                        [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                        [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
                    ]
                )
            )
            gradients = (
                np.array(
                    [
                        [xy[1, 1] - xy[2, 1], xy[2, 1] - xy[0, 1], xy[0, 1] - xy[1, 1]],
                        [xy[2, 0] - xy[1, 0], xy[0, 0] - xy[2, 0], xy[1, 0] - xy[0, 0]],
                    ]
                )
                / det
            )
            geometry.append((nodes, gradients, max(float(xy[:, 0].mean()), 1e-12)))
        mesh._cache["flux_element_geometry"] = geometry
    for index, (nodes, gradients, radius) in enumerate(geometry):
        result[index] = (
            -gradients[1] @ a_phi[nodes],
            gradients[0] @ a_phi[nodes] + np.mean(a_phi[nodes]) / radius,
        )
    return result


def magnetic_flux_density_nodal(mesh: AxisymmetricMesh, a_phi: np.ndarray) -> np.ndarray:
    """Return an area-weighted continuous nodal ``(B_r, B_z)`` field.

    The P1 derivative is constant per triangle, so element values are
    inherently discontinuous.  Averaging adjacent element-centre values at
    shared nodes provides the continuous field used for visualization.
    """
    element_field = magnetic_flux_density(mesh, a_phi)
    nodal = np.zeros((len(mesh.points), 2), dtype=element_field.dtype)
    weights = np.zeros(len(mesh.points), dtype=float)
    for element_index, nodes in enumerate(mesh.triangles):
        xy = mesh.points[nodes]
        area = (
            abs(
                np.linalg.det(
                    np.array(
                        [
                            [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                            [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
                        ]
                    )
                )
            )
            / 2.0
        )
        nodal[nodes] += area * element_field[element_index]
        weights[nodes] += area
    if np.any(weights == 0.0):
        raise ValueError("mesh contains an isolated node")
    return nodal / weights[:, None]


def magnetic_energy(
    mesh: AxisymmetricMesh, a_phi: np.ndarray, current_density: float | Mapping[str, float]
) -> float:
    """Return stored magnetic energy in joules."""
    if np.iscomplexobj(a_phi):
        raise ValueError("magnetic_energy requires a real magnetostatic solution")
    currents = _currents(current_density)
    energy = 0.0
    for nodes, region in zip(mesh.triangles, mesh.regions):
        xy = mesh.points[nodes]
        area = (
            abs(
                np.linalg.det(
                    np.array(
                        [
                            [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                            [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
                        ]
                    )
                )
            )
            / 2
        )
        energy += np.pi * area * np.mean(xy[:, 0]) * currents[str(region)] * np.mean(a_phi[nodes])
    return float(energy)


def _currents(current_density: float | Mapping[str, float]) -> dict[str, float]:
    currents = {"core": 0.0, "air": 0.0, "winding": 0.0}
    if isinstance(current_density, Mapping):
        currents.update(current_density)
    else:
        currents["winding"] = float(current_density)
    return currents


def inductance(
    mesh: AxisymmetricMesh,
    a_phi: np.ndarray,
    current: float,
    current_density: float | Mapping[str, float],
) -> float:
    """Return energy-based inductance in henries."""
    if current == 0:
        raise ValueError("current must be non-zero")
    return 2.0 * magnetic_energy(mesh, a_phi, current_density) / current**2


def winding_flux_linkage(
    mesh: AxisymmetricMesh, a_phi: np.ndarray, current: float, turns: int | None = None
) -> float | complex:
    """Estimate total winding flux linkage by averaging flux over each conductor."""
    if current == 0:
        raise ValueError("current must be non-zero")
    values = []
    signs = mesh.geometry.get("conductor_signs", [1.0] * len(mesh.conductor_regions or {}))
    fractions = mesh.geometry.get("conductor_current_fractions", [1.0] * len(signs))
    for index, item in enumerate(mesh.geometry["p_cond_center_1"]):
        center = np.asarray(item, dtype=float)
        elements = np.asarray(
            (mesh.conductor_regions or {}).get(f"conductor_{index}", []), dtype=int
        )
        if len(elements) == 0:
            distances = np.linalg.norm(mesh.points - center[:2], axis=1)
            local = np.array([np.argmin(distances)])
            values.append(float(signs[index]) * float(fractions[index])
                          * 2.0 * np.pi * center[0] * a_phi[local[0]])
            continue
        weighted_flux = 0.0
        area_total = 0.0
        for element in elements:
            nodes = mesh.triangles[element]
            xy = mesh.points[nodes]
            area = (
                abs(
                    np.linalg.det(
                        np.array(
                            [
                                [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                                [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
                            ]
                        )
                    )
                )
                / 2.0
            )
            weighted_flux += area * 2.0 * np.pi * np.mean(xy[:, 0]) * np.mean(a_phi[nodes])
            area_total += area
        values.append(float(signs[index]) * float(fractions[index]) * weighted_flux / area_total)
    linkage = np.sum(values)
    return linkage if turns is None else linkage * float(turns) / max(len(values), 1)


def winding_flux_linkages(
    mesh: AxisymmetricMesh,
    a_phi: np.ndarray,
) -> dict[str, complex]:
    """Return signed terminal flux linkage, averaging equal-current parallel paths."""
    names = mesh.geometry.get("conductor_windings", [])
    centers = mesh.geometry.get("p_cond_center_1", [])
    signs = mesh.geometry.get("conductor_signs", [1.0] * len(centers))
    fractions = mesh.geometry.get("conductor_current_fractions", [1.0] * len(centers))
    result: dict[str, complex] = {}
    for index, center in enumerate(centers):
        winding = names[index] if index < len(names) else "winding"
        elements = np.asarray(
            (mesh.conductor_regions or {}).get(f"conductor_{index}", []),
            dtype=int,
        )
        if len(elements) == 0:
            local = np.argmin(np.linalg.norm(mesh.points - np.asarray(center), axis=1))
            flux = 2.0 * np.pi * float(center[0]) * a_phi[local]
        else:
            weighted_flux = 0.0j
            area_total = 0.0
            for element in elements:
                nodes = mesh.triangles[element]
                xy = mesh.points[nodes]
                area = (
                    abs(
                        np.linalg.det(
                            np.array(
                                [
                                    [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                                    [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
                                ]
                            )
                        )
                    )
                    / 2.0
                )
                weighted_flux += area * 2.0 * np.pi * np.mean(xy[:, 0]) * np.mean(a_phi[nodes])
                area_total += area
            flux = weighted_flux / area_total
        result[winding] = (result.get(winding, 0.0j)
                   + float(signs[index]) * float(fractions[index]) * flux)
    return result


def inductance_matrix(
    model: FEMModel,
    *,
    mesh: AxisymmetricMesh | None = None,
    winding_order: list[str] | None = None,
    test_current: float = 1.0,
    return_results: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, FrequencyDomainResult]]:
    """Solve open-circuit basis excitations and return complex inductances in H.

    Rows are linked windings and columns are excited windings, both in
    ``winding_order`` (model winding order by default). Each column is the
    flux-linkage vector divided by ``test_current`` with every other circuit
    held at zero net current. Materials, frequency, and phasor convention come
    from the model. The same mesh is reused and the input model is not mutated.

    Windings must have distinct circuits. This is a linear frequency-domain
    identification, not an incremental nonlinear inductance calculation.
    With ``return_results=True``, also return solved cases keyed by driven
    winding name. No reciprocity averaging is applied.
    """
    from .simulation import FrequencyDomainSimulation

    if not np.isfinite(test_current) or test_current == 0:
        raise ValueError("test_current must be finite and non-zero")
    windings = {winding.name: winding for winding in model.windings}
    if not windings or len(windings) != len(model.windings):
        raise ValueError("model must contain uniquely named windings")
    order = list(windings) if winding_order is None else list(winding_order)
    if len(order) != len(windings) or set(order) != set(windings):
        raise ValueError("winding_order must contain each model winding exactly once")
    circuits = [windings[name].circuit for name in order]
    if any(not circuit for circuit in circuits) or len(set(circuits)) != len(circuits):
        raise ValueError("each winding must have a distinct, non-empty circuit")
    if mesh is None:
        mesh = FrequencyDomainSimulation(model).create_mesh()
    matrix = np.empty((len(order), len(order)), dtype=complex)
    results = {}
    for column, driven_name in enumerate(order):
        test_model = deepcopy(model)
        test_model.excitation.circuits = {
            circuit: test_current if circuit == windings[driven_name].circuit else 0.0
            for circuit in circuits
        }
        result = FrequencyDomainSimulation(test_model, mesh=mesh).solve()
        linkages = result.winding_flux_linkages()
        matrix[:, column] = [linkages[name] / test_current for name in order]
        results[driven_name] = result
    if not np.isfinite(matrix).all():
        raise ValueError("inductance matrix contains non-finite values")
    return (matrix, results) if return_results else matrix


@dataclass(frozen=True)
class TransformerEquivalent:
    """Inductive two-winding T-equivalent with primary-side magnetizing branch."""

    turns_ratio: float
    coupling_coefficient: float
    magnetizing_primary_h: float
    leakage_primary_h: float
    leakage_secondary_h: float
    secondary_series_inductance_h: float
    reciprocity_error_h: float
    equivalent_type: Literal[
        "physical_turns_ratio",
        "balanced_leakage",
        "primary_concentrated_leakage",
        "secondary_concentrated_leakage",
    ] = "balanced_leakage"

    @property
    def leakage_secondary_referred_primary_h(self) -> float:
        return self.turns_ratio**2 * self.leakage_secondary_h

    @property
    def transformer_leakage_referred_primary_h(self) -> float:
        return self.leakage_primary_h + self.leakage_secondary_referred_primary_h

    @property
    def external_inductance_referred_primary_h(self) -> float:
        return self.turns_ratio**2 * self.secondary_series_inductance_h

    @property
    def total_leakage_referred_primary_h(self) -> float:
        return (self.transformer_leakage_referred_primary_h
                + self.external_inductance_referred_primary_h)

    def to_dict(self) -> dict[str, float | str]:
        return {
            **asdict(self),
            "leakage_secondary_referred_primary_h": self.leakage_secondary_referred_primary_h,
            "transformer_leakage_referred_primary_h": self.transformer_leakage_referred_primary_h,
            "external_inductance_referred_primary_h": self.external_inductance_referred_primary_h,
            "total_leakage_referred_primary_h": self.total_leakage_referred_primary_h,
        }

    def print(self) -> None:
        print(f"Equivalent type: {self.equivalent_type}")
        print(f"Ideal turns ratio: {self.turns_ratio:.6f}:1")
        print(f"Coupling coefficient: {self.coupling_coefficient:.8f}")
        for label, value in (
            ("Primary magnetizing inductance", self.magnetizing_primary_h),
            ("Primary series leakage", self.leakage_primary_h),
            ("Secondary series leakage", self.leakage_secondary_h),
            ("Transformer-only leakage referred to primary", self.transformer_leakage_referred_primary_h),
            ("External secondary series inductance", self.secondary_series_inductance_h),
            ("External inductance referred to primary", self.external_inductance_referred_primary_h),
            ("Total series leakage referred to primary", self.total_leakage_referred_primary_h),
        ):
            print(f"{label}: {value * 1e6:.6f} uH")
        if abs(self.coupling_coefficient) > 1:
            print("Warning: |k| > 1; check matrix passivity and mesh convergence.")


def transformer_equivalent(
    matrix_h: np.ndarray,
    *,
    secondary_series_inductance_h: float = 0.0,
    equivalent_type: Literal[
        "physical_turns_ratio",
        "balanced_leakage",
        "primary_concentrated_leakage",
        "secondary_concentrated_leakage",
    ] = "balanced_leakage",
    physical_turns_ratio: float | None = None,
) -> TransformerEquivalent:
    """Extract an inductive T-model from a primary/secondary 2x2 matrix in H.

    Uses real self-inductances and the mean real mutual term. ``equivalent_type``
    selects the referral ratio or the side on which leakage is concentrated.
    The default, ``balanced_leakage``, uses n=sqrt(L11/L22). The
    ``physical_turns_ratio`` type requires the positive physical ratio as an
    additional argument. The two concentrated-leakage types choose n so that
    the secondary or primary leakage, respectively, is zero. Ratios are signed
    when needed to preserve mutual-inductance polarity. The input matrix is not
    changed; losses remain in the original complex matrix. Optional external
    secondary series inductance is referred by the selected ratio squared.
    """
    matrix = np.asarray(matrix_h, dtype=complex)
    if matrix.shape != (2, 2) or not np.isfinite(matrix).all():
        raise ValueError("matrix_h must be a finite 2x2 inductance matrix")
    external = float(secondary_series_inductance_h)
    if not np.isfinite(external) or external < 0:
        raise ValueError("secondary_series_inductance_h must be finite and non-negative")
    l11, l22 = float(matrix[0, 0].real), float(matrix[1, 1].real)
    if l11 <= 0 or l22 <= 0:
        raise ValueError("self-inductances must be positive")
    mutual = float((matrix[0, 1].real + matrix[1, 0].real) / 2.0)
    equivalent_types = {
        "physical_turns_ratio",
        "balanced_leakage",
        "primary_concentrated_leakage",
        "secondary_concentrated_leakage",
    }
    if equivalent_type not in equivalent_types:
        raise ValueError(f"equivalent_type must be one of {sorted(equivalent_types)}")
    if equivalent_type == "physical_turns_ratio":
        if physical_turns_ratio is None:
            raise ValueError("physical_turns_ratio is required for this equivalent_type")
        ratio = float(physical_turns_ratio)
        if not np.isfinite(ratio) or ratio <= 0.0:
            raise ValueError("physical_turns_ratio must be finite and positive")
    else:
        if physical_turns_ratio is not None:
            raise ValueError("physical_turns_ratio is only valid for equivalent_type='physical_turns_ratio'")
        if equivalent_type == "balanced_leakage":
            ratio = float(np.sqrt(l11 / l22))
        elif mutual == 0.0:
            raise ValueError("concentrated-leakage equivalents require non-zero mutual inductance")
        elif equivalent_type == "primary_concentrated_leakage":
            ratio = mutual / l22
        else:
            ratio = l11 / mutual
    magnetizing = ratio * mutual
    return TransformerEquivalent(
        turns_ratio=ratio,
        coupling_coefficient=mutual / float(np.sqrt(l11 * l22)),
        magnetizing_primary_h=magnetizing,
        leakage_primary_h=l11 - magnetizing,
        leakage_secondary_h=l22 - mutual / ratio,
        secondary_series_inductance_h=external,
        reciprocity_error_h=float(abs(matrix[0, 1] - matrix[1, 0])),
        equivalent_type=equivalent_type,
    )


def winding_voltages(
    mesh: AxisymmetricMesh,
    a_phi: np.ndarray,
    densities: Mapping[str, np.ndarray],
    current: complex,
    frequency: float,
    conductivity: float = 58e6,
    phasor_convention: str = "cosine",
    conductor_conductivities: Mapping[str, complex] | None = None,
) -> dict[str, complex]:
    """Return the per-conductor ring-source voltages.

    The source voltage is reconstructed from the constitutive relation used by
    the frequency-domain solver. It already contains both the resistive and
    induced contributions; adding an induced-voltage estimate to a loss-based
    resistance drop would count the field contribution twice. Optional
    per-conductor conductivities support homogenized conductor models.
    """
    if current == 0:
        raise ValueError("current must be non-zero")
    if frequency < 0 or conductivity <= 0:
        raise ValueError("frequency must be non-negative and conductivity must be positive")
    if phasor_convention not in {"cosine", "sine"}:
        raise ValueError("phasor_convention must be 'cosine' or 'sine'")
    signs = mesh.geometry.get("conductor_signs", [1.0] * len(mesh.conductor_regions or {}))
    result: dict[str, complex] = {}
    omega_sign = 1j if phasor_convention == "cosine" else -1j
    from .frequency_solver import _TRIANGLE_QUADRATURE

    for index, center in enumerate(mesh.geometry["p_cond_center_1"]):
        name = f"conductor_{index}"
        conductor_conductivity = complex(
            (conductor_conductivities or {}).get(name, conductivity)
        )
        if not np.isfinite(conductor_conductivity) or conductor_conductivity == 0.0:
            raise ValueError(f"conductivity for {name} must be finite and non-zero")
        elements = np.asarray((mesh.conductor_regions or {}).get(name, []), dtype=int)
        density = densities.get(name)
        if density is None or len(elements) == 0:
            raise ValueError(f"missing current density for {name}")
        area_total = 0.0
        voltage = 0.0j
        center = np.asarray(center, dtype=float)
        for element in elements:
            nodes = mesh.triangles[element]
            xy = mesh.points[nodes]
            area = (
                abs(
                    np.linalg.det(
                        np.array(
                            [
                                [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                                [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
                            ]
                        )
                    )
                )
                / 2.0
            )
            inverse_radius = sum(
                weight / max(float(np.dot(barycentric, xy[:, 0])), 1e-12)
                for weight, barycentric in _TRIANGLE_QUADRATURE
            )
            voltage += area * (
                2.0
                * np.pi
                / inverse_radius
                * (
                    density[element] / conductor_conductivity
                    + omega_sign * 2.0 * np.pi * float(frequency) * np.mean(a_phi[nodes])
                )
            )
            area_total += area
        result[name] = float(signs[index]) * voltage / area_total
    return result


def winding_terminal_voltages(
    mesh: AxisymmetricMesh,
    conductor_voltages: Mapping[str, complex],
) -> dict[str, complex]:
    """Sum polarity-aligned turn voltages, averaging ideal parallel paths."""
    centers = mesh.geometry.get("p_cond_center_1", [])
    windings = mesh.geometry.get("conductor_windings", [])
    fractions = mesh.geometry.get("conductor_current_fractions", [1.0] * len(centers))
    result: dict[str, complex] = {}
    for index in range(len(centers)):
        conductor_name = f"conductor_{index}"
        if conductor_name not in conductor_voltages:
            raise ValueError(f"missing voltage for {conductor_name}")
        winding = windings[index] if index < len(windings) else "winding"
        fraction = float(fractions[index]) if index < len(fractions) else 1.0
        result[winding] = result.get(winding, 0.0j) + (
            complex(conductor_voltages[conductor_name]) * fraction
        )
    return result


def winding_complex_powers(
    terminal_voltages: Mapping[str, complex],
    terminal_currents: Mapping[str, complex],
) -> dict[str, complex]:
    """Return winding complex powers in VA from peak voltage/current phasors."""
    return {
        name: 0.5 * complex(voltage) * complex(terminal_currents.get(name, 0.0)).conjugate()
        for name, voltage in terminal_voltages.items()
    }
