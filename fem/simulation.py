"""High-level frequency-domain simulation API."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal, cast

import numpy as np

from model import FEMModel

from .constitutive import conductor_properties, core_properties, effective_conductivities
from .frequency_solver import solve_frequency_domain
from .litz import LitzLoss, LitzProperties
from .losses import LossDensities, element_volumes, loss_densities
from .materials import MU0
from .nonlinear import NonlinearCoreMaterial
from .physics import complex_relative_permeability
from .postprocess import (
    WindingLosses,
    core_losses,
    magnetic_field_intensity,
    magnetic_flux_density,
    winding_complex_powers,
    winding_flux_linkages,
    winding_terminal_voltages,
    winding_voltages,
)
from .preprocess import AxisymmetricMesh, build_mesh


def _material_map(model: FEMModel) -> dict[str, Any]:
    return {material.name: material for material in model.materials}


def _material_value(
    model: FEMModel,
    name: str,
    attribute: str,
    default: float,
) -> float:
    material = _material_map(model).get(name)
    return float(getattr(material, attribute, default)) if material is not None else default


def _relative_permeability(
    model: FEMModel,
    material_name: str,
    default: float,
) -> float:
    """Resolve relative permeability from the material catalogue."""
    material = _material_map(model).get(material_name)
    return (
        float(material.relative_permeability)
        if material is not None
        else default
    )


def _permeability_map(model: FEMModel) -> dict[str, complex]:
    materials = _material_map(model)
    core_name = (
        model.geometry.core.material
        if model.geometry.core is not None
        else "core"
    )
    core_material = materials.get(core_name)
    winding_material = (
        model.conductors[0].material
        if model.conductors
        else "winding"
    )
    return {
        "core": complex_relative_permeability(
            _relative_permeability(model, core_name, 1000.0),
            float(getattr(core_material, "permeability_loss_angle_deg", 0.0)),
        ),
        "air": complex_relative_permeability(
            _relative_permeability(model, "air", 1.0), 0.0
        ),
        "winding": complex_relative_permeability(
            _relative_permeability(model, winding_material, 1.0), 0.0
        ),
    }


def _conductivity(model: FEMModel) -> float:
    return _material_value(
        model,
        model.conductors[0].material if model.conductors else "copper",
        "conductivity",
        58e6,
    )


@dataclass
class FrequencyDomainResult:
    """Solved peak-phasor fields with constitutive data for post-processing.

    Per-conductor conductivity and effective Litz properties are retained on
    the result so losses and voltages use the solve's material values rather
    than a single winding-wide copper conductivity. Linear core conductivity
    includes the dielectric contribution at ``frequency_hz``.
    """

    model: FEMModel
    mesh: AxisymmetricMesh
    a_phi: np.ndarray
    current_densities: dict[str, np.ndarray]
    turn_voltages: dict[str, complex]
    frequency_hz: float
    conductivity: float
    relative_permeability: dict[str, complex]
    nonlinear_material: Any | None = None
    nonlinear_materials: dict[str, NonlinearCoreMaterial] = field(default_factory=dict)
    litz_properties: dict[str, LitzProperties] = field(default_factory=dict)
    conductor_conductivities: dict[str, complex] = field(default_factory=dict)
    core_conductivity: complex | None = None

    def __post_init__(self) -> None:
        """Fill model-derived property maps without replacing supplied solve data."""
        conductivities, properties = conductor_properties(self.model, self.frequency_hz)
        self.conductor_conductivities = {**conductivities, **self.conductor_conductivities}
        self.litz_properties = {**properties, **self.litz_properties}
        if self.core_conductivity is None:
            core_name = self.model.geometry.core.material if self.model.geometry.core else "core"
            material = _material_map(self.model).get(
                core_name, (self.mesh.materials or {}).get("core"),
            )
            _, self.core_conductivity = core_properties(material, self.frequency_hz)

    @classmethod
    def from_solution(
        cls,
        model: FEMModel,
        mesh: AxisymmetricMesh,
        a_phi: np.ndarray,
        current_densities: dict[str, np.ndarray],
        *,
        frequency_hz: float | None = None,
        turn_voltages: dict[str, complex] | None = None,
        nonlinear_material: Any | None = None,
        nonlinear_materials: Mapping[str, NonlinearCoreMaterial] | None = None,
        litz_properties: Mapping[str, LitzProperties] | None = None,
        conductor_conductivities: Mapping[str, complex] | None = None,
    ) -> FrequencyDomainResult:
        """Wrap an externally solved field without solving or changing its arrays.

        ``frequency_hz`` defaults to the model excitation. Missing conductor
        properties are derived at that frequency; supplied ``litz_properties``
        and ``conductor_conductivities`` override matching derived entries.
        Pass the effective properties used by the external solve if they differ
        from the model. Effective Litz conductivity takes precedence over a
        matching conductivity-map entry during post-processing.
        """
        return cls(
            model=model,
            mesh=mesh,
            a_phi=a_phi,
            current_densities=current_densities,
            turn_voltages=turn_voltages or {},
            frequency_hz=float(
                model.excitation.frequency if frequency_hz is None else frequency_hz
            ),
            conductivity=_conductivity(model),
            relative_permeability=_permeability_map(model),
            nonlinear_material=nonlinear_material,
            nonlinear_materials=dict(nonlinear_materials or {}),
            litz_properties=dict(litz_properties or {}),
            conductor_conductivities=dict(conductor_conductivities or {}),
        )

    def _core_nonlinear_material(self) -> NonlinearCoreMaterial | None:
        return self.nonlinear_materials.get("core", self.nonlinear_material)

    def magnetic_flux_density(self) -> np.ndarray:
        return magnetic_flux_density(self.mesh, self.a_phi)

    def magnetic_field_intensity(self) -> np.ndarray:
        field_intensity = magnetic_field_intensity(
            self.mesh,
            self.a_phi,
            relative_permeability=self.relative_permeability,
            nonlinear_material=self._core_nonlinear_material(),
        )
        if self.litz_properties:
            flux_density = self.magnetic_flux_density()
            for name, properties in self.litz_properties.items():
                elements = self.mesh.conductor_regions[name]
                field_intensity[elements] = flux_density[elements] / properties.absolute_permeability
        return field_intensity

    def winding_copper_losses(self) -> WindingLosses:
        """Return solid losses; use winding_losses() for a result containing Litz bundles."""
        if self.litz_properties:
            raise ValueError("winding_copper_losses is solid-only; use winding_losses for Litz")
        return WindingLosses(tuple(self._conductor_losses().values()))

    def loss_densities(self) -> LossDensities:
        """Return full-mesh electrical, magnetic, and total densities in W/m^3.

        Uses the retained conductor properties and local nonlinear core
        permeability when present. Arrays follow mesh triangle order and are
        identical to those used by loss plots and axisymmetric loss integration.
        Lossless air and solid-conductor magnetic components are zero.
        """
        return loss_densities(
            self.mesh, self.frequency_hz, a_phi=self.a_phi,
            current_densities=self.current_densities, magnetic_field=self.magnetic_field_intensity(),
            conductivity=self.conductivity, conductor_conductivities=self.conductor_conductivities,
            litz_properties=self.litz_properties, nonlinear_material=self._core_nonlinear_material(),
            core_permeability=MU0 * self.relative_permeability["core"],
            core_conductivity=self.core_conductivity,
        )

    def _conductor_losses(self) -> dict[str, LitzLoss]:
        """Integrate transport and proximity watts for each physical conductor."""
        densities = loss_densities(
            self.mesh, self.frequency_hz, current_densities=self.current_densities,
            magnetic_field=self.magnetic_field_intensity(), conductivity=self.conductivity,
            conductor_conductivities=self.conductor_conductivities,
            litz_properties=self.litz_properties, include_core=False,
        )
        volumes = element_volumes(self.mesh)
        return {
            name: LitzLoss(
                float(np.dot(volumes[elements], densities.electrical[elements])),
                float(np.dot(volumes[elements], densities.magnetic[elements])),
            )
            for name, indices in (self.mesh.conductor_regions or {}).items()
            for elements in [np.asarray(indices, dtype=int)]
        }

    def core_losses(self) -> tuple[float, float, float]:
        """Return integrated magnetic, electrical, and total core losses in watts.

        Retained complex linear properties already contain their loss angles.
        A nonlinear material instead supplies local permeability and core
        conductivity, using the same evaluator as the density plots.
        """
        core_material = _material_map(self.model).get(
            self.model.geometry.core.material
            if self.model.geometry.core is not None
            else "core"
        )
        return core_losses(
            self.mesh,
            self.magnetic_field_intensity(),
            self.frequency_hz,
            relative_permeability=(
                float(abs(self.relative_permeability["core"]))
            ),
            loss_angle_deg=float(
                getattr(core_material, "permeability_loss_angle_deg", 0.0)
            ),
            a_phi=self.a_phi,
            conductivity=float(getattr(core_material, "conductivity", 0.0)),
            relative_permittivity=float(
                getattr(core_material, "relative_permittivity", 0.0)
            ),
            permittivity_loss_angle_deg=float(
                getattr(core_material, "permittivity_loss_angle_deg", 0.0)
            ),
            nonlinear_material=self._core_nonlinear_material(),
            core_permeability=MU0 * self.relative_permeability["core"],
            core_conductivity=self.core_conductivity,
        )

    def winding_losses(self) -> dict[str, WindingLosses]:
        """Return physical bundle losses grouped by winding, including Litz effects.

        Parallel paths are already included in the solved bundle currents.
        Sum their losses; do not divide the resulting watts by the path count.
        """
        conductor_losses = self._conductor_losses()
        losses = {}
        conductor_index = 0
        for winding in self.model.windings:
            per_turn = []
            for turn in winding.turns:
                name = f"conductor_{conductor_index}"
                per_turn.append(conductor_losses[name])
                conductor_index += 1
            losses[winding.name] = WindingLosses(tuple(per_turn))
        return losses

    def loss_summary(self) -> dict[str, float]:
        """Return winding transport/proximity and core losses in watts."""
        windings = self.winding_losses()
        summary = {}
        for name, losses in windings.items():
            summary[f"{name}_copper"] = losses.total
            summary[f"{name}_transport"] = losses.transport
            summary[f"{name}_proximity"] = losses.proximity
        magnetic, electrical, core_total = self.core_losses()
        summary.update(core_magnetic=magnetic, core_electrical=electrical)
        summary["total"] = sum(losses.total for losses in windings.values()) + core_total
        return summary

    def winding_flux_linkages(self) -> dict[str, complex]:
        return winding_flux_linkages(self.mesh, self.a_phi)

    def winding_voltages(self) -> dict[str, complex]:
        return winding_voltages(
            self.mesh,
            self.a_phi,
            self.current_densities,
            self._representative_current(),
            self.frequency_hz,
            conductivity=self.conductivity,
            phasor_convention=self.model.excitation.phasor_convention,
            conductor_conductivities=effective_conductivities(
                list(self.mesh.conductor_regions or {}), self.conductivity,
                self.conductor_conductivities, self.litz_properties,
            ),
        )

    def winding_terminal_voltages(self) -> dict[str, complex]:
        """Return terminal voltage phasors by winding, in the model references."""
        return winding_terminal_voltages(self.mesh, self.winding_voltages())

    def winding_complex_powers(self) -> dict[str, complex]:
        """Return absorbed complex winding powers in VA for peak phasors."""
        currents = self._currents()
        winding_currents = {
            winding.name: complex(
                currents.get(winding.circuit, currents.get(winding.name, 0.0))
            )
            for winding in self.model.windings
        }
        return winding_complex_powers(self.winding_terminal_voltages(), winding_currents)

    def plot_physical_field(
        self,
        field: Any,
        *,
        color_scale: Literal["linear", "log"] = "linear",
        **kwargs: Any,
    ) -> Any:
        """Plot a solved field, filling in mesh and solved-field arguments.

        ``color_scale="linear"`` is the default. Use ``color_scale="log"``
        for logarithmic field colours and colourbar; non-positive values are
        masked and at least one positive value is required. Spatial axes
        remain linear. Other options are forwarded to
        :func:`fem.plot.plot_physical_field`.
        """
        from .plot import plot_physical_field

        kwargs.setdefault("a_phi", self.a_phi)
        kwargs.setdefault("current_densities", self.current_densities)
        kwargs.setdefault("magnetic_field", self.magnetic_field_intensity())
        kwargs.setdefault("litz_properties", self.litz_properties)
        kwargs.setdefault("nonlinear_material", self._core_nonlinear_material())
        kwargs.setdefault("frequency", self.frequency_hz)
        kwargs.setdefault("conductor_conductivities", self.conductor_conductivities)
        kwargs.setdefault("core_permeability", MU0 * self.relative_permeability["core"])
        kwargs.setdefault("core_conductivity", self.core_conductivity)
        return plot_physical_field(self.mesh, field, color_scale=color_scale, **kwargs)

    def report(self):
        """Build the standard report using this result's model context."""
        from .report import build_result_report

        currents = self._currents()
        linkages = self.winding_flux_linkages()
        inductances = {
            name: linkage / complex(currents.get(name, 1.0))
            for name, linkage in linkages.items()
            if complex(currents.get(name, 1.0)) != 0
        }
        solid_losses = {}
        litz_losses = {}
        winding_losses = self.winding_losses()
        definitions = {conductor.name: conductor for conductor in self.model.conductors}
        for winding in self.model.windings:
            definition = definitions.get(winding.conductor)
            target = (litz_losses if definition is not None
                      and definition.construction == "litz" else solid_losses)
            if winding.turns:
                target[winding.name] = winding_losses[winding.name]
        if not self.model.windings:
            winding_name = next(iter(linkages), "winding")
            target = litz_losses if self.litz_properties else solid_losses
            target[winding_name] = WindingLosses(tuple(self._conductor_losses().values()))
        return build_result_report(
            self.model,
            self.mesh,
            frequency_hz=self.frequency_hz,
            currents_a=currents,
            current_densities=self.current_densities,
            flux_density=self.magnetic_flux_density(),
            inductance_h=inductances,
            solid_losses=solid_losses,
            litz_losses=litz_losses,
            core_losses={"core": self.core_losses()},
            turn_voltages=self.turn_voltages,
            terminal_voltages=self.winding_voltages(),
        )

    def _current_for_solver(self) -> complex | dict[str, complex]:
        currents = self._currents()
        if len(currents) == 1:
            return complex(next(iter(currents.values())))
        return {name: complex(value) for name, value in currents.items()}

    def _currents(self) -> dict[str, float]:
        if self.model.excitation.circuits:
            return dict(self.model.excitation.circuits)
        return {
            circuit.name: circuit.current
            for circuit in self.model.circuits
        }

    def _representative_current(self) -> complex:
        currents = self._currents()
        if not currents:
            raise ValueError("at least one excitation current is required")
        current = next(iter(currents.values()))
        if current == 0:
            raise ValueError("excitation current must be non-zero")
        return complex(current)

def solve_operating_point(
    model: FEMModel,
    *,
    frequency_hz: float,
    currents: Mapping[str, complex],
    mesh: AxisymmetricMesh | None = None,
    nonlinear_materials: Mapping[str, NonlinearCoreMaterial] | None = None,
    nonlinear_tolerance: float | None = None,
) -> FrequencyDomainResult:
    """Solve terminal-current phasors on an independent model copy.

    Phasor amplitudes follow the model convention (peak amplitudes for loss
    calculations). Missing circuits are open; unknown circuits are rejected.
    Geometry and materials are unchanged, allowing reuse of an existing mesh.
    ``nonlinear_tolerance`` overrides the relative update threshold when
    nonlinear materials are supplied.
    """
    if not np.isfinite(frequency_hz) or frequency_hz < 0:
        raise ValueError("frequency_hz must be finite and non-negative")
    circuits = {circuit.name for circuit in model.circuits}
    circuits.update(winding.circuit for winding in model.windings)
    if set(currents) - circuits:
        raise ValueError("currents reference unknown circuits")
    if nonlinear_tolerance is not None and not nonlinear_materials:
        raise ValueError("nonlinear_tolerance requires nonlinear_materials")
    if nonlinear_tolerance is not None and (
        not np.isfinite(nonlinear_tolerance) or nonlinear_tolerance <= 0.0
    ):
        raise ValueError("nonlinear_tolerance must be finite and positive")
    values = {name: complex(currents.get(name, 0.0)) for name in circuits}
    if not values or not all(np.isfinite(value) for value in values.values()):
        raise ValueError("currents must contain finite phasors for a model with circuits")
    component = deepcopy(model)
    component.excitation.frequency = frequency_hz
    component.excitation.circuits = values
    simulation = FrequencyDomainSimulation(component, mesh=mesh)
    if nonlinear_materials is None:
        return simulation.solve()
    return simulation.solve(
        nonlinear_materials=nonlinear_materials,
        nonlinear_tolerance=nonlinear_tolerance,
    )


@dataclass
class FrequencyDomainSimulation:
    """Model-aware frequency-domain simulation facade."""

    model: FEMModel
    mesh: AxisymmetricMesh | None = None

    def create_mesh(self, **kwargs: Any) -> AxisymmetricMesh:
        self.mesh = build_mesh(self.model, **kwargs)
        return self.mesh

    def _currents(self) -> dict[str, float]:
        if self.model.excitation.circuits:
            return dict(self.model.excitation.circuits)
        return {
            circuit.name: circuit.current
            for circuit in self.model.circuits
        }

    def solve_voltage(
        self,
        voltage_v: complex,
        *,
        winding: str,
        nonlinear_materials: Mapping[str, NonlinearCoreMaterial] | None = None,
        max_iterations: int = 20,
        tolerance: float = 1e-3,
        relaxation: float = 0.7,
    ) -> FrequencyDomainResult:
        """Solve a winding to a target terminal-voltage phasor.

        Current is iteratively adjusted using the measured terminal voltage.
        Each trial uses the normal frequency-domain simulation path, including
        nonlinear material iteration when ``nonlinear_materials`` is supplied.
        The input model is not mutated and its mesh is reused across trials.
        """
        voltage_v = complex(voltage_v)
        if not np.isfinite(voltage_v.real) or not np.isfinite(voltage_v.imag):
            raise ValueError("voltage_v must be finite")
        if abs(voltage_v) == 0.0:
            raise ValueError("voltage_v must be non-zero")
        if isinstance(max_iterations, bool) or max_iterations < 1:
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("tolerance must be finite and positive")
        if not np.isfinite(relaxation) or not 0.0 < relaxation <= 1.0:
            raise ValueError("relaxation must be in (0, 1]")

        winding_spec = next(
            (item for item in self.model.windings if item.name == winding), None
        )
        if winding_spec is None or not winding_spec.circuit:
            raise ValueError(f"unknown winding or missing circuit: {winding!r}")
        circuit = winding_spec.circuit

        mesh = self.mesh if self.mesh is not None else self.create_mesh()
        currents = self._currents()
        current = complex(currents.get(circuit, 0.0))
        if abs(current) == 0.0:
            current = 1.0 + 0.0j

        for _iteration in range(max_iterations):
            component = deepcopy(self.model)
            trial_currents = dict(currents)
            trial_currents[circuit] = current
            component.excitation.circuits = trial_currents
            trial = FrequencyDomainSimulation(component, mesh=mesh).solve(
                nonlinear_materials=nonlinear_materials,
            )
            terminal_voltage_v = trial.winding_terminal_voltages().get(winding)
            if terminal_voltage_v is None:
                raise ValueError(f"no terminal voltage was produced for winding {winding!r}")
            if not np.isfinite(terminal_voltage_v.real) or not np.isfinite(terminal_voltage_v.imag):
                raise RuntimeError("voltage-driven solve produced a non-finite terminal voltage")
            if abs(terminal_voltage_v) == 0.0:
                raise RuntimeError("voltage-driven solve produced zero terminal voltage")
            relative_error = abs(terminal_voltage_v - voltage_v) / abs(voltage_v)
            if relative_error <= tolerance:
                self.mesh = mesh
                return trial

            correction = voltage_v / terminal_voltage_v
            current *= (1.0 - relaxation) + relaxation * correction

        raise RuntimeError(
            f"voltage excitation did not converge within {max_iterations} iterations"
        )

    def solve(
        self,
        *,
        nonlinear_materials: Mapping[str, NonlinearCoreMaterial] | None = None,
        nonlinear_tolerance: float | None = None,
    ) -> FrequencyDomainResult:
        """Solve the configured peak-current excitation and retain material data.

        Solid conductors use their assigned material conductivity and total
        current constraints. Litz bundles retain the existing impressed-density
        source model, with effective properties used for losses and voltages.
        Nonlinear core permeability and conductivity are forwarded together;
        ``nonlinear_tolerance`` optionally relaxes or tightens field convergence.
        """
        if nonlinear_tolerance is not None and (
            not np.isfinite(nonlinear_tolerance) or nonlinear_tolerance <= 0.0
        ):
            raise ValueError("nonlinear_tolerance must be finite and positive")
        if nonlinear_tolerance is not None and not nonlinear_materials:
            raise ValueError("nonlinear_tolerance requires nonlinear_materials")
        mesh = self.mesh
        if mesh is None:
            mesh = self.create_mesh()
        frequency = float(self.model.excitation.frequency)
        currents = self._currents()
        relative_permeability = _permeability_map(self.model)
        conductivity = _conductivity(self.model)
        from .solver import conductor_area

        current_modes = {}
        fixed_current_density = {}
        conductor_conductivities, litz_properties = conductor_properties(self.model, frequency)
        core_name = self.model.geometry.core.material if self.model.geometry.core else "core"
        _, core_conductivity = core_properties(
            _material_map(self.model).get(core_name, (mesh.materials or {}).get("core")), frequency,
        )
        conductor_index = 0
        for winding in self.model.windings:
            for turn in winding.turns:
                name = f"conductor_{conductor_index}"
                if name in litz_properties:
                    current_modes[name] = "fixed_current_density"
                    circuit = turn.circuit or winding.circuit
                    current = complex(currents.get(circuit, currents.get(winding.name, 0.0)))
                    current_fraction = (
                        turn.current_fraction
                        if turn.current_fraction is not None
                        else 1.0 / winding.parallel_paths
                    )
                    fixed_current_density[name] = (
                        current * turn.current_sign * current_fraction
                        / conductor_area(mesh, conductor_index)
                    )
                conductor_index += 1
        solve_options = {}
        if nonlinear_materials is not None:
            solve_options["nonlinear_materials"] = nonlinear_materials
        if nonlinear_tolerance is not None:
            solve_options["nonlinear_tolerance"] = nonlinear_tolerance
        solved = solve_frequency_domain(
            mesh,
            frequency=frequency,
            currents=currents,
            relative_permeability=relative_permeability,
            conductivity=conductivity,
            phasor_convention=self.model.excitation.phasor_convention,
            return_voltages=True,
            current_modes=current_modes,
            fixed_current_density=fixed_current_density,
            litz_properties=litz_properties,
            conductor_conductivities=conductor_conductivities,
            material_conductivities={"core": core_conductivity},
            **solve_options,
        )
        a_phi, densities, voltages = cast(
            tuple[np.ndarray, dict[str, np.ndarray], dict[str, complex]],
            solved,
        )
        return FrequencyDomainResult(
            model=self.model,
            mesh=mesh,
            a_phi=a_phi,
            current_densities=densities,
            turn_voltages=voltages,
            frequency_hz=frequency,
            conductivity=conductivity,
            relative_permeability=relative_permeability,
            nonlinear_material=(nonlinear_materials or {}).get("core"),
            nonlinear_materials=dict(nonlinear_materials or {}),
            litz_properties=litz_properties,
            conductor_conductivities=conductor_conductivities,
            core_conductivity=core_conductivity,
        )
