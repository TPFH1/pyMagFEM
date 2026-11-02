"""High-level frequency-domain simulation API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np

from model import FEMModel

from .frequency_solver import solve_frequency_domain
from .physics import complex_relative_permeability
from .postprocess import (
    WindingLosses,
    core_losses,
    magnetic_field_intensity,
    magnetic_flux_density,
    winding_copper_losses,
    winding_flux_linkages,
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
    """Solved fields with model context for safe post-processing."""

    model: FEMModel
    mesh: AxisymmetricMesh
    a_phi: np.ndarray
    current_densities: dict[str, np.ndarray]
    turn_voltages: dict[str, complex]
    frequency_hz: float
    conductivity: float
    relative_permeability: dict[str, complex]
    nonlinear_material: Any | None = None

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
    ) -> FrequencyDomainResult:
        """Wrap an externally solved field in the model-aware result API."""
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
        )

    def magnetic_flux_density(self) -> np.ndarray:
        return magnetic_flux_density(self.mesh, self.a_phi)

    def magnetic_field_intensity(self) -> np.ndarray:
        return magnetic_field_intensity(
            self.mesh,
            self.a_phi,
            relative_permeability=self.relative_permeability,
            nonlinear_material=self.nonlinear_material,
        )

    def winding_copper_losses(self) -> WindingLosses:
        return winding_copper_losses(
            self.mesh,
            self.current_densities,
            conductivity=self.conductivity,
        )

    def core_losses(self) -> tuple[float, float, float]:
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
                float(self.relative_permeability["core"].real)
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
        )

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
        )

    def plot_physical_field(self, field: Any, **kwargs: Any) -> Any:
        """Plot a solved field without repeating mesh or result arguments."""
        from .plot import plot_physical_field

        kwargs.setdefault("a_phi", self.a_phi)
        kwargs.setdefault("current_densities", self.current_densities)
        kwargs.setdefault("magnetic_field", self.magnetic_field_intensity())
        kwargs.setdefault("nonlinear_material", self.nonlinear_material)
        kwargs.setdefault("frequency", self.frequency_hz)
        return plot_physical_field(self.mesh, field, **kwargs)

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
        turn_offset = 0
        for winding in self.model.windings:
            conductor_names = [
                f"conductor_{index}"
                for index in range(turn_offset, turn_offset + len(winding.turns))
            ]
            if conductor_names:
                solid_losses[winding.name] = winding_copper_losses(
                    self.mesh,
                    self.current_densities,
                    conductivity=self.conductivity,
                    conductor_names=conductor_names,
                )
            turn_offset += len(winding.turns)
        if not self.model.windings:
            winding_name = next(iter(linkages), "winding")
            solid_losses[winding_name] = self.winding_copper_losses()
        return build_result_report(
            self.model,
            self.mesh,
            frequency_hz=self.frequency_hz,
            currents_a=currents,
            current_densities=self.current_densities,
            flux_density=self.magnetic_flux_density(),
            inductance_h=inductances,
            solid_losses=solid_losses,
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

    def solve(self) -> FrequencyDomainResult:
        mesh = self.mesh
        if mesh is None:
            mesh = self.create_mesh()
        frequency = float(self.model.excitation.frequency)
        currents = self._currents()
        relative_permeability = _permeability_map(self.model)
        conductivity = _conductivity(self.model)
        solved = solve_frequency_domain(
            mesh,
            frequency=frequency,
            currents=currents,
            relative_permeability=relative_permeability,
            conductivity=conductivity,
            phasor_convention=self.model.excitation.phasor_convention,
            return_voltages=True,
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
        )
