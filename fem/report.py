"""Standard, JSON-serializable reports for frequency-domain simulations."""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .frequency_solver import average_abs_current_density, integrated_current
from .preprocess import AxisymmetricMesh


def _json_value(value: Any) -> Any:
    """Convert numerical values, including complex phasors, to JSON values."""
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, complex):
        return {
            "real": float(value.real),
            "imaginary": float(value.imag),
            "magnitude": float(abs(value)),
            "phase_deg": float(np.angle(value, deg=True)),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _loss_data(losses: Any) -> dict[str, Any]:
    if losses is None:
        return {}
    per_turn = getattr(losses, "per_turn", ())
    return {
        "per_turn_w": [
            {
                "total": float(loss.total),
                "transport": float(getattr(loss, "transport", 0.0)),
                "proximity": float(getattr(loss, "proximity", 0.0)),
            }
            for loss in per_turn
        ],
        "total_w": float(losses.total),
        "transport_w": float(getattr(losses, "transport", 0.0)),
        "proximity_w": float(getattr(losses, "proximity", 0.0)),
    }


def _model_data(model: Any) -> dict[str, Any]:
    """Return a model-shaped mapping without private loader state."""
    data = asdict(model)
    data.pop("_legacy", None)
    return _json_value(data)


@dataclass(frozen=True)
class SimulationResultReport:
    """Standard result report for inductors, transformers, and multi-core models."""

    data: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return _json_value(self.data)

    def write_json(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )
        return destination

    def print(self) -> None:
        """Print the report in a stable human-readable format."""
        simulation = self.data["simulation"]
        mesh = self.data["mesh"]
        print(f"Frequency: {simulation['frequency_hz']:,.0f} Hz")
        for name, current in simulation["currents_a"].items():
            print(f"Current {name}: {current:.3f} A")
        print(f"Mesh: {mesh['nodes']} nodes, {mesh['triangles']} triangles")
        for name, result in self.data["conductors"].items():
            print(
                f"Average |J| {name}: {result['average_abs_current_density_a_per_m2']:,.3e} "
                f"A/m²"
            )
            print(
                f"Maximum |J| {name}: "
                f"{result['maximum_abs_current_density_a_per_m2']:,.3e} A/m²"
            )
            print(f"Current constraint error {name}: {result['current_error_a']:.3e} A")
        for name, result in self.data["losses"]["windings"].items():
            if result.get("solid"):
                print(f"{name} solid loss per turn: "
                      f"{[item['total'] for item in result['solid']['per_turn_w']]} W")
                print(f"{name} solid winding loss: {result['solid']['total_w']:.4f} W")
            if result.get("litz"):
                print(f"{name} Litz loss per turn: "
                      f"{[item['total'] for item in result['litz']['per_turn_w']]} W")
                print(f"{name} Litz winding loss: {result['litz']['total_w']:.4f} W")
        for name, result in self.data["losses"]["cores"].items():
            print(f"{name} magnetic loss: {result['magnetic_w']:.4f} W")
            print(f"{name} electrical loss: {result['electrical_w']:.4f} W")
            print(f"{name} total loss: {result['total_w']:.4f} W")
        for name, voltage in self.data["voltages"]["turns"].items():
            print(
                f"{name} voltage: {voltage['magnitude']:.4f} V "
                f"({voltage['phase_deg']:.2f}°)"
            )
        source_voltage = self.data["voltages"]["turn_source_total"]
        if source_voltage:
            print(
                f"Total ring-source voltage magnitude: "
                f"{source_voltage['magnitude']:.4f} V"
            )
            print(
                f"Total ring-source voltage phase: "
                f"{source_voltage['phase_deg']:.2f}°"
            )
        for name, voltage in self.data["voltages"]["terminals"].items():
            print(
                f"{name} terminal voltage: {voltage['real']:.4f} "
                f"+ j{voltage['imaginary']:.4f} V"
            )
        terminal_voltage = self.data["voltages"]["terminal_total"]
        if terminal_voltage:
            print(
                f"Total terminal voltage: {terminal_voltage['real']:.4f} "
                f"+ j{terminal_voltage['imaginary']:.4f} V"
            )
        if self.data["inductance"]:
            for name, value in self.data["inductance"].items():
                print(
                    f"{name} flux-linkage inductance magnitude: "
                    f"{value['magnitude'] * 1e6:.4f} µH"
                )
                print(
                    f"{name} flux-linkage inductance phase: "
                    f"{value['phase_deg']:.2f}°"
                )
        print(f"Maximum |B|: {self.data['fields']['maximum_abs_b_t']:.4f} T")


def build_result_report(
    model: Any,
    mesh: AxisymmetricMesh,
    *,
    frequency_hz: float,
    currents_a: float | Mapping[str, float],
    current_densities: Mapping[str, np.ndarray],
    flux_density: np.ndarray,
    inductance_h: complex | Mapping[str, complex] | None = None,
    solid_losses: Any = None,
    litz_losses: Any = None,
    core_losses: tuple[float, float, float]
    | Mapping[str, tuple[float, float, float]] = (0.0, 0.0, 0.0),
    turn_voltages: Mapping[str, complex] | None = None,
    terminal_voltages: Mapping[str, complex] | None = None,
) -> SimulationResultReport:
    """Build a standard report from solver and post-processing results.

    Conductor, winding, and core entries are keyed by their model names where
    available. The optional loss mappings can therefore represent arbitrary
    numbers of windings and core parts without changing the report schema.
    """
    current_map = (
        {"current": float(currents_a)}
        if not isinstance(currents_a, Mapping)
        else {str(name): float(value) for name, value in currents_a.items()}
    )
    conductor_names = list((mesh.conductor_regions or {}).keys())
    geometry = mesh.geometry
    winding_names = geometry.get("conductor_windings", [])
    current_signs = geometry.get("conductor_signs", [])
    expected_currents = list(current_map.values())
    conductors: dict[str, Any] = {}
    for index, name in enumerate(conductor_names):
        density = np.asarray(current_densities[name])
        winding = (
            str(winding_names[index])
            if index < len(winding_names)
            else name
        )
        sign = float(current_signs[index]) if index < len(current_signs) else 1.0
        expected = (
            (current_map.get(winding)
             if winding in current_map
             else current_map.get("current")
             if "current" in current_map
             else (expected_currents[0] if expected_currents else 0.0))
            * sign
        )
        actual = integrated_current(mesh, density, name)
        conductors[name] = {
            "winding": winding,
            "average_abs_current_density_a_per_m2": (
                average_abs_current_density(mesh, density, name)
            ),
            "maximum_abs_current_density_a_per_m2": float(np.abs(density).max()),
            "integrated_current_a": actual,
            "current_error_a": float(abs(actual - expected)),
        }

    core_data = {}
    if isinstance(core_losses, Mapping):
        core_items = core_losses.items()
    else:
        core_items = [("core", core_losses)]
    for name, losses in core_items:
        magnetic, electrical, total = losses
        core_data[str(name)] = {
            "magnetic_w": float(magnetic),
            "electrical_w": float(electrical),
            "total_w": float(total),
        }

    winding_data: dict[str, Any] = {}
    solid_by_winding = solid_losses if isinstance(solid_losses, Mapping) else {}
    litz_by_winding = litz_losses if isinstance(litz_losses, Mapping) else {}
    if solid_losses is not None and not solid_by_winding:
        solid_by_winding = {"winding": solid_losses}
    if litz_losses is not None and not litz_by_winding:
        litz_by_winding = {"winding": litz_losses}
    for name in set(solid_by_winding) | set(litz_by_winding):
        winding_data[str(name)] = {}
        if name in solid_by_winding:
            winding_data[str(name)]["solid"] = _loss_data(solid_by_winding[name])
        if name in litz_by_winding:
            winding_data[str(name)]["litz"] = _loss_data(litz_by_winding[name])

    if inductance_h is None:
        inductance_data = {}
    elif isinstance(inductance_h, Mapping):
        inductance_data = _json_value(inductance_h)
    else:
        inductance_data = {"winding": _json_value(inductance_h)}

    turn_voltage_data = _json_value(turn_voltages or {})
    terminal_voltage_data = _json_value(terminal_voltages or {})
    turn_source_total = (
        _json_value(sum(turn_voltages.values()))
        if turn_voltages
        else None
    )
    terminal_total = (
        _json_value(sum(terminal_voltages.values()))
        if terminal_voltages
        else None
    )
    data = {
        "schema": "pyMagBEM.result",
        "version": 1,
        "model": _model_data(model),
        "simulation": {
            "frequency_hz": float(frequency_hz),
            "phasor_convention": getattr(model.excitation, "phasor_convention", "cosine"),
            "currents_a": current_map,
        },
        "mesh": {
            "nodes": len(mesh.points),
            "triangles": len(mesh.triangles),
            "regions": {
                str(name): int(count)
                for name, count in zip(*np.unique(mesh.regions, return_counts=True))
            },
        },
        "conductors": conductors,
        "losses": {"windings": winding_data, "cores": core_data},
        "voltages": {
            "turns": turn_voltage_data,
            "turn_source_total": turn_source_total,
            "terminals": terminal_voltage_data,
            "terminal_total": terminal_total,
        },
        "inductance": inductance_data,
        "fields": {
            "maximum_abs_b_t": float(np.linalg.norm(flux_density, axis=1).max()),
        },
    }
    return SimulationResultReport(data)
