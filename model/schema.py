"""Versioned, extensible JSON model schema for FEM simulations.

The public loader accepts both the current schema and the original flat geometry
example.  Normalization keeps the solver-facing geometry representation stable.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _points(value: Any, name: str) -> list[list[float]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list of points")
    result = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            raise ValueError(f"{name} contains an invalid point")
        result.append([float(point[0]), float(point[1])])
    return result


@dataclass
class Metadata:
    name: str = ""
    description: str = ""
    author: str = ""
    units: str = "m"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Region:
    name: str
    material: str = "air"
    corners: list[list[float]] = field(default_factory=list)
    center: list[float] | None = None
    size: list[float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Geometry:
    coordinate_system: str = "axisymmetric_rz"
    core: Region | None = None
    air_gaps: list[Region] = field(default_factory=list)
    shunts: list[Region] = field(default_factory=list)
    winding_window: Region | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Material:
    name: str
    relative_permeability: float = 1.0
    permeability_loss_angle_deg: float = 0.0
    conductivity: float = 0.0
    relative_permittivity: float = 0.0
    permittivity_loss_angle_deg: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Conductor:
    name: str
    shape: str = "round"
    construction: str = "solid"
    radius: float | None = None
    width: float | None = None
    height: float | None = None
    strands: int | None = None
    strand_diameter: float | None = None
    material: str = "copper"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TurnPlacement:
    center: list[float]
    conductor: str = "conductor_1"
    circuit: str = "circuit_1"
    current_sign: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Winding:
    name: str
    turns: list[TurnPlacement] = field(default_factory=list)
    conductor: str = "conductor_1"
    circuit: str = "circuit_1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Circuit:
    name: str
    turns: int = 0
    current: float = 0.0
    voltage: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Excitation:
    type: str = "current"
    frequency: float = 0.0
    phasor_convention: str = "cosine"
    time: float | None = None
    circuits: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MeshSettings:
    max_size: float | None = None
    algorithm: str = "default"
    order: int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SolverSettings:
    type: str = "axisymmetric_magnetostatic"
    tolerance: float = 1e-10
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OutputSettings:
    directory: str | None = None
    fields: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FEMModel:
    schema: str = "pyMagBEM"
    version: int = SCHEMA_VERSION
    metadata: Metadata = field(default_factory=Metadata)
    geometry: Geometry = field(default_factory=Geometry)
    materials: list[Material] = field(default_factory=list)
    conductors: list[Conductor] = field(default_factory=list)
    windings: list[Winding] = field(default_factory=list)
    circuits: list[Circuit] = field(default_factory=list)
    excitation: Excitation = field(default_factory=Excitation)
    mesh: MeshSettings = field(default_factory=MeshSettings)
    solver: SolverSettings = field(default_factory=SolverSettings)
    output: OutputSettings = field(default_factory=OutputSettings)
    extensions: dict[str, Any] = field(default_factory=dict)
    _legacy: dict[str, Any] = field(default_factory=dict, repr=False)

    def legacy_geometry(self) -> dict[str, Any]:
        """Return the normalized flat geometry mapping used by the FEM backend."""
        if self._legacy:
            return dict(self._legacy)
        if self.geometry.core is None or self.geometry.winding_window is None:
            raise ValueError("geometry requires core and winding_window")
        gaps = self.geometry.air_gaps
        conductors = self.conductors
        conductor = conductors[0] if conductors else Conductor("conductor_1")
        radius = conductor.radius or next(
            (item.radius for item in conductors if item.radius is not None), None
        )
        # Kept only as a numeric compatibility field for callers that inspect
        # the flat mapping; rectangular conductors use their per-turn dimensions.
        radius = float(radius or 0.0)
        winding_names = []
        winding_circuits = []
        turn_centers = []
        conductor_signs = []
        conductor_shapes = []
        conductor_widths = []
        conductor_heights = []
        conductor_radii = []
        conductors_by_name = {item.name: item for item in conductors}
        for winding in self.windings:
            for turn in winding.turns:
                definition = conductors_by_name.get(
                    turn.conductor or winding.conductor, conductor
                )
                if definition.shape == "rectangular":
                    if definition.width is None or definition.height is None:
                        raise ValueError(
                            f"rectangular conductor {definition.name!r} requires width and height"
                        )
                    conductor_shapes.append("rectangular")
                    conductor_widths.append(float(definition.width))
                    conductor_heights.append(float(definition.height))
                    conductor_radii.append(0.0)
                else:
                    if definition.radius is None:
                        raise ValueError(f"round conductor {definition.name!r} requires radius")
                    conductor_shapes.append("round")
                    conductor_widths.append(0.0)
                    conductor_heights.append(0.0)
                    conductor_radii.append(float(definition.radius))
                turn_centers.append(list(turn.center))
                winding_names.append(winding.name)
                winding_circuits.append(turn.circuit or winding.circuit)
                conductor_signs.append(float(turn.current_sign))
        return {
            "core_type": self.geometry.core.extra.get("core_type", "Single"),
            "p_outer": self.geometry.core.corners,
            "p_ww": self.geometry.winding_window.corners,
            "p_air_gap_center": [g.center for g in gaps if g.center is not None],
            "widths_air_gap": [float((g.size or [0, 0])[0]) for g in gaps],
            "lengths_air_gap": [float((g.size or [0, 0])[1]) for g in gaps],
            "p_cond_center_1": turn_centers,
            "conductor_signs": conductor_signs,
            "conductor_windings": winding_names,
            "conductor_circuits": winding_circuits,
            "radius_cond_1": radius,
            "conductor_shapes": conductor_shapes,
            "conductor_widths": conductor_widths,
            "conductor_heights": conductor_heights,
            "conductor_radii": conductor_radii,
        }


def _read(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, (str, Path)):
        with open(source, encoding="utf-8-sig") as stream:
            source = json.load(stream)
    if not isinstance(source, Mapping):
        raise TypeError("model source must be a path or mapping")
    return dict(source)


def _region(data: Mapping[str, Any], default_name: str, material: str = "air") -> Region:
    return Region(name=str(data.get("name", default_name)), material=str(data.get("material", material)),
                  corners=_points(data.get("corners", data.get("points", [])), default_name)
                  if data.get("corners", data.get("points")) else [],
                  center=[float(x) for x in data["center"]] if data.get("center") is not None else None,
                  size=[float(x) for x in data["size"]] if data.get("size") is not None else None,
                  extra={k: v for k, v in data.items() if k not in {"name", "material", "corners", "points", "center", "size"}})


def _from_legacy(data: Mapping[str, Any]) -> FEMModel:
    core = Region("core", "core", _points(data["p_outer"], "p_outer"), extra={"core_type": data.get("core_type", "Single")})
    window = Region("winding_window", "air", _points(data["p_ww"], "p_ww"))
    gaps = [Region(f"air_gap_{i + 1}", "air", center=_points([center], "p_air_gap_center")[0],
                   size=[0.0, float(length)]) for i, (center, length) in
            enumerate(zip(data["p_air_gap_center"], data["lengths_air_gap"]))]
    radius = float(data["radius_cond_1"])
    turns = [TurnPlacement([float(point[0]), float(point[1])]) for point in data["p_cond_center_1"]]
    model = FEMModel(metadata=Metadata(name="legacy model"), geometry=Geometry(core=core, air_gaps=gaps, winding_window=window),
                     conductors=[Conductor("conductor_1", radius=radius)],
                     windings=[Winding("winding_1", turns)], circuits=[Circuit("circuit_1", len(turns))],
                     _legacy=dict(data))
    return model


def normalize(source: str | Path | Mapping[str, Any] | FEMModel) -> FEMModel:
    """Load and normalize a current or legacy model into :class:`FEMModel`."""
    if isinstance(source, FEMModel):
        validate(source)
        return source
    data = _read(source)
    if "schema" not in data and "version" not in data and "p_outer" in data:
        model = _from_legacy(data)
    else:
        version = int(data.get("version", 1))
        if version > SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {version}")
        g = data.get("geometry", {})
        core = _region(g.get("core", {}), "core", "core")
        window = _region(g.get("winding_window", g.get("window", {})), "winding_window")
        gaps = [_region(x, f"air_gap_{i + 1}") for i, x in enumerate(g.get("air_gaps", []))]
        geometry = Geometry(g.get("coordinate_system", "axisymmetric_rz"), core, gaps,
                            [_region(x, f"shunt_{i + 1}") for i, x in enumerate(g.get("shunts", []))], window,
                            {k: v for k, v in g.items() if k not in {"coordinate_system", "core", "air_gaps", "shunts", "winding_window", "window"}})
        model = FEMModel(schema=str(data.get("schema", "pyMagBEM")), version=version, geometry=geometry)
        model.materials = [
            Material(
                name=str(x["name"]),
                relative_permeability=float(x.get("relative_permeability", 1.0)),
                permeability_loss_angle_deg=float(
                    x.get("permeability_loss_angle_deg", 0.0)
                ),
                conductivity=float(x.get("conductivity", 0.0)),
                relative_permittivity=float(x.get("relative_permittivity", 0.0)),
                permittivity_loss_angle_deg=float(
                    x.get("permittivity_loss_angle_deg", 0.0)
                ),
                properties=dict(x.get("properties", {})),
            )
            for x in data.get("materials", [])
        ]
        model.conductors = [Conductor(str(x["name"]), **{k: x[k] for k in ("shape", "construction", "radius", "width", "height", "strands", "strand_diameter", "material") if k in x}, metadata=dict(x.get("metadata", {}))) for x in data.get("conductors", [])]
        model.windings = [Winding(str(x["name"]), [TurnPlacement(list(t["center"]), t.get("conductor", "conductor_1"), t.get("circuit", "circuit_1"), float(t.get("current_sign", 1.0)), dict(t.get("metadata", {}))) for t in x.get("turns", [])], x.get("conductor", "conductor_1"), x.get("circuit", "circuit_1"), dict(x.get("metadata", {}))) for x in data.get("windings", [])]
        model.circuits = [Circuit(str(x["name"]), int(x.get("turns", 0)), float(x.get("current", 0.0)), x.get("voltage"), dict(x.get("metadata", {}))) for x in data.get("circuits", [])]
        model.metadata = Metadata(**{k: data.get("metadata", {}).get(k, getattr(Metadata(), k)) for k in ("name", "description", "author", "units")}, extra=dict(data.get("metadata", {}).get("extra", {})))
        model.excitation = Excitation(**{k: data.get("excitation", {}).get(k, getattr(Excitation(), k)) for k in ("type", "frequency", "phasor_convention", "time", "circuits")}, metadata=dict(data.get("excitation", {}).get("metadata", {})))
        model.mesh = MeshSettings(**{k: data.get("mesh", {}).get(k, getattr(MeshSettings(), k)) for k in ("max_size", "algorithm", "order")}, metadata=dict(data.get("mesh", {}).get("metadata", {})))
        model.solver = SolverSettings(
            type=data.get("solver", {}).get("type", SolverSettings().type),
            tolerance=float(
                data.get("solver", {}).get("tolerance", SolverSettings().tolerance)
            ),
            metadata=dict(data.get("solver", {}).get("metadata", {})),
        )
        model.output = OutputSettings(**{k: data.get("output", {}).get(k, getattr(OutputSettings(), k)) for k in ("directory", "fields")}, metadata=dict(data.get("output", {}).get("metadata", {})))
        model.extensions = dict(data.get("extensions", {}))
    validate(model)
    return model


def validate(model: FEMModel | Mapping[str, Any]) -> FEMModel:
    """Validate a normalized model and return it."""
    if not isinstance(model, FEMModel):
        model = normalize(model)
    if model.version < 1 or model.version > SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version {model.version}")
    if model.excitation.type not in {"current", "voltage"}:
        raise ValueError("excitation.type must be 'current' or 'voltage'")
    if model.excitation.phasor_convention not in {"cosine", "sine"}:
        raise ValueError("excitation.phasor_convention must be 'cosine' or 'sine'")
    if model.excitation.type != "current":
        raise ValueError("only current excitation is supported by the magnetostatic solver")
    if model.geometry.core is None or not model.geometry.core.corners:
        raise ValueError("geometry.core.corners is required")
    if model.geometry.winding_window is None or not model.geometry.winding_window.corners:
        raise ValueError("geometry.winding_window.corners is required")
    if not model.conductors and model.windings:
        raise ValueError("at least one conductor is required for windings")
    circuit_names = {c.name for c in model.circuits}
    if model.excitation.type == "current":
        unknown = set(model.excitation.circuits) - circuit_names
        if unknown:
            raise ValueError(f"excitation references unknown circuits: {sorted(unknown)}")
    for conductor in model.conductors:
        if conductor.shape not in {"round", "rectangle"}:
            raise ValueError(f"unsupported conductor shape: {conductor.shape}")
        if conductor.construction not in {"solid", "litz"}:
            raise ValueError(f"unsupported conductor construction: {conductor.construction}")
        if conductor.shape == "round" and (conductor.radius is None or conductor.radius <= 0):
            raise ValueError(f"conductor {conductor.name} needs a positive radius")
        if conductor.shape == "rectangle" and (conductor.width is None or conductor.height is None or
                                                 conductor.width <= 0 or conductor.height <= 0):
            raise ValueError(f"conductor {conductor.name} needs positive width and height")
        if conductor.construction == "litz" and (conductor.strands is None or conductor.strands < 1):
            raise ValueError(f"litz conductor {conductor.name} needs a positive strand count")
    return model


def load(source: str | Path | Mapping[str, Any]) -> FEMModel:
    """Read, migrate, validate, and normalize a model JSON document."""
    return normalize(source)
