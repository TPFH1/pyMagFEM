"""Composable constructors for axisymmetric magnetic component geometries.

The classes in this module describe the physical groups independently:
conductors, windings, core, air gaps, and insulation.  ``Constructor``
then converts those specifications into the existing :class:`FEMModel` schema.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .schema import (
    Circuit,
    FEMModel,
    Geometry,
    Metadata,
    Region,
    TurnPlacement,
)
from .schema import (
    Conductor as ModelConductor,
)
from .schema import (
    Winding as ModelWinding,
)


@dataclass(frozen=True)
class ConductorSpec:
    """Base specification shared by round and rectangular conductors."""

    name: str
    material: str = "copper"
    construction: str = "solid"

    def to_model(self) -> ModelConductor:
        raise NotImplementedError

    @property
    def height(self) -> float:
        raise NotImplementedError

    @property
    def radial_half_width(self) -> float:
        raise NotImplementedError

    def area(self) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class RoundConductor(ConductorSpec):
    """Round conductor specified by radius in metres."""

    radius: float = 0.0
    strands: int | None = None
    strand_diameter: float | None = None

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("round conductor radius must be positive")
        if self.construction == "litz":
            if self.strands is None or self.strands <= 0:
                raise ValueError("litz conductors require a positive strand count")
            if self.strand_diameter is None or self.strand_diameter <= 0.0:
                raise ValueError("litz conductors require a positive strand diameter")

    @property
    def height(self) -> float:
        return 2.0 * self.radius

    @property
    def radial_half_width(self) -> float:
        return self.radius

    def area(self) -> float:
        import math

        return math.pi * self.radius**2

    @property
    def fill_factor(self) -> float:
        """Return the implied copper fill factor for a Litz conductor."""
        if self.construction != "litz":
            raise ValueError("fill_factor is only defined for Litz conductors")
        if self.strands is None or self.strand_diameter is None:
            raise ValueError("Litz strand data is incomplete")
        return self.strands * (self.strand_diameter / 2.0) ** 2 / self.radius**2

    def to_model(self) -> ModelConductor:
        return ModelConductor(
            name=self.name,
            shape="round",
            construction=self.construction,
            radius=self.radius,
            strands=self.strands,
            strand_diameter=self.strand_diameter,
            material=self.material,
        )


@dataclass(frozen=True)
class RectangularConductor(ConductorSpec):
    """Rectangular conductor specified by width and height in metres."""

    width: float = 0.0
    rectangular_height: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0.0 or self.rectangular_height <= 0.0:
            raise ValueError("rectangular conductor dimensions must be positive")

    @property
    def height(self) -> float:
        return self.rectangular_height

    @property
    def radial_half_width(self) -> float:
        return self.width / 2.0

    def area(self) -> float:
        return self.width * self.rectangular_height

    def to_model(self) -> ModelConductor:
        return ModelConductor(
            name=self.name,
            shape="rectangular",
            construction=self.construction,
            width=self.width,
            height=self.rectangular_height,
            material=self.material,
        )


@dataclass(frozen=True)
class WindingLayer:
    """One horizontal layer in a winding stack."""

    winding: str
    conductor: str
    z: float
    radial_centers: tuple[float, ...]
    circuit: str | None = None
    current_sign: float = 1.0

    def turns(self, default_circuit: str | None = None) -> list[TurnPlacement]:
        circuit = self.circuit or default_circuit or self.winding
        return [
            TurnPlacement(
                center=[radius, self.z],
                conductor=self.conductor,
                circuit=circuit,
                current_sign=self.current_sign,
            )
            for radius in self.radial_centers
        ]


@dataclass(frozen=True)
class WindingSpec:
    """Winding definition containing an ordered list of physical layers."""

    name: str
    conductor: ConductorSpec
    circuit: str
    layers: tuple[WindingLayer, ...]
    current: float = 0.0

    def model_winding(self) -> ModelWinding:
        turns = [
            turn
            for layer in self.layers
            for turn in layer.turns(self.circuit)
        ]
        return ModelWinding(
            name=self.name,
            turns=turns,
            conductor=self.conductor.name,
            circuit=self.circuit,
        )

    @property
    def turns_count(self) -> int:
        return sum(len(layer.radial_centers) for layer in self.layers)


@dataclass(frozen=True)
class CoreSpec:
    """Core and winding-window dimensions for a rectangular axisymmetric model."""

    name: str
    material: str
    outer_radius: float
    outer_half_height: float
    window_inner_radius: float
    window_outer_radius: float
    window_half_height: float
    core_type: str = "unspecified"

    def __post_init__(self) -> None:
        dimensions = (
            self.outer_radius,
            self.outer_half_height,
            self.window_inner_radius,
            self.window_outer_radius,
            self.window_half_height,
        )
        if any(value <= 0.0 for value in dimensions):
            raise ValueError("core dimensions must be positive")
        if self.window_inner_radius >= self.window_outer_radius:
            raise ValueError("core window radial bounds are invalid")
        if self.window_outer_radius > self.outer_radius:
            raise ValueError("core window cannot extend beyond the core")
        if self.window_half_height >= self.outer_half_height:
            raise ValueError("core window cannot extend beyond the core")

    def regions(self, window_half_height: float | None = None) -> tuple[Region, Region]:
        half_height = self.window_half_height if window_half_height is None else window_half_height
        core = Region(
            self.name,
            self.material,
            [
                [0.0, -self.outer_half_height],
                [self.outer_radius, -self.outer_half_height],
                [0.0, self.outer_half_height],
                [self.outer_radius, self.outer_half_height],
            ],
            extra={"core_type": self.core_type},
        )
        window = Region(
            "winding_window",
            "air",
            [
                [self.window_inner_radius, -half_height],
                [self.window_outer_radius, -half_height],
                [self.window_inner_radius, half_height],
                [self.window_outer_radius, half_height],
            ],
        )
        return core, window


@dataclass(frozen=True)
class AirGapSpec:
    """Air gap specified by centre, radial width, and axial length."""

    name: str
    center_r: float
    center_z: float
    radial_width: float
    axial_length: float

    def to_region(self) -> Region:
        return Region(
            self.name,
            "air",
            center=[self.center_r, self.center_z],
            size=[self.radial_width, self.axial_length],
        )


@dataclass(frozen=True)
class InsulationSpec:
    """Clearances between core and windings and between adjacent layers."""

    c2w_left: float = 0.0
    c2w_right: float = 0.0
    c2w_top: float = 0.0
    c2w_bottom: float = 0.0
    w2w: float = 0.0

    def __post_init__(self) -> None:
        if min(self.c2w_left, self.c2w_right, self.c2w_top, self.c2w_bottom, self.w2w) < 0:
            raise ValueError("insulation distances cannot be negative")


@dataclass
class Constructor:
    """Build an FEM model for an inductor or transformer."""

    core: CoreSpec
    windings: tuple[WindingSpec, ...]
    air_gaps: tuple[AirGapSpec, ...] = ()
    insulation: InsulationSpec = field(default_factory=InsulationSpec)
    metadata: dict[str, object] = field(default_factory=dict)

    @staticmethod
    def place_winding(
        *,
        core: CoreSpec,
        insulation: InsulationSpec,
        name: str,
        conductor: ConductorSpec,
        circuit: str,
        turns: int,
        radial_columns: int,
        current: float = 0.0,
        traversal: str = "inside2outside-bottom2top",
        serpentine: bool = True,
    ) -> WindingSpec:
        """Place turns in a rectangular window using insulation clearances.

        ``traversal`` defines the first radial and axial directions. The
        optional serpentine traversal reverses the radial order on alternate
        axial rows, matching common layer-wound coils.
        """
        if turns <= 0 or radial_columns <= 0:
            raise ValueError("turns and radial_columns must be positive")
        if traversal not in {
            "inside2outside-bottom2top",
            "outside2inside-bottom2top",
            "inside2outside-top2bottom",
            "outside2inside-top2bottom",
        }:
            raise ValueError(f"unsupported winding traversal: {traversal!r}")
        radial_forward = traversal.startswith("inside2outside")
        axial_forward = traversal.endswith("bottom2top")
        rows = (turns + radial_columns - 1) // radial_columns
        radial_pitch = 2.0 * conductor.radial_half_width + insulation.w2w
        axial_pitch = conductor.height + insulation.w2w
        radial_start = (
            core.window_inner_radius
            + insulation.c2w_left
            + conductor.radial_half_width
        )
        axial_start = (
            -core.window_half_height
            + insulation.c2w_bottom
            + conductor.height / 2.0
        )
        radial_centers = [
            round(radial_start + index * radial_pitch, 15)
            for index in range(radial_columns)
        ]
        if not radial_forward:
            radial_centers.reverse()
        axial_centers = [
            round(axial_start + index * axial_pitch, 15) for index in range(rows)
        ]
        if not axial_forward:
            axial_centers.reverse()
        layers: list[WindingLayer] = []
        remaining = turns
        for row, z in enumerate(axial_centers):
            count = min(radial_columns, remaining)
            row_centers = radial_centers[:count]
            if serpentine and row % 2:
                row_centers = list(reversed(row_centers))
            layers.append(WindingLayer(
                name,
                conductor.name,
                z,
                tuple(row_centers),
                circuit,
            ))
            remaining -= count
        result = WindingSpec(name, conductor, circuit, tuple(layers), current)
        probe = Constructor(
            core=core,
            windings=(result,),
            insulation=insulation,
        )
        lower, upper = probe._stack_limits()
        if lower < -core.window_half_height + insulation.c2w_bottom - 1e-12:
            raise ValueError("winding placement exceeds the bottom clearance")
        if upper > core.window_half_height - insulation.c2w_top + 1e-12:
            raise ValueError("winding placement exceeds the top clearance")
        if any(
            radius - conductor.radial_half_width
            < core.window_inner_radius + insulation.c2w_left - 1e-12
            or radius + conductor.radial_half_width
            > core.window_outer_radius - insulation.c2w_right + 1e-12
            for layer in result.layers
            for radius in layer.radial_centers
        ):
            raise ValueError("winding placement exceeds radial insulation clearances")
        return result

    def _stack_limits(self) -> tuple[float, float]:
        winding_by_name = {winding.name: winding for winding in self.windings}
        if len(winding_by_name) != len(self.windings):
            raise ValueError("winding names must be unique")
        layers = [layer for winding in self.windings for layer in winding.layers]
        if not layers:
            raise ValueError("at least one winding layer is required")
        for layer in layers:
            if layer.winding not in winding_by_name:
                raise ValueError(
                    f"layer references unknown winding {layer.winding!r}"
                )
            if layer.conductor != winding_by_name[layer.winding].conductor.name:
                raise ValueError(
                    f"layer conductor does not match winding {layer.winding!r}"
                )
        height_by_winding = {
            name: winding.conductor.height / 2.0
            for name, winding in winding_by_name.items()
        }
        radial_half_width = {
            name: winding.conductor.radial_half_width
            for name, winding in winding_by_name.items()
        }
        intervals = sorted(
            (
                layer.z - height_by_winding[layer.winding],
                layer.z + height_by_winding[layer.winding],
                radius - radial_half_width[layer.winding],
                radius + radial_half_width[layer.winding],
            )
            for layer in layers
            for radius in layer.radial_centers
        )
        tolerance = 1e-12
        for previous, current in combinations(intervals, 2):
            axial_clearance = current[0] - previous[1]
            radial_clearance = max(current[2] - previous[3], previous[2] - current[3])
            if max(axial_clearance, radial_clearance) < self.insulation.w2w - tolerance:
                raise ValueError("winding layers violate the winding-to-winding clearance")
        lower = min(
            layer.z - height_by_winding[layer.winding]
            for layer in layers
        )
        upper = max(
            layer.z + height_by_winding[layer.winding]
            for layer in layers
        )
        return lower, upper

    def build(self) -> FEMModel:
        lower, upper = self._stack_limits()
        window_lower = lower - self.insulation.c2w_bottom
        window_upper = upper + self.insulation.c2w_top
        window_half_height = max(abs(window_lower), abs(window_upper))
        if window_half_height > self.core.outer_half_height:
            raise ValueError("winding stack and insulation exceed core height")
        radial_half_width = {
            winding.name: winding.conductor.radial_half_width
            for winding in self.windings
        }
        radial_min = min(
            radius - radial_half_width[layer.winding]
            for winding in self.windings
            for layer in winding.layers
            for radius in layer.radial_centers
        )
        radial_max = max(
            radius + radial_half_width[layer.winding]
            for winding in self.windings
            for layer in winding.layers
            for radius in layer.radial_centers
        )
        if radial_min < self.core.window_inner_radius or radial_max > self.core.window_outer_radius:
            raise ValueError("winding stack and insulation exceed core window radius")
        core_ligament = self.core.outer_half_height - self.core.window_half_height
        core, window = self.core.regions(window_half_height)
        core.corners = [
            [point[0], -window_half_height - core_ligament]
            if point[1] < 0 else [point[0], window_half_height + core_ligament]
            for point in core.corners
        ]
        core.extra["insulation_mm"] = {
            "c2w_left": self.insulation.c2w_left * 1e3,
            "c2w_right": self.insulation.c2w_right * 1e3,
            "c2w_top": self.insulation.c2w_top * 1e3,
            "c2w_bottom": self.insulation.c2w_bottom * 1e3,
            "w2w": self.insulation.w2w * 1e3,
        }
        conductors = {winding.conductor.name: winding.conductor for winding in self.windings}
        model_windings = [winding.model_winding() for winding in self.windings]
        circuits = [
            Circuit(winding.circuit, turns=winding.turns_count, current=winding.current)
            for winding in self.windings
        ]
        return FEMModel(
            metadata=Metadata(
                name=str(self.metadata.get("name", "magnetic component")),
                extra=dict(self.metadata),
            ),
            geometry=Geometry(
                core=core,
                winding_window=window,
                air_gaps=[gap.to_region() for gap in self.air_gaps],
                extra=dict(self.metadata),
            ),
            conductors=[conductor.to_model() for conductor in conductors.values()],
            windings=model_windings,
            circuits=circuits,
        )


TransformerConstructor = Constructor
