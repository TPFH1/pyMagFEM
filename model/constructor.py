"""Composable constructors for axisymmetric magnetic component geometries.

The classes in this module describe the physical groups independently:
conductors, windings, core, air gaps, and insulation.  ``Constructor``
then converts those specifications into the existing :class:`FEMModel` schema.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from itertools import combinations
from math import isclose, isfinite, pi, sqrt
from typing import Literal

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
    parallel_paths: int = 1

    def __post_init__(self) -> None:
        if (isinstance(self.parallel_paths, bool) or not isinstance(self.parallel_paths, int)
                or self.parallel_paths < 1):
            raise ValueError("parallel_paths must be a positive integer")
        if sum(len(layer.radial_centers) for layer in self.layers) % self.parallel_paths:
            raise ValueError("physical conductor count must be divisible by parallel_paths")

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
            parallel_paths=self.parallel_paths,
        )

    @property
    def turns_count(self) -> int:
        return sum(len(layer.radial_centers) for layer in self.layers) // self.parallel_paths


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
    """Core clearances and winding-to-winding insulation distances in metres.

    ``w2w_matrix`` is ordered like the supplied windings. Its diagonal gives
    same-winding bundle spacing, and off-diagonal entries give pairwise
    winding spacing. It must be square, symmetric, finite, and non-negative.
    When omitted, the legacy scalar ``w2w`` applies to every pair.
    """

    c2w_left: float = 0.0
    c2w_right: float = 0.0
    c2w_top: float = 0.0
    c2w_bottom: float = 0.0
    w2w: float = 0.0
    w2w_matrix: Sequence[Sequence[float]] | None = None

    def __post_init__(self) -> None:
        distances = (self.c2w_left, self.c2w_right, self.c2w_top, self.c2w_bottom, self.w2w)
        if not all(isfinite(value) and value >= 0 for value in distances):
            raise ValueError("insulation distances must be finite and non-negative")
        if self.w2w_matrix is not None:
            try:
                matrix = tuple(tuple(float(value) for value in row) for row in self.w2w_matrix)
            except (TypeError, ValueError) as exc:
                raise ValueError("w2w_matrix must be a square numeric matrix") from exc
            if not matrix or any(len(row) != len(matrix) for row in matrix):
                raise ValueError("w2w_matrix must be non-empty and square")
            if not all(isfinite(value) and value >= 0 for row in matrix for value in row):
                raise ValueError("w2w_matrix entries must be finite and non-negative")
            if any(not isclose(matrix[row][column], matrix[column][row], rel_tol=1e-12, abs_tol=1e-15)
                   for row in range(len(matrix)) for column in range(row)):
                raise ValueError("w2w_matrix must be symmetric")
            object.__setattr__(self, "w2w_matrix", matrix)

    def validate_winding_count(self, count: int) -> None:
        if self.w2w_matrix is not None and len(self.w2w_matrix) != count:
            raise ValueError("w2w_matrix size must match the number of windings")

    def winding_clearance(self, first: int, second: int) -> float:
        if self.w2w_matrix is None:
            return self.w2w
        if not (0 <= first < len(self.w2w_matrix) and 0 <= second < len(self.w2w_matrix)):
            raise ValueError("winding insulation index is out of range")
        return self.w2w_matrix[first][second]


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
        parallel_paths: int = 1,
        winding_index: int = 0,
    ) -> WindingSpec:
        """Place turns in a rectangular window using insulation clearances.

        ``traversal`` defines the first radial and axial directions. The
        optional serpentine traversal reverses the radial order on alternate
        axial rows, matching common layer-wound coils.
        ``turns`` is the electrical series turn count. ``parallel_paths``
        places that many equal-current physical bundles for each turn.
        ``winding_index`` selects the diagonal entry when the insulation
        contains a matrix; off-diagonal gaps are checked by ``build()``.
        """
        insulation = replace(
            insulation, w2w=insulation.winding_clearance(winding_index, winding_index),
            w2w_matrix=None,
        )
        if (isinstance(parallel_paths, bool) or not isinstance(parallel_paths, int)
                or parallel_paths < 1):
            raise ValueError("parallel_paths must be a positive integer")
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
        physical_turns = turns * parallel_paths
        rows = (physical_turns + radial_columns - 1) // radial_columns
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
        remaining = physical_turns
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
        result = WindingSpec(name, conductor, circuit, tuple(layers), current, parallel_paths)
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
        self.insulation.validate_winding_count(len(self.windings))
        winding_indices = {name: index for index, name in enumerate(winding_by_name)}
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
                winding_indices[layer.winding],
            )
            for layer in layers
            for radius in layer.radial_centers
        )
        tolerance = 1e-12
        for previous, current in combinations(intervals, 2):
            axial_clearance = current[0] - previous[1]
            radial_clearance = max(current[2] - previous[3], previous[2] - current[3])
            required_clearance = self.insulation.winding_clearance(previous[4], current[4])
            if max(axial_clearance, radial_clearance) < required_clearance - tolerance:
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
        if self.insulation.w2w_matrix is not None:
            core.extra["insulation_mm"]["w2w_matrix"] = [
                [value * 1e3 for value in row] for row in self.insulation.w2w_matrix
            ]
            core.extra["insulation_winding_order"] = [winding.name for winding in self.windings]
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


@dataclass(frozen=True)
class RadialWindingSpec:
    """Winding specification with physical bundles arranged in radial columns."""

    name: str
    conductor: ConductorSpec
    turns: int
    parallel_paths: int = 1
    alignment: Literal["bottom", "center", "top"] = "center"
    current: float = 0.0
    radial_columns: int = 1

    def __post_init__(self) -> None:
        for value in (self.turns, self.parallel_paths):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("turns and parallel_paths must be positive integers")
        if self.alignment not in {"bottom", "center", "top"}:
            raise ValueError("alignment must be 'bottom', 'center', or 'top'")
        if (isinstance(self.radial_columns, bool) or not isinstance(self.radial_columns, int)
                or self.radial_columns < 1):
            raise ValueError("radial_columns must be a positive integer")
        if self.radial_columns > self.turns * self.parallel_paths:
            raise ValueError("radial_columns cannot exceed the physical bundle count")


def build_concentric_model(
    *,
    core_area: float,
    windings: tuple[RadialWindingSpec, ...],
    insulation: InsulationSpec,
    air_gap_length: float = 0.0,
    name: str = "concentric magnetic component",
    winding_layout: Literal["concentric", "vertical"] = "concentric",
) -> FEMModel:
    """Size an equal-area axisymmetric core and place transformer windings.

    All lengths are in metres, ``core_area`` in square metres. In the default
    ``concentric`` layout, windings are ordered inside to outside. In ``vertical``
    layout, exactly two windings are ordered bottom to top and share the same
    radial origin; the off-diagonal insulation entry sets their axial gap. Each
    winding's ``radial_columns`` controls its physical bundle arrangement.
    Centre and outer legs have ``core_area``; the yokes have this minimum
    radial-flux cross section at the centre-leg radius. Materials and
    excitation frequency are intentionally left to the caller.
    """
    if not isfinite(core_area) or core_area <= 0:
        raise ValueError("core_area must be finite and positive")
    if not isfinite(air_gap_length) or air_gap_length < 0:
        raise ValueError("air_gap_length must be finite and non-negative")
    if winding_layout not in {"concentric", "vertical"}:
        raise ValueError("winding_layout must be 'concentric' or 'vertical'")
    if not windings or len({winding.name for winding in windings}) != len(windings):
        raise ValueError("windings must be non-empty and uniquely named")
    if winding_layout == "vertical" and len(windings) != 2:
        raise ValueError("vertical winding_layout requires exactly two windings")
    insulation.validate_winding_count(len(windings))
    rows = [
        (winding.turns * winding.parallel_paths + winding.radial_columns - 1)
        // winding.radial_columns
        for winding in windings
    ]
    heights = [
        row_count * winding.conductor.height
        + (row_count - 1) * insulation.winding_clearance(index, index)
        for index, (winding, row_count) in enumerate(zip(windings, rows))
    ]
    radial_widths = [
        winding.radial_columns * 2.0 * winding.conductor.radial_half_width
        + (winding.radial_columns - 1) * insulation.winding_clearance(index, index)
        for index, winding in enumerate(windings)
    ]
    available_height = max(heights)
    if winding_layout == "vertical":
        vertical_gap = insulation.winding_clearance(0, 1)
        window_height = (
            sum(heights) + vertical_gap + insulation.c2w_top + insulation.c2w_bottom
        )
        radial_stack_width = max(radial_widths)
        interwinding_gaps = 0.0
    else:
        window_height = available_height + insulation.c2w_top + insulation.c2w_bottom
        radial_stack_width = sum(radial_widths)
        interwinding_gaps = sum(
            insulation.winding_clearance(index, index + 1)
            for index in range(len(windings) - 1)
        )
    if air_gap_length >= window_height:
        raise ValueError("air gap must be shorter than the winding window")
    center_radius = sqrt(core_area / pi)
    window_outer_radius = (
        center_radius + insulation.c2w_left + radial_stack_width
        + interwinding_gaps + insulation.c2w_right
    )
    core = CoreSpec(
        "core", "core",
        outer_radius=sqrt(window_outer_radius**2 + core_area / pi),
        outer_half_height=window_height / 2.0 + core_area / (2.0 * pi * center_radius),
        window_inner_radius=center_radius,
        window_outer_radius=window_outer_radius,
        window_half_height=window_height / 2.0,
        core_type="Single",
    )
    placed = []
    if winding_layout == "vertical":
        for index, winding in enumerate(windings):
            bottom_offset = (
                0.0 if index == 0 else heights[0] + vertical_gap
            )
            top_offset = (
                heights[1] + vertical_gap if index == 0 else 0.0
            )
            local_insulation = InsulationSpec(
                c2w_left=insulation.c2w_left,
                c2w_right=insulation.c2w_right,
                c2w_bottom=insulation.c2w_bottom + bottom_offset,
                c2w_top=insulation.c2w_top + top_offset,
                w2w=insulation.winding_clearance(index, index),
            )
            placed.append(Constructor.place_winding(
                core=core, insulation=local_insulation, name=winding.name,
                conductor=winding.conductor, circuit=winding.name,
                turns=winding.turns, parallel_paths=winding.parallel_paths,
                radial_columns=winding.radial_columns, current=winding.current,
            ))
    else:
        radial_clearance = insulation.c2w_left
        for index, (winding, height) in enumerate(zip(windings, heights)):
            spare_height = available_height - height
            offset = {"bottom": 0.0, "center": spare_height / 2.0, "top": spare_height}[
                winding.alignment
            ]
            local_insulation = InsulationSpec(
                c2w_left=radial_clearance,
                c2w_right=insulation.c2w_right,
                c2w_bottom=insulation.c2w_bottom + offset,
                c2w_top=insulation.c2w_top + spare_height - offset,
                w2w=insulation.winding_clearance(index, index),
            )
            placed.append(Constructor.place_winding(
                core=core, insulation=local_insulation, name=winding.name,
                conductor=winding.conductor, circuit=winding.name,
                turns=winding.turns, parallel_paths=winding.parallel_paths,
                radial_columns=winding.radial_columns, current=winding.current,
            ))
            radial_clearance += radial_widths[index]
            if index + 1 < len(windings):
                radial_clearance += insulation.winding_clearance(index, index + 1)
    gaps = ((AirGapSpec("center_gap", center_radius / 2.0, 0.0,
                        center_radius, air_gap_length),) if air_gap_length else ())
    return Constructor(core, tuple(placed), gaps, insulation, {"name": name}).build()
