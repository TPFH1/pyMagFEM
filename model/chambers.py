"""Placement-only chambers and rectangular chamber splitting."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations, pairwise
from math import ceil, floor, isfinite, pi, sqrt

from .constructor import AirGapSpec, ConductorSpec, CoreSpec, InsulationSpec
from .schema import Circuit, FEMModel, Geometry, Metadata, Region, TurnPlacement, Winding


class SplitDirection(str, Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class Align(str, Enum):
    Left = "left"
    Center = "center"
    Right = "right"
    Top = "top"
    Bot = "bottom"


class Distribute(str, Enum):
    Horizontally = "horizontally"
    Vertically = "vertically"


class WindingStart(str, Enum):
    Bottom = "bottom"
    Top = "top"


@dataclass
class Chamber:
    """Axis-aligned placement area in the meridian (r, z) plane, in metres."""

    name: str
    r_min: float
    r_max: float
    z_min: float
    z_max: float
    parent: str | None = None
    split_direction: SplitDirection | None = None
    split_positions_percent: tuple[float, ...] = ()
    split_insulation_m: float = 0.0
    children: list[Chamber] = field(default_factory=list)

    def __post_init__(self) -> None:
        bounds = (self.r_min, self.r_max, self.z_min, self.z_max)
        if not self.name:
            raise ValueError("chamber name must be non-empty")
        if not all(isfinite(value) for value in bounds):
            raise ValueError("chamber bounds must be finite")
        if self.r_min >= self.r_max or self.z_min >= self.z_max:
            raise ValueError("chamber bounds must have positive width and height")
        if not isfinite(self.split_insulation_m) or self.split_insulation_m < 0:
            raise ValueError("split insulation must be finite and non-negative")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.r_min, self.r_max, self.z_min, self.z_max

    @property
    def width(self) -> float:
        return self.r_max - self.r_min

    @property
    def height(self) -> float:
        return self.z_max - self.z_min

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "bounds_m": list(self.bounds),
            "parent": self.parent,
            "split_direction": self.split_direction.value if self.split_direction else None,
            "split_positions_percent": list(self.split_positions_percent),
            "split_insulation_m": self.split_insulation_m,
            "children": [child.to_dict() for child in self.children],
        }


@dataclass(frozen=True)
class WindingDefinition:
    """Electrical winding data shared by its chamber assignments."""

    name: str
    conductor: ConductorSpec
    turns: int
    circuit: str | None = None
    current: float = 0.0
    turn_insulation: float | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("winding name must be non-empty")
        if isinstance(self.turns, bool) or not isinstance(self.turns, int) or self.turns < 1:
            raise ValueError("winding turns must be a positive integer")
        if not isfinite(self.current):
            raise ValueError("winding current must be finite")
        if (self.turn_insulation is not None
            and (not isfinite(self.turn_insulation) or self.turn_insulation < 0)):
            raise ValueError("turn_insulation must be finite and non-negative")

    @property
    def circuit_name(self) -> str:
        return self.circuit or self.name


def _location(value: Align | Distribute | str, axis: str) -> Align | Distribute:
    if isinstance(value, (Align, Distribute)):
        location = value
    else:
        try:
            location = Align(value)
        except ValueError:
            try:
                location = Distribute(value)
            except ValueError as exc:
                raise ValueError(f"invalid {axis} location: {value!r}") from exc
    allowed = (
        {Align.Left, Align.Center, Align.Right, Distribute.Horizontally}
        if axis == "horizontal"
        else {Align.Top, Align.Center, Align.Bot, Distribute.Vertically}
    )
    if location not in allowed:
        raise ValueError(f"{location.value!r} is not a valid {axis} location")
    return location


def _validate_turn_definition(
    turns: int | None,
    conductor_count: int | None,
    parallel_paths: int,
    label: str,
) -> None:
    if (turns is None) == (conductor_count is None):
        raise ValueError(f"{label} must specify exactly one of turns or conductor_count")
    if turns is not None and (
        isinstance(turns, bool) or not isinstance(turns, int) or turns < 1
    ):
        raise ValueError(f"{label} turns must be a positive integer")
    if conductor_count is not None and (
        isinstance(conductor_count, bool)
        or not isinstance(conductor_count, int)
        or conductor_count < 1
    ):
        raise ValueError(f"{label} conductor_count must be a positive integer")


@dataclass(frozen=True)
class ChamberWindingAssignment:
    """Place physical conductors for one winding in a leaf chamber."""

    chamber: Chamber
    winding: str
    turns: int | None = None
    parallel_paths: int = 1
    layers: int | None = None
    h_loc: Align | Distribute | str = Align.Center
    v_loc: Align | Distribute | str = Align.Center
    start: WindingStart | str = WindingStart.Bottom
    serpentine: bool = True
    conductor_count: int | None = None

    def __post_init__(self) -> None:
        if not self.winding:
            raise ValueError("assignment winding must be non-empty")
        if (isinstance(self.parallel_paths, bool) or not isinstance(self.parallel_paths, int)
                or self.parallel_paths < 1):
            raise ValueError("assignment parallel_paths must be a positive integer")
        _validate_turn_definition(
            self.turns,
            self.conductor_count,
            self.parallel_paths,
            "assignment",
        )
        if self.layers is not None and (
            isinstance(self.layers, bool) or not isinstance(self.layers, int) or self.layers < 1
        ):
            raise ValueError("assignment layers must be a positive integer")
        object.__setattr__(self, "h_loc", _location(self.h_loc, "horizontal"))
        object.__setattr__(self, "v_loc", _location(self.v_loc, "vertical"))
        try:
            object.__setattr__(self, "start", WindingStart(self.start))
        except ValueError as exc:
            raise ValueError("start must be 'bottom' or 'top'") from exc
        if not isinstance(self.serpentine, bool):
            raise TypeError("serpentine must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "chamber": self.chamber.name,
            "winding": self.winding,
            "turns": self.turns,
            "conductor_count": self.conductor_count,
            "parallel_paths": self.parallel_paths,
            "layers": self.layers,
            "h_loc": self.h_loc.value,
            "v_loc": self.v_loc.value,
            "start": self.start.value,
            "serpentine": self.serpentine,
        }

    @property
    def physical_conductor_count(self) -> int:
        if self.conductor_count is not None:
            return self.conductor_count
        assert self.turns is not None
        return self.turns * self.parallel_paths


@dataclass(frozen=True)
class WindingSection:
    """One winding section with an integer physical conductor count.

    ``turns`` remains available for legacy whole-turn sections. New definitions
    should use ``conductor_count`` to specify the physical conductors placed in
    the section.
    """

    winding: str
    turns: int | None = None
    parallel_paths: int = 1
    layers: int = 1
    h_loc: Align | Distribute | str = Align.Center
    v_loc: Align | Distribute | str = Align.Center
    start: WindingStart | str = WindingStart.Bottom
    serpentine: bool = True
    conductor_count: int | None = None

    def __post_init__(self) -> None:
        if not self.winding:
            raise ValueError("section winding must be non-empty")
        for value, name in (
            (self.parallel_paths, "parallel_paths"),
            (self.layers, "layers"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"section {name} must be a positive integer")
        _validate_turn_definition(
            self.turns,
            self.conductor_count,
            self.parallel_paths,
            "section",
        )
        object.__setattr__(self, "h_loc", _location(self.h_loc, "horizontal"))
        object.__setattr__(self, "v_loc", _location(self.v_loc, "vertical"))
        try:
            object.__setattr__(self, "start", WindingStart(self.start))
        except ValueError as exc:
            raise ValueError("start must be 'bottom' or 'top'") from exc
        if not isinstance(self.serpentine, bool):
            raise TypeError("serpentine must be a boolean")

    def to_assignment(self, chamber: Chamber) -> ChamberWindingAssignment:
        return ChamberWindingAssignment(
            chamber=chamber,
            winding=self.winding,
            turns=self.turns,
            conductor_count=self.conductor_count,
            parallel_paths=self.parallel_paths,
            layers=self.layers,
            h_loc=self.h_loc,
            v_loc=self.v_loc,
            start=self.start,
            serpentine=self.serpentine,
        )

    @property
    def physical_conductor_count(self) -> int:
        if self.conductor_count is not None:
            return self.conductor_count
        assert self.turns is not None
        return self.turns * self.parallel_paths


@dataclass(frozen=True)
class WindingSchemeLayout:
    """Resolved winding chambers and derived axisymmetric core-window bounds."""

    scheme: str
    core: CoreSpec
    root_chamber: Chamber
    windings: tuple[WindingDefinition, ...]
    assignments: tuple[ChamberWindingAssignment, ...]
    insulation: InsulationSpec
    chamber_width_m: float
    chamber_height_m: float

    @property
    def window_outer_radius_m(self) -> float:
        return self.core.window_outer_radius

    @property
    def window_height_m(self) -> float:
        return 2.0 * self.core.window_half_height

    def air_gaps_from_positions(
        self,
        definitions: Sequence[Mapping[str, float]],
    ) -> tuple[AirGapSpec, ...]:
        """Build full-center-leg gaps from window-relative position definitions."""
        air_gaps = []
        for index, definition in enumerate(definitions):
            if not isinstance(definition, Mapping):
                raise TypeError(f"air gap definition {index + 1} must be a mapping")
            try:
                axial_length = definition["axial_length_m"]
                position_percent = definition["position_percent"]
            except KeyError as error:
                raise ValueError(
                    f"air gap definition {index + 1} requires position_percent "
                    "and axial_length_m"
                ) from error
            air_gaps.append(AirGapSpec.from_window_position_percent(
                f"center_gap_{index + 1}",
                center_r=self.core.window_inner_radius / 2.0,
                radial_width=self.core.window_inner_radius,
                axial_length=axial_length,
                position_percent=position_percent,
                window_half_height=self.core.window_half_height,
            ))
        return tuple(air_gaps)

    def build_model(
        self,
        *,
        air_gaps: Sequence[AirGapSpec] = (),
        name: str = "winding scheme component",
    ) -> FEMModel:
        component = build_chambered_model(
            core=self.core,
            windings=self.windings,
            root_chamber=self.root_chamber,
            assignments=self.assignments,
            insulation=self.insulation,
            air_gaps=air_gaps,
            name=name,
        )
        component.geometry.extra["winding_scheme"] = {
            "name": self.scheme,
            "chamber_width_m": self.chamber_width_m,
            "chamber_height_m": self.chamber_height_m,
            "window_outer_radius_m": self.window_outer_radius_m,
            "window_height_m": self.window_height_m,
        }
        return component


@dataclass(frozen=True)
class RadialStack:
    """Place winding sections radially from inner to outer."""

    sections: Sequence[WindingSection]
    split_ins: float

    def layout(
        self,
        *,
        core_area: float,
        windings: Sequence[WindingDefinition],
        insulation: InsulationSpec,
    ) -> WindingSchemeLayout:
        return _layout_winding_scheme(
            scheme="radial_inside_to_outside",
            sections=self.sections,
            split_ins=self.split_ins,
            core_area=core_area,
            windings=windings,
            insulation=insulation,
        )


@dataclass(frozen=True)
class AxialStack:
    """Place winding sections from top to bottom in the supplied order."""

    sections: Sequence[WindingSection]
    split_ins: float

    def layout(
        self,
        *,
        core_area: float,
        windings: Sequence[WindingDefinition],
        insulation: InsulationSpec,
    ) -> WindingSchemeLayout:
        return _layout_winding_scheme(
            scheme="axial_top_to_bottom",
            sections=self.sections,
            split_ins=self.split_ins,
            core_area=core_area,
            windings=windings,
            insulation=insulation,
        )


def _section_envelope(
    section: WindingSection,
    winding: WindingDefinition,
    turn_insulation: float,
) -> tuple[float, float]:
    physical_count = section.physical_conductor_count
    if section.layers > physical_count:
        raise ValueError("section layers cannot exceed physical conductor count")
    bundle_width = 2.0 * winding.conductor.radial_half_width
    bundle_height = winding.conductor.height
    turns_per_layer, extra_turns = divmod(physical_count, section.layers)
    tallest_layer = turns_per_layer + (1 if extra_turns else 0)
    width = section.layers * bundle_width + (section.layers - 1) * turn_insulation
    height = tallest_layer * bundle_height + (tallest_layer - 1) * turn_insulation
    return width, height


def _divider_positions_percent(section_spans: Sequence[float], split_ins: float) -> tuple[float, ...]:
    total_span = sum(section_spans) + (len(section_spans) - 1) * split_ins
    positions = []
    accumulated = 0.0
    for section_span in section_spans[:-1]:
        accumulated += section_span + split_ins / 2.0
        positions.append(accumulated / total_span * 100.0)
        accumulated += split_ins / 2.0
    return tuple(positions)


def _layout_winding_scheme(
    *,
    scheme: str,
    sections: Sequence[WindingSection],
    split_ins: float,
    core_area: float,
    windings: Sequence[WindingDefinition],
    insulation: InsulationSpec,
) -> WindingSchemeLayout:
    if not sections:
        raise ValueError("a winding scheme must contain at least one section")
    if not isfinite(split_ins) or split_ins < 0.0:
        raise ValueError("split_ins must be finite and non-negative")
    winding_by_name = {winding.name: winding for winding in windings}
    if len(winding_by_name) != len(windings) or not winding_by_name:
        raise ValueError("windings must be non-empty and uniquely named")
    insulation.validate_winding_count(len(windings))

    turn_insulation = {
        winding.name: (
            winding.turn_insulation
            if winding.turn_insulation is not None
            else insulation.winding_clearance(index, index)
        )
        for index, winding in enumerate(windings)
    }
    conductor_counts_by_winding = {name: 0 for name in winding_by_name}
    paths_by_winding: dict[str, int] = {}
    section_sizes = []
    for section in sections:
        winding = winding_by_name.get(section.winding)
        if winding is None:
            raise ValueError(f"unknown winding in section: {section.winding!r}")
        conductor_counts_by_winding[section.winding] += section.physical_conductor_count
        paths = paths_by_winding.setdefault(section.winding, section.parallel_paths)
        if paths != section.parallel_paths:
            raise ValueError(
                f"parallel_paths must match across sections for winding {section.winding!r}"
            )
        section_sizes.append(
            _section_envelope(section, winding, turn_insulation[section.winding])
        )
    for winding in windings:
        paths = paths_by_winding.get(winding.name)
        expected_conductor_count = winding.turns * paths if paths is not None else -1
        actual_conductor_count = conductor_counts_by_winding[winding.name]
        if actual_conductor_count != expected_conductor_count:
            raise ValueError(
                f"winding {winding.name!r} has {actual_conductor_count} section conductors; "
                f"expected {expected_conductor_count} from turns and parallel_paths"
            )

    if scheme == "radial_inside_to_outside":
        section_widths = [width for width, _height in section_sizes]
        chamber_width = sum(section_widths) + (len(sections) - 1) * split_ins
        chamber_height = max(height for _width, height in section_sizes)
        split_direction = SplitDirection.HORIZONTAL
        split_spans = section_widths
    elif scheme == "axial_top_to_bottom":
        section_heights = [height for _width, height in section_sizes]
        chamber_width = max(width for width, _height in section_sizes)
        chamber_height = sum(section_heights) + (len(sections) - 1) * split_ins
        split_direction = SplitDirection.VERTICAL
        split_spans = list(reversed(section_heights))
    else:
        raise ValueError(f"unsupported winding scheme: {scheme!r}")

    if not isfinite(core_area) or core_area <= 0.0:
        raise ValueError("core_area must be finite and positive")
    center_radius = sqrt(core_area / pi)
    window_outer_radius = (
        center_radius + insulation.c2w_left + chamber_width + insulation.c2w_right
    )
    window_height = chamber_height + insulation.c2w_top + insulation.c2w_bottom
    core = core_for_window(
        core_area=core_area,
        window_outer_radius=window_outer_radius,
        window_half_height=window_height / 2.0,
    )
    root = initial_chamber(core, insulation)

    if len(sections) == 1:
        chambers = (root,)
        section_chambers = chambers
    else:
        children = split_chamber(
            root,
            split_dir=split_direction,
            positions_percent=_divider_positions_percent(split_spans, split_ins),
            split_ins=split_ins,
        )
        if scheme == "radial_inside_to_outside":
            section_chambers = children
        else:
            section_chambers = tuple(reversed(children))

    assignments = tuple(
        section.to_assignment(chamber)
        for section, chamber in zip(sections, section_chambers, strict=True)
    )
    return WindingSchemeLayout(
        scheme=scheme,
        core=core,
        root_chamber=root,
        windings=tuple(windings),
        assignments=assignments,
        insulation=insulation,
        chamber_width_m=chamber_width,
        chamber_height_m=chamber_height,
    )


def split_chamber(
    chamber: Chamber,
    *,
    split_dir: SplitDirection | str,
    split_ins: float,
    count: int | None = None,
    positions_percent: Sequence[float] | None = None,
) -> tuple[Chamber, ...]:
    """Split a chamber into equal or explicitly positioned rectangular children.

    ``split_dir="vertical"`` creates vertically stacked chambers, dividing the
    z span. ``split_dir="horizontal"`` divides the r span. Explicit positions
    are percentages measured from the low-coordinate edge of the parent span.
    Each insulation gap is centered on its divider and removes half its width
    from each neighboring child.
    """
    try:
        direction = SplitDirection(split_dir)
    except ValueError as exc:
        raise ValueError("split_dir must be 'horizontal' or 'vertical'") from exc
    if not isfinite(split_ins) or split_ins < 0:
        raise ValueError("split_ins must be finite and non-negative")
    if chamber.children:
        raise ValueError(f"chamber {chamber.name!r} has already been split")
    if (count is None) == (positions_percent is None):
        raise ValueError("provide exactly one of count or positions_percent")

    if direction is SplitDirection.VERTICAL:
        low, high = chamber.z_min, chamber.z_max
    else:
        low, high = chamber.r_min, chamber.r_max
    span = high - low

    if count is not None:
        if isinstance(count, bool) or not isinstance(count, int) or count < 2:
            raise ValueError("count must be an integer of at least 2")
        remaining = span - (count - 1) * split_ins
        if remaining <= 0:
            raise ValueError("split insulation leaves no positive child span")
        child_span = remaining / count
        dividers = [low + child_span * index + split_ins * (index - 0.5)
                    for index in range(1, count)]
        percentages = tuple((divider - low) / span * 100.0 for divider in dividers)
    else:
        percentages = tuple(
            float(value) for value in (() if positions_percent is None else positions_percent)
        )
        if not percentages:
            raise ValueError("positions_percent must contain at least one divider")
        if not all(isfinite(value) and 0.0 < value < 100.0 for value in percentages):
            raise ValueError("split positions must be finite percentages strictly between 0 and 100")
        if any(left >= right for left, right in pairwise(percentages)):
            raise ValueError("split positions must be strictly increasing")
        dividers = [low + span * value / 100.0 for value in percentages]

    intervals = [(low, dividers[0] - split_ins / 2.0)]
    intervals.extend(
        (left + split_ins / 2.0, right - split_ins / 2.0)
        for left, right in pairwise(dividers)
    )
    intervals.append((dividers[-1] + split_ins / 2.0, high))
    if any(start >= end for start, end in intervals):
        raise ValueError("split positions and insulation must leave positive child dimensions")

    children = []
    for index, (start, end) in enumerate(intervals):
        bounds = (
            (chamber.r_min, chamber.r_max, start, end)
            if direction is SplitDirection.VERTICAL
            else (start, end, chamber.z_min, chamber.z_max)
        )
        children.append(Chamber(
            name=f"{chamber.name}.{index + 1}",
            r_min=bounds[0],
            r_max=bounds[1],
            z_min=bounds[2],
            z_max=bounds[3],
            parent=chamber.name,
        ))

    chamber.split_direction = direction
    chamber.split_positions_percent = percentages
    chamber.split_insulation_m = split_ins
    chamber.children = children
    return tuple(children)


def initial_chamber(
    core: CoreSpec,
    insulation: InsulationSpec,
    *,
    name: str = "winding_window",
) -> Chamber:
    """Create the usable placement chamber inside core-to-winding clearances."""
    r_min = core.window_inner_radius + insulation.c2w_left
    r_max = core.window_outer_radius - insulation.c2w_right
    z_min = -core.window_half_height + insulation.c2w_bottom
    z_max = core.window_half_height - insulation.c2w_top
    if r_min >= r_max or z_min >= z_max:
        raise ValueError("core-to-winding insulation leaves no positive placement chamber")
    return Chamber(name, r_min, r_max, z_min, z_max)


def core_for_window(
    *,
    core_area: float,
    window_outer_radius: float,
    window_half_height: float,
    name: str = "core",
    material: str = "core",
) -> CoreSpec:
    """Size the existing equal-area core around a caller-defined winding window.

    The centre-leg radius is derived from ``core_area``. The outer-leg radial
    section and yokes retain the current equal-area construction convention.
    All dimensions are in metres and the area is in square metres.
    """
    if not isfinite(core_area) or core_area <= 0.0:
        raise ValueError("core_area must be finite and positive")
    if not isfinite(window_outer_radius) or window_outer_radius <= 0.0:
        raise ValueError("window_outer_radius must be finite and positive")
    if not isfinite(window_half_height) or window_half_height <= 0.0:
        raise ValueError("window_half_height must be finite and positive")
    center_radius = sqrt(core_area / pi)
    if window_outer_radius <= center_radius:
        raise ValueError("window_outer_radius must exceed the center-leg radius")
    yoke_thickness = core_area / (2.0 * pi * center_radius)
    return CoreSpec(
        name=name,
        material=material,
        outer_radius=sqrt(window_outer_radius**2 + core_area / pi),
        outer_half_height=window_half_height + yoke_thickness,
        window_inner_radius=center_radius,
        window_outer_radius=window_outer_radius,
        window_half_height=window_half_height,
        core_type="Single",
    )


def _leaf_chambers(root: Chamber) -> dict[str, Chamber]:
    leaves: dict[str, Chamber] = {}
    seen: set[str] = set()

    def visit(chamber: Chamber, expected_parent: str | None) -> None:
        if chamber.name in seen:
            raise ValueError(f"duplicate chamber name: {chamber.name!r}")
        if chamber.parent != expected_parent:
            raise ValueError(f"chamber {chamber.name!r} has an inconsistent parent")
        seen.add(chamber.name)
        if not chamber.children:
            leaves[chamber.name] = chamber
            return
        for child in chamber.children:
            if (child.r_min < chamber.r_min or child.r_max > chamber.r_max
                    or child.z_min < chamber.z_min or child.z_max > chamber.z_max):
                raise ValueError(f"child chamber {child.name!r} exceeds its parent bounds")
        for first, second in combinations(chamber.children, 2):
            overlaps_r = min(first.r_max, second.r_max) - max(first.r_min, second.r_min)
            overlaps_z = min(first.z_max, second.z_max) - max(first.z_min, second.z_min)
            if overlaps_r > 1e-12 and overlaps_z > 1e-12:
                raise ValueError("sibling chambers must not overlap")
        for child in chamber.children:
            visit(child, chamber.name)

    visit(root, None)
    return leaves


def _aligned_centers(
    *,
    low: float,
    high: float,
    count: int,
    footprint: float,
    insulation: float,
    location: Align | Distribute,
    axis: str,
) -> list[float]:
    span = high - low
    if count < 1 or footprint <= 0 or span <= 0:
        raise ValueError(f"invalid {axis} placement dimensions")
    minimum_span = count * footprint + (count - 1) * insulation
    if minimum_span > span + 1e-12:
        raise ValueError(f"conductors do not fit chamber in {axis} direction")

    if isinstance(location, Distribute):
        if count == 1:
            return [(low + high) / 2.0]
        spacing = (span - count * footprint) / (count - 1)
        if spacing < insulation - 1e-12:
            raise ValueError(f"distributed conductors violate {axis} insulation")
        pitch = footprint + spacing
        return [low + footprint / 2.0 + index * pitch for index in range(count)]

    envelope = minimum_span
    if axis == "horizontal":
        offset = {
            Align.Left: 0.0,
            Align.Center: (span - envelope) / 2.0,
            Align.Right: span - envelope,
        }[location]
    else:
        offset = {
            Align.Bot: 0.0,
            Align.Center: (span - envelope) / 2.0,
            Align.Top: span - envelope,
        }[location]
    return [low + offset + footprint / 2.0 + index * (footprint + insulation)
            for index in range(count)]


def _assignment_centers(
    assignment: ChamberWindingAssignment,
    winding: WindingDefinition,
    turn_insulation: float,
) -> list[tuple[float, float]]:
    chamber = assignment.chamber
    conductor = winding.conductor
    bundle_width = 2.0 * conductor.radial_half_width
    bundle_height = conductor.height
    insulation = turn_insulation
    physical_count = assignment.physical_conductor_count
    if assignment.layers is None:
        per_layer = floor((chamber.height + insulation + 1e-12) / (bundle_height + insulation))
        if per_layer < 1:
            raise ValueError(
                f"winding {winding.name!r} conductor does not fit chamber "
                f"{chamber.name!r} vertically"
            )
        layer_count = ceil(physical_count / per_layer)
        layer_turn_counts = [min(per_layer, physical_count - index * per_layer)
                             for index in range(layer_count)]
    else:
        layer_count = assignment.layers
        if layer_count > physical_count:
            raise ValueError("assignment layers cannot exceed physical conductor count")
        turns_per_layer, extra_turns = divmod(physical_count, layer_count)
        layer_turn_counts = [
            turns_per_layer + (1 if index < extra_turns else 0)
            for index in range(layer_count)
        ]
    horizontal_centers = _aligned_centers(
        low=chamber.r_min,
        high=chamber.r_max,
        count=layer_count,
        footprint=bundle_width,
        insulation=insulation,
        location=assignment.h_loc,
        axis="horizontal",
    )

    placements: list[tuple[float, float]] = []
    for layer_index, horizontal_center in enumerate(horizontal_centers):
        layer_count_here = layer_turn_counts[layer_index]
        vertical_centers = _aligned_centers(
            low=chamber.z_min,
            high=chamber.z_max,
            count=layer_count_here,
            footprint=bundle_height,
            insulation=insulation,
            location=assignment.v_loc,
            axis="vertical",
        )
        ascending = (assignment.start is WindingStart.Bottom) == (
            not (assignment.serpentine and layer_index % 2)
        )
        if not ascending:
            vertical_centers.reverse()
        placements.extend(
            (horizontal_center, vertical_center) for vertical_center in vertical_centers
        )
    return placements


def _air_gap_regions(air_gaps: Sequence[AirGapSpec], core: CoreSpec) -> list[Region]:
    regions = []
    bounds = []
    for gap in air_gaps:
        radial_min = gap.center_r - gap.radial_width / 2.0
        radial_max = gap.center_r + gap.radial_width / 2.0
        axial_min = gap.center_z - gap.axial_length / 2.0
        axial_max = gap.center_z + gap.axial_length / 2.0
        if (radial_min < -1e-12 or radial_max > core.window_inner_radius + 1e-12
                or axial_min < -core.window_half_height - 1e-12
                or axial_max > core.window_half_height + 1e-12):
            raise ValueError(f"air gap {gap.name!r} must fit inside the winding window")
        for previous_gap, previous_bounds in bounds:
            previous_radial_min, previous_radial_max, previous_axial_min, previous_axial_max = (
                previous_bounds
            )
            radial_overlap = min(radial_max, previous_radial_max) - max(
                radial_min, previous_radial_min
            )
            axial_overlap = min(axial_max, previous_axial_max) - max(
                axial_min, previous_axial_min
            )
            if radial_overlap > 1e-12 and axial_overlap > 1e-12:
                raise ValueError(
                    f"air gaps {previous_gap.name!r} and {gap.name!r} overlap"
                )
        bounds.append((gap, (radial_min, radial_max, axial_min, axial_max)))
        regions.append(gap.to_region())
    return regions


def build_chambered_model(
    *,
    core: CoreSpec,
    windings: Sequence[WindingDefinition],
    root_chamber: Chamber,
    assignments: Sequence[ChamberWindingAssignment],
    insulation: InsulationSpec,
    air_gaps: Sequence[AirGapSpec] = (),
    name: str = "chambered magnetic component",
) -> FEMModel:
    """Compile chamber assignments to explicit physical conductor placements."""
    if not windings or not assignments:
        raise ValueError("windings and chamber assignments must be non-empty")
    insulation.validate_winding_count(len(windings))
    winding_by_name = {winding.name: winding for winding in windings}
    if len(winding_by_name) != len(windings):
        raise ValueError("winding names must be unique")
    circuit_names = [winding.circuit_name for winding in windings]
    if len(set(circuit_names)) != len(circuit_names):
        raise ValueError("each winding must use a unique circuit")

    usable = initial_chamber(core, insulation, name=root_chamber.name)
    if (root_chamber.r_min < usable.r_min - 1e-12
            or root_chamber.r_max > usable.r_max + 1e-12
            or root_chamber.z_min < usable.z_min - 1e-12
            or root_chamber.z_max > usable.z_max + 1e-12):
        raise ValueError("root chamber exceeds core-to-winding insulation clearances")
    leaves = _leaf_chambers(root_chamber)

    assignments_by_chamber: dict[str, ChamberWindingAssignment] = {}
    conductor_counts_by_winding = {name: 0 for name in winding_by_name}
    paths_by_winding: dict[str, int] = {}
    for assignment in assignments:
        leaf = leaves.get(assignment.chamber.name)
        if leaf is None or leaf.bounds != assignment.chamber.bounds:
            raise ValueError(
                f"assignment must reference a leaf chamber: {assignment.chamber.name!r}"
            )
        if assignment.chamber.name in assignments_by_chamber:
            raise ValueError(f"chamber {assignment.chamber.name!r} has multiple assignments")
        if assignment.winding not in winding_by_name:
            raise ValueError(f"assignment references unknown winding {assignment.winding!r}")
        assignments_by_chamber[assignment.chamber.name] = assignment
        conductor_counts_by_winding[assignment.winding] += (
            assignment.physical_conductor_count
        )
        paths = paths_by_winding.setdefault(assignment.winding, assignment.parallel_paths)
        if paths != assignment.parallel_paths:
            raise ValueError(
                f"parallel_paths must match across assignments for winding {assignment.winding!r}"
            )

    unassigned_chambers = set(leaves).difference(assignments_by_chamber)
    if unassigned_chambers:
        raise ValueError(f"leaf chambers need winding assignments: {sorted(unassigned_chambers)}")

    for winding in windings:
        paths = paths_by_winding.get(winding.name)
        expected = winding.turns * paths if paths is not None else -1
        assigned = conductor_counts_by_winding[winding.name]
        if assigned != expected:
            raise ValueError(
                f"winding {winding.name!r} has {assigned} assigned conductors; expected {expected}"
            )

    conductor_by_name: dict[str, ConductorSpec] = {}
    model_windings = []
    turn_insulation_by_winding = {
        winding.name: (
            winding.turn_insulation
            if winding.turn_insulation is not None
            else insulation.winding_clearance(index, index)
        )
        for index, winding in enumerate(windings)
    }
    for winding in windings:
        paths = paths_by_winding[winding.name]
        turn_placements: list[TurnPlacement] = []
        next_path = 0
        turns_by_path = [0] * paths
        for assignment in assignments:
            if assignment.winding != winding.name:
                continue
            chamber = leaves[assignment.chamber.name]
            centers = _assignment_centers(
                assignment, winding, turn_insulation_by_winding[winding.name]
            )
            for physical_index, (radius, z) in enumerate(centers):
                path_index = (next_path + physical_index) % paths
                turn_index = turns_by_path[path_index]
                turns_by_path[path_index] += 1
                turn_placements.append(TurnPlacement(
                    center=[round(radius, 15), round(z, 15)],
                    conductor=winding.conductor.name,
                    circuit=winding.circuit_name,
                    parallel_path=path_index,
                    current_fraction=1.0 / paths,
                    metadata={
                        "chamber": chamber.name,
                        "turn_index": turn_index,
                    },
                ))
            next_path = (next_path + len(centers)) % paths
            existing = conductor_by_name.get(winding.conductor.name)
            if existing is not None and existing != winding.conductor:
                raise ValueError(
                    f"conductor name {winding.conductor.name!r} has conflicting definitions"
                )
            conductor_by_name[winding.conductor.name] = winding.conductor
        if turns_by_path != [winding.turns] * paths:
            raise ValueError(
                f"winding {winding.name!r} must have {winding.turns} conductors on each path"
            )
        model_windings.append(Winding(
            name=winding.name,
            turns=turn_placements,
            conductor=winding.conductor.name,
            circuit=winding.circuit_name,
            parallel_paths=paths,
        ))

    core_region, window_region = core.regions()
    core_region.extra["insulation_m"] = {
        "c2w_left": insulation.c2w_left,
        "c2w_right": insulation.c2w_right,
        "c2w_top": insulation.c2w_top,
        "c2w_bottom": insulation.c2w_bottom,
    }
    geometry = Geometry(
        core=core_region,
        air_gaps=_air_gap_regions(air_gaps, core),
        winding_window=window_region,
        extra={
            "chamber_layout": {
                "root": root_chamber.to_dict(),
                "windings": [
                    {
                        "name": winding.name,
                        "conductor": winding.conductor.name,
                        "turns": winding.turns,
                        "circuit": winding.circuit_name,
                        "current": winding.current,
                        "turn_insulation_m": turn_insulation_by_winding[winding.name],
                    }
                    for winding in windings
                ],
                "assignments": [assignment.to_dict() for assignment in assignments],
            }
        },
    )
    return FEMModel(
        metadata=Metadata(name=name),
        geometry=geometry,
        conductors=[conductor.to_model() for conductor in conductor_by_name.values()],
        windings=model_windings,
        circuits=[
            Circuit(winding.circuit_name, turns=winding.turns, current=winding.current)
            for winding in windings
        ],
    )