"""Physical material-volume calculations for axisymmetric components."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import Conductor, FEMModel


@dataclass(frozen=True)
class ComponentVolumeSummary:
    """Physical material volumes in cubic metres."""

    core_material_volume_m3: float
    copper_volume_m3: float
    copper_volume_by_winding_m3: dict[str, float]


def _bounds(corners: list[list[float]], name: str) -> tuple[float, float, float, float]:
    if not corners:
        raise ValueError(f"{name} requires corners")
    radii = [float(point[0]) for point in corners]
    heights = [float(point[1]) for point in corners]
    if not all(isfinite(value) for value in (*radii, *heights)):
        raise ValueError(f"{name} corners must be finite")
    bounds = min(radii), max(radii), min(heights), max(heights)
    if bounds[0] < 0.0 or bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError(f"{name} must have positive radial and axial extent at r >= 0")
    return bounds


def _axisymmetric_rectangle_volume(
    bounds: tuple[float, float, float, float],
) -> float:
    radial_min, radial_max, axial_min, axial_max = bounds
    return pi * (radial_max**2 - radial_min**2) * (axial_max - axial_min)


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    radial_overlap = min(first[1], second[1]) - max(first[0], second[0])
    axial_overlap = min(first[3], second[3]) - max(first[2], second[2])
    return max(0.0, radial_overlap) * max(0.0, axial_overlap)


def _core_material_volume_m3(component: FEMModel) -> float:
    geometry = component.geometry
    core = geometry.core
    window = geometry.winding_window
    if core is None or window is None:
        raise ValueError("geometry requires core and winding_window")
    outer_bounds = _bounds(core.corners, "core")
    window_bounds = _bounds(window.corners, "winding_window")
    if (
        window_bounds[0] < outer_bounds[0]
        or window_bounds[1] > outer_bounds[1]
        or window_bounds[2] < outer_bounds[2]
        or window_bounds[3] > outer_bounds[3]
    ):
        raise ValueError("winding_window must be contained in the core bounds")

    volume = _axisymmetric_rectangle_volume(outer_bounds)
    volume -= _axisymmetric_rectangle_volume(window_bounds)
    gap_bounds = []
    for gap in geometry.air_gaps:
        if gap.center is None or len(gap.center) < 2:
            raise ValueError(f"air gap {gap.name!r} requires an (r, z) center")
        if gap.size is None or len(gap.size) < 2:
            raise ValueError(f"air gap {gap.name!r} requires radial width and axial length")
        center_r, center_z = map(float, gap.center[:2])
        radial_width, axial_length = map(float, gap.size[:2])
        if not all(isfinite(value) for value in (center_r, center_z, radial_width, axial_length)):
            raise ValueError(f"air gap {gap.name!r} dimensions must be finite")
        if axial_length <= 0.0:
            raise ValueError(f"air gap {gap.name!r} axial length must be positive")
        radial_min = (
            outer_bounds[0]
            if radial_width <= 0.0
            else center_r - radial_width / 2.0
        )
        radial_max = (
            window_bounds[0]
            if radial_width <= 0.0
            else center_r + radial_width / 2.0
        )
        bounds = (
            radial_min,
            radial_max,
            center_z - axial_length / 2.0,
            center_z + axial_length / 2.0,
        )
        if (
            bounds[0] < outer_bounds[0]
            or bounds[1] > outer_bounds[1]
            or bounds[2] < outer_bounds[2]
            or bounds[3] > outer_bounds[3]
            or bounds[0] >= bounds[1]
        ):
            raise ValueError(f"air gap {gap.name!r} must fit inside the core bounds")
        if _intersection_area(bounds, window_bounds) > 0.0:
            raise ValueError(f"air gap {gap.name!r} overlaps the winding window")
        if any(_intersection_area(bounds, previous) > 0.0 for previous in gap_bounds):
            raise ValueError("air gaps must not overlap when calculating core material volume")
        gap_bounds.append(bounds)
        volume -= _axisymmetric_rectangle_volume(bounds)

    if not isfinite(volume) or volume <= 0.0:
        raise ValueError("core material volume must be finite and positive")
    return volume


def _copper_cross_section_m2(conductor: Conductor) -> float:
    if conductor.construction not in {"solid", "litz"}:
        raise ValueError(f"unsupported conductor construction: {conductor.construction!r}")
    if conductor.shape in {"round", "circle"}:
        if conductor.radius is None or not isfinite(conductor.radius) or conductor.radius <= 0.0:
            raise ValueError(f"round conductor {conductor.name!r} requires a positive radius")
        bundle_area_m2 = pi * conductor.radius**2
        if conductor.construction == "litz":
            if (
                conductor.strands is None
                or isinstance(conductor.strands, bool)
                or not isinstance(conductor.strands, int)
                or conductor.strands < 1
                or conductor.strand_diameter is None
                or not isfinite(conductor.strand_diameter)
                or conductor.strand_diameter <= 0.0
            ):
                raise ValueError(f"Litz conductor {conductor.name!r} has invalid strand data")
            copper_area_m2 = (
                conductor.strands * pi * (conductor.strand_diameter / 2.0) ** 2
            )
            if copper_area_m2 > bundle_area_m2 * (1.0 + 1e-12):
                raise ValueError(f"Litz conductor {conductor.name!r} has fill factor above 1")
            return copper_area_m2
        return bundle_area_m2
    if conductor.shape in {"rectangular", "rectangle"}:
        if (
            conductor.width is None
            or conductor.height is None
            or not isfinite(conductor.width)
            or not isfinite(conductor.height)
            or conductor.width <= 0.0
            or conductor.height <= 0.0
        ):
            raise ValueError(
                f"rectangular conductor {conductor.name!r} requires positive dimensions"
            )
        if conductor.construction == "litz":
            raise ValueError("Litz conductor construction requires round conductor geometry")
        return conductor.width * conductor.height
    raise ValueError(f"unsupported conductor shape: {conductor.shape!r}")


def physical_volume_summary(component: FEMModel) -> ComponentVolumeSummary:
    """Calculate core-material and copper volumes for an axisymmetric component.

    Core volume is the outer axisymmetric envelope minus the winding window and
    air gaps. Conductor volume uses each physical placement's radial centroid;
    Litz copper area is derived from its strand count and strand diameter.
    """
    if component.geometry.coordinate_system != "axisymmetric_rz":
        raise ValueError(
            "physical volume calculations require coordinate_system='axisymmetric_rz'"
        )

    core_volume_m3 = _core_material_volume_m3(component)
    conductors = {conductor.name: conductor for conductor in component.conductors}
    copper_volume_by_winding_m3 = {}
    for winding in component.windings:
        if winding.name in copper_volume_by_winding_m3:
            raise ValueError(f"duplicate winding name: {winding.name!r}")
        winding_volume_m3 = 0.0
        for turn in winding.turns:
            conductor_name = turn.conductor or winding.conductor
            conductor = conductors.get(conductor_name)
            if conductor is None:
                raise ValueError(f"unknown conductor {conductor_name!r}")
            if conductor.material.casefold() != "copper":
                continue
            if len(turn.center) < 2:
                raise ValueError("turn center must contain radial and axial coordinates")
            radius_m, height_m = map(float, turn.center[:2])
            if not isfinite(radius_m) or not isfinite(height_m) or radius_m <= 0.0:
                raise ValueError("turn centers must be finite and have positive radius")
            cross_section_m2 = _copper_cross_section_m2(conductor)
            if conductor.shape in {"round", "circle"}:
                if conductor.radius is None:
                    raise ValueError(f"round conductor {conductor.name!r} requires a radius")
                radial_half_width_m = conductor.radius
            else:
                if conductor.width is None:
                    raise ValueError(
                        f"rectangular conductor {conductor.name!r} requires a width"
                    )
                radial_half_width_m = conductor.width / 2.0
            if radius_m < radial_half_width_m:
                raise ValueError("conductor placements must not cross the axis of revolution")
            winding_volume_m3 += 2.0 * pi * radius_m * cross_section_m2
        copper_volume_by_winding_m3[winding.name] = winding_volume_m3

    copper_volume_m3 = sum(copper_volume_by_winding_m3.values())
    return ComponentVolumeSummary(
        core_material_volume_m3=core_volume_m3,
        copper_volume_m3=copper_volume_m3,
        copper_volume_by_winding_m3=copper_volume_by_winding_m3,
    )
