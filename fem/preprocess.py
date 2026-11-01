"""Mesh generation for 2-D axisymmetric magnetostatic FEM models.

The mesh is generated with Gmsh through pygmsh and uses explicit OCC material
interfaces for the ferrite, air gap, winding window, and conductors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from model import FEMModel, load


@dataclass
class AxisymmetricMesh:
    """Triangular meridian-plane mesh and physical tags."""

    points: np.ndarray
    triangles: np.ndarray
    regions: np.ndarray
    boundaries: dict[str, np.ndarray]
    geometry: dict[str, Any]
    conductor_regions: dict[str, np.ndarray] | None = None
    materials: dict[str, object] | None = None
    _cache: dict[str, object] = field(default_factory=dict, repr=False)

    @property
    def nodes(self) -> np.ndarray:
        return self.points

    @property
    def elements(self) -> np.ndarray:
        return self.triangles

    def region_areas(self) -> dict[str, float]:
        """Return meridian-plane area represented by each physical region."""
        areas = {name: 0.0 for name in ("core", "air", "winding")}
        for nodes, region in zip(self.triangles, self.regions):
            xy = self.points[nodes]
            area = abs(np.linalg.det(np.array([
                [xy[1, 0] - xy[0, 0], xy[2, 0] - xy[0, 0]],
                [xy[1, 1] - xy[0, 1], xy[2, 1] - xy[0, 1]],
            ]))) / 2.0
            areas[str(region)] += area
        return areas


def load_geometry(source: str | Path | dict[str, Any] | FEMModel) -> dict[str, Any]:
    """Load JSON or a model object and return the flat FEM geometry mapping."""
    return source.legacy_geometry() if isinstance(source, FEMModel) else load(source).legacy_geometry()


def _geometry_data(model: FEMModel) -> dict[str, Any]:
    """Convert a normalized model at the FEM preprocessing boundary."""
    if not isinstance(model, FEMModel):
        raise TypeError("build_mesh expects a normalized FEMModel; use mesh_from_json for JSON")
    return model.legacy_geometry()


def _bounds(corners: Any) -> tuple[float, float, float, float]:
    a = np.asarray(corners, dtype=float)
    return float(a[:, 0].min()), float(a[:, 0].max()), float(a[:, 1].min()), float(a[:, 1].max())


def _conductor_dimensions(g: dict[str, Any]) -> tuple[list[str], list[float], list[float], list[float]]:
    count = len(g["p_cond_center_1"])
    shapes = g.get("conductor_shapes", ["round"] * count)
    widths = g.get("conductor_widths", [0.0] * count)
    heights = g.get("conductor_heights", [0.0] * count)
    radii = g.get("conductor_radii", [0.0] * count)
    if not all(len(values) == count for values in (shapes, widths, heights, radii)):
        raise ValueError("conductor geometry arrays must have one entry per conductor")
    for shape, width, height, radius in zip(shapes, widths, heights, radii):
        if shape == "rectangular" and (float(width) <= 0.0 or float(height) <= 0.0):
            raise ValueError("rectangular conductors require positive width and height")
        if shape == "round" and float(radius) <= 0.0:
            raise ValueError("round conductors require a positive radius")
    return shapes, widths, heights, radii


def _rectangle_points(bounds: tuple[float, float, float, float], h: float) -> np.ndarray:
    r0, r1, z0, z1 = bounds
    nr = max(2, int(np.ceil((r1 - r0) / h)))
    nz = max(2, int(np.ceil((z1 - z0) / h)))
    rr = np.linspace(r0, r1, nr + 1)
    zz = np.linspace(z0, z1, nz + 1)
    return np.array([(r, z) for z in zz for r in rr])


def _classify(points: np.ndarray, triangles: np.ndarray, g: dict[str, Any]) -> np.ndarray:
    """Assign solver material names from element centroids."""
    outer = _bounds(g["p_outer"])
    window = _bounds(g["p_ww"])
    cent = points[triangles].mean(axis=1)
    regions = np.full(len(triangles), "core", dtype="U8")
    regions[((cent[:, 0] >= window[0]) & (cent[:, 0] <= window[1]) &
             (cent[:, 1] >= window[2]) & (cent[:, 1] <= window[3]))] = "air"
    widths = g.get("widths_air_gap", [0.0] * len(g["p_air_gap_center"]))
    for center, width, length in zip(g["p_air_gap_center"], widths, g["lengths_air_gap"]):
        radial_min = outer[0] if float(width) <= 0.0 else float(center[0]) - float(width) / 2.0
        radial_max = window[0] if float(width) <= 0.0 else float(center[0]) + float(width) / 2.0
        regions[(np.abs(cent[:, 1] - float(center[1])) <= float(length) / 2) &
                (cent[:, 0] >= radial_min) & (cent[:, 0] <= radial_max)] = "air"
    shapes, widths, heights, radii = _conductor_dimensions(g)
    for center, shape, width, height, radius in zip(
        g["p_cond_center_1"], shapes, widths, heights, radii
    ):
        c = np.asarray(center, dtype=float)
        if shape == "rectangular":
            inside = (
                (np.abs(cent[:, 0] - c[0]) <= float(width) / 2.0)
                & (np.abs(cent[:, 1] - c[1]) <= float(height) / 2.0)
            )
        else:
            inside = (cent[:, 0] - c[0]) ** 2 + (cent[:, 1] - c[1]) ** 2 <= float(radius) ** 2
        regions[inside] = "winding"
    return regions


def _boundary_nodes(points: np.ndarray, triangles: np.ndarray, outer: tuple[float, float, float, float]) -> dict[str, np.ndarray]:
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1].ravel()
    r0 = boundary[np.isclose(points[boundary, 0], outer[0], atol=1e-10)]
    outer_nodes = boundary[
        np.isclose(points[boundary, 0], outer[1], atol=1e-10)
        | np.isclose(points[boundary, 1], outer[2], atol=1e-10)
        | np.isclose(points[boundary, 1], outer[3], atol=1e-10)
    ]
    return {"r0": np.unique(r0), "outer": np.unique(outer_nodes)}


def _build_pygmsh(
    g: dict[str, Any],
    h: float,
    conductor_size: float | None = None,
    conductor_geometry_size: float | None = None,
    physics_adaptive: bool = False,
) -> AxisymmetricMesh:
    """Build a conforming OCC/pygmsh mesh from explicit material surfaces."""
    import gmsh
    import pygmsh

    outer = _bounds(g["p_outer"])
    window = _bounds(g["p_ww"])
    with pygmsh.occ.Geometry() as geom:
        skin_size = 2.0 * conductor_size if conductor_size is not None else h
        geometry_size = (
            conductor_geometry_size if conductor_geometry_size is not None else h
        )
        # Skin refinement is applied by its local distance field below.  It
        # must not lower the global characteristic length in core or air.
        geom.characteristic_length_min = min(h, geometry_size)
        geom.characteristic_length_max = (
            2.0 * h
            if conductor_size is not None or conductor_geometry_size is not None
            else h
        )
        outer_surface = geom.add_rectangle(
            [outer[0], outer[2], 0.0], outer[1] - outer[0], outer[3] - outer[2]
        )
        window_surface = geom.add_rectangle(
            [window[0], window[2], 0.0], window[1] - window[0], window[3] - window[2]
        )
        widths = g.get("widths_air_gap", [0.0] * len(g["p_air_gap_center"]))
        gap_surfaces = [
            geom.add_rectangle(
                [
                    outer[0] if float(width) <= 0.0 else float(c[0]) - float(width) / 2.0,
                    float(c[1]) - float(length) / 2,
                    0.0,
                ],
                window[0] - outer[0] if float(width) <= 0.0 else float(width),
                float(length),
            )
            for c, width, length in zip(
                g["p_air_gap_center"], widths, g["lengths_air_gap"]
            )
        ]
        shapes, widths, heights, radii = _conductor_dimensions(g)
        windings = []
        for c, shape, width, height, radius in zip(
            g["p_cond_center_1"], shapes, widths, heights, radii
        ):
            if shape == "rectangular":
                windings.append(
                    geom.add_rectangle(
                        [float(c[0]) - float(width) / 2.0, float(c[1]) - float(height) / 2.0, 0.0],
                        float(width),
                        float(height),
                    )
                )
            else:
                windings.append(geom.add_disk([float(c[0]), float(c[1]), 0.0], float(radius)))
        # Fragment the complete set in one operation.  This removes each
        # source surface and leaves one non-overlapping OCC surface per region.
        geom.boolean_fragments(outer_surface, [window_surface, *gap_surfaces, *windings])
        if conductor_size is not None and conductor_size <= 0:
            raise ValueError("conductor_size must be positive")
        if conductor_geometry_size is not None and conductor_geometry_size <= 0:
            raise ValueError("conductor_geometry_size must be positive")
        if (
            physics_adaptive
            or conductor_size is not None
            or conductor_geometry_size is not None
        ):
            centers = np.asarray(g["p_cond_center_1"], dtype=float)
            _, _, _, radii = _conductor_dimensions(g)

            geom.synchronize()
            conductor_curves: list[int] = []
            for _, surface_tag in gmsh.model.getEntities(2):
                x, y, _ = gmsh.model.occ.getCenterOfMass(2, surface_tag)
                inside_conductor = False
                for center, shape, width, height, radius in zip(
                    centers,
                    g.get("conductor_shapes", ["round"] * len(centers)),
                    g.get("conductor_widths", [0.0] * len(centers)),
                    g.get("conductor_heights", [0.0] * len(centers)),
                    radii,
                ):
                    if shape == "rectangular":
                        inside_conductor |= (
                            abs(center[0] - x) <= float(width) / 2.0
                            and abs(center[1] - y) <= float(height) / 2.0
                        )
                    else:
                        inside_conductor |= (
                            (center[0] - x) ** 2 + (center[1] - y) ** 2
                            <= float(radius) ** 2
                        )
                if inside_conductor:
                    boundary = gmsh.model.getBoundary(
                        [(2, surface_tag)], combined=False, oriented=False, recursive=False
                    )
                    conductor_curves.extend(tag for dim, tag in boundary if dim == 1)
            fields: list[int] = []
            if conductor_curves and conductor_geometry_size is not None:
                geometry_distance = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(
                    geometry_distance,
                    "CurvesList",
                    sorted(set(conductor_curves)),
                )
                geometry_threshold = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(
                    geometry_threshold, "InField", geometry_distance
                )
                gmsh.model.mesh.field.setNumber(
                    geometry_threshold, "SizeMin", conductor_geometry_size
                )
                gmsh.model.mesh.field.setNumber(
                    geometry_threshold, "SizeMax", 2.0 * h
                )
                gmsh.model.mesh.field.setNumber(
                    geometry_threshold, "DistMin", 0.0
                )
                gmsh.model.mesh.field.setNumber(
                    geometry_threshold,
                    "DistMax",
                    max(max(radii), 3.0 * h),
                )
                fields.append(geometry_threshold)

            if conductor_curves and conductor_size is not None:
                distance = gmsh.model.mesh.field.add("Distance")
                gmsh.model.mesh.field.setNumbers(
                    distance, "CurvesList", sorted(set(conductor_curves))
                )
                threshold = gmsh.model.mesh.field.add("Threshold")
                gmsh.model.mesh.field.setNumber(threshold, "InField", distance)
                gmsh.model.mesh.field.setNumber(threshold, "SizeMin", skin_size)
                gmsh.model.mesh.field.setNumber(threshold, "SizeMax", 2.0 * h)
                gmsh.model.mesh.field.setNumber(threshold, "DistMin", 0.0)
                gmsh.model.mesh.field.setNumber(
                    threshold, "DistMax", max(max(radii), 3.0 * h)
                )
                fields.append(threshold)

            if physics_adaptive:
                # The winding-window corners are the core's inner corners in
                # this geometry and are the main flux-accumulation locations.
                for corner in g["p_ww"]:
                    corner_field = gmsh.model.mesh.field.add("Ball")
                    radius = 1.0 * h
                    gmsh.model.mesh.field.setNumber(
                        corner_field, "XCenter", float(corner[0])
                    )
                    gmsh.model.mesh.field.setNumber(
                        corner_field, "YCenter", float(corner[1])
                    )
                    gmsh.model.mesh.field.setNumber(
                        corner_field, "ZCenter", 0.0
                    )
                    gmsh.model.mesh.field.setNumber(
                        corner_field, "Radius", radius
                    )
                    gmsh.model.mesh.field.setNumber(
                        corner_field, "VIn", 0.25 * h
                    )
                    gmsh.model.mesh.field.setNumber(
                        corner_field, "VOut", h
                    )
                    fields.append(corner_field)

                # Only the window-facing gap corners control the relevant
                # fringing field. The axis-side corners at r=0 are not
                # accuracy-critical and intentionally remain coarse.
                gap_centers = np.asarray(g["p_air_gap_center"], dtype=float)
                gap_widths = g.get(
                    "widths_air_gap", [0.0] * len(gap_centers)
                )
                gap_lengths = g["lengths_air_gap"]
                window_inner_radius = float(_bounds(g["p_ww"])[0])
                for center, width, length in zip(
                    gap_centers, gap_widths, gap_lengths
                ):
                    gap_inner_radius = (
                        window_inner_radius
                        if float(width) <= 0.0
                        else float(center[0]) + float(width) / 2.0
                    )
                    for z in (
                        float(center[1]) - float(length) / 2.0,
                        float(center[1]) + float(length) / 2.0,
                    ):
                        gap_field = gmsh.model.mesh.field.add("Ball")
                        radius = 1.0 * h
                        gmsh.model.mesh.field.setNumber(
                            gap_field, "XCenter", gap_inner_radius
                        )
                        gmsh.model.mesh.field.setNumber(
                            gap_field, "YCenter", z
                        )
                        gmsh.model.mesh.field.setNumber(
                            gap_field, "ZCenter", 0.0
                        )
                        gmsh.model.mesh.field.setNumber(
                            gap_field, "Radius", radius
                        )
                        gmsh.model.mesh.field.setNumber(
                            gap_field, "VIn", 0.25 * h
                        )
                        gmsh.model.mesh.field.setNumber(
                            gap_field, "VOut", h
                        )
                        fields.append(gap_field)

            if fields:
                background = gmsh.model.mesh.field.add("Min")
                gmsh.model.mesh.field.setNumbers(background, "FieldsList", fields)
                gmsh.model.mesh.field.setAsBackgroundMesh(background)
            if conductor_size is not None or conductor_geometry_size is not None:
                conductor_surfaces = []
                for _, surface_tag in gmsh.model.getEntities(2):
                    x, y, _ = gmsh.model.occ.getCenterOfMass(2, surface_tag)
                    if any(
                        (
                            abs(center[0] - x) <= float(width) / 2.0
                            and abs(center[1] - y) <= float(height) / 2.0
                        )
                        if shape == "rectangular"
                        else (center[0] - x) ** 2 + (center[1] - y) ** 2 <= float(radius) ** 2
                        for center, shape, width, height, radius in zip(
                            centers,
                            g.get("conductor_shapes", ["round"] * len(centers)),
                            g.get("conductor_widths", [0.0] * len(centers)),
                            g.get("conductor_heights", [0.0] * len(centers)),
                            radii,
                        )
                    ):
                        conductor_surfaces.append(surface_tag)
                for center in centers:
                    distances = np.linalg.norm(centers - center, axis=1)
                    surface_tag = conductor_surfaces[int(np.argmin(distances))]
                    point_tag = gmsh.model.occ.addPoint(
                        float(center[0]), float(center[1]), 0.0
                    )
                    gmsh.model.occ.synchronize()
                    gmsh.model.mesh.embed(0, [point_tag], 2, surface_tag)
                    gmsh.model.mesh.setSize([(0, point_tag)], h)
        msh = geom.generate_mesh(dim=2)
    points = np.asarray(msh.points[:, :2], dtype=float)
    cells = msh.cells_dict.get("triangle")
    if cells is None:
        raise RuntimeError("pygmsh did not produce triangular cells")
    triangles = np.asarray(cells, dtype=int)
    regions = _classify(points, triangles, g)
    cent = points[triangles].mean(axis=1)
    conductor_regions = {}
    for index, center in enumerate(g["p_cond_center_1"]):
        shape = g.get("conductor_shapes", ["round"] * len(g["p_cond_center_1"]))[index]
        if shape == "rectangular":
            inside = (
                (np.abs(cent[:, 0] - float(center[0])) <=
                 float(g["conductor_widths"][index]) / 2.0)
                & (np.abs(cent[:, 1] - float(center[1])) <=
                   float(g["conductor_heights"][index]) / 2.0)
            )
        else:
            radius = g.get("conductor_radii", [g["radius_cond_1"]] *
                           len(g["p_cond_center_1"]))[index]
            inside = (
                (cent[:, 0] - float(center[0])) ** 2 +
                (cent[:, 1] - float(center[1])) ** 2 <= float(radius) ** 2
            )
        conductor_regions[f"conductor_{index}"] = np.flatnonzero(inside)
    materials = {
        material.name: material
        for material in g.get("materials", [])
    }
    return AxisymmetricMesh(
        points, triangles, regions, _boundary_nodes(points, triangles, outer),
        g, conductor_regions, materials,
    )


def build_mesh(
    model: FEMModel,
    max_size: float | None = None,
    conductor_size: float | None = None,
    physics_adaptive: bool = True,
    conductor_geometry_size: float | None = None,
) -> AxisymmetricMesh:
    """Create a conforming Gmsh/pygmsh mesh from a normalized model object.

    ``conductor_geometry_size`` controls moderate refinement along conductor
    boundaries. ``conductor_size`` adds the finer skin-depth refinement used
    for solid conductors.
    """
    g = _geometry_data(model)
    g["materials"] = model.materials
    outer = _bounds(g["p_outer"])
    span = max(outer[1] - outer[0], outer[3] - outer[2])
    h = max_size or span / 35.0
    if h <= 0:
        raise ValueError("max_size must be positive")
    if conductor_geometry_size is None:
        conductor_geometry_size = h / 2.0
    return _build_pygmsh(
        g,
        h,
        conductor_size=conductor_size,
        conductor_geometry_size=conductor_geometry_size,
        physics_adaptive=physics_adaptive,
    )


def mesh_from_json(
    path: str | Path,
    max_size: float | None = None,
    conductor_size: float | None = None,
    physics_adaptive: bool = True,
    conductor_geometry_size: float | None = None,
) -> AxisymmetricMesh:
    """Load a JSON model and build its mesh."""
    return build_mesh(
        load(path),
        max_size=max_size,
        conductor_size=conductor_size,
        conductor_geometry_size=conductor_geometry_size,
        physics_adaptive=physics_adaptive,
    )
