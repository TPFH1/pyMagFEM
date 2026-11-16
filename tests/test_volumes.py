from dataclasses import asdict
from math import pi

import pytest

from model import (
    AirGapSpec,
    ComponentVolumeSummary,
    Conductor,
    FEMModel,
    Geometry,
    Region,
    TurnPlacement,
    Winding,
    normalize,
    physical_volume_summary,
)


def component_with_core(*, air_gaps=(), conductors=(), windings=()):
    return FEMModel(
        geometry=Geometry(
            core=Region(
                "core",
                "ferrite",
                [[0.0, -2.0], [4.0, -2.0], [0.0, 2.0], [4.0, 2.0]],
            ),
            winding_window=Region(
                "winding_window",
                corners=[[1.0, -1.0], [3.0, -1.0], [1.0, 1.0], [3.0, 1.0]],
            ),
            air_gaps=list(air_gaps),
        ),
        conductors=list(conductors),
        windings=list(windings),
    )


def test_core_material_volume_excludes_window_and_air_gaps():
    component = component_with_core(
        air_gaps=(AirGapSpec("center_gap", 0.5, 0.0, 1.0, 0.2).to_region(),)
    )

    summary = component.physical_volume_summary()

    assert summary.core_material_volume_m3 == pytest.approx((64.0 - 16.0 - 0.2) * pi)
    assert summary.copper_volume_m3 == 0.0
    assert isinstance(summary, ComponentVolumeSummary)
    assert physical_volume_summary(component) == summary


def test_copper_volume_uses_turn_radius_and_litz_strand_area():
    solid = Conductor("solid", radius=0.1)
    litz = Conductor(
        "litz",
        construction="litz",
        radius=0.2,
        strands=4,
        strand_diameter=0.1,
    )
    aluminum = Conductor("aluminum", radius=0.1, material="aluminum")
    component = component_with_core(
        conductors=(solid, litz, aluminum),
        windings=(
            Winding("primary", [TurnPlacement([1.5, 0.0], conductor="solid")]),
            Winding("secondary", [
                TurnPlacement([2.0, 0.0], conductor="litz"),
                TurnPlacement([2.5, 0.0], conductor="aluminum"),
            ]),
        ),
    )

    summary = component.physical_volume_summary()

    expected_solid_m3 = 2.0 * pi * 1.5 * (pi * 0.1**2)
    expected_litz_m3 = 2.0 * pi * 2.0 * (4 * pi * (0.1 / 2.0) ** 2)
    assert summary.copper_volume_by_winding_m3 == pytest.approx({
        "primary": expected_solid_m3,
        "secondary": expected_litz_m3,
    })
    assert summary.copper_volume_m3 == pytest.approx(expected_solid_m3 + expected_litz_m3)


def test_physical_volume_summary_rejects_overlapping_air_gaps():
    component = component_with_core(air_gaps=(
        AirGapSpec("gap_1", 0.5, 0.0, 1.0, 0.5).to_region(),
        AirGapSpec("gap_2", 0.5, 0.1, 1.0, 0.5).to_region(),
    ))

    with pytest.raises(ValueError, match="air gaps must not overlap"):
        component.physical_volume_summary()


def test_physical_volume_summary_survives_model_serialization():
    component = component_with_core(
        conductors=(Conductor("copper", radius=0.1),),
        windings=(Winding(
            "primary",
            [TurnPlacement([1.5, 0.0], conductor="copper")],
        ),),
    )

    restored = normalize(asdict(component))

    assert restored.physical_volume_summary() == component.physical_volume_summary()


def test_physical_volume_summary_requires_axisymmetric_geometry():
    component = component_with_core()
    component.geometry.coordinate_system = "cartesian"

    with pytest.raises(ValueError, match="axisymmetric_rz"):
        component.physical_volume_summary()
