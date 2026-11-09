from dataclasses import asdict

import pytest

from model import (
    Align,
    AxialStack,
    Chamber,
    ChamberWindingAssignment,
    CoreSpec,
    Distribute,
    InsulationSpec,
    RadialStack,
    RectangularConductor,
    RoundConductor,
    SplitDirection,
    WindingDefinition,
    WindingSection,
    WindingStart,
    build_chambered_model,
    core_for_window,
    initial_chamber,
    normalize,
    split_chamber,
)


def test_equal_vertical_split_reserves_shared_insulation_and_builds_tree():
    parent = Chamber("window", 0.01, 0.03, -0.02, 0.02)

    lower, upper = split_chamber(
        parent,
        split_dir=SplitDirection.VERTICAL,
        count=2,
        split_ins=0.002,
    )

    assert lower.bounds == pytest.approx((0.01, 0.03, -0.02, -0.001))
    assert upper.bounds == pytest.approx((0.01, 0.03, 0.001, 0.02))
    assert lower.parent == upper.parent == "window"
    assert parent.children == [lower, upper]
    assert parent.split_direction is SplitDirection.VERTICAL
    assert parent.split_positions_percent == pytest.approx((50.0,))


def test_explicit_horizontal_split_positions_are_percentages_from_low_edge():
    chamber = Chamber("window", 0.0, 0.1, 0.0, 0.04)

    left, middle, right = split_chamber(
        chamber,
        split_dir="horizontal",
        positions_percent=(25, 75),
        split_ins=0.004,
    )

    expected_bounds = (
        (0.0, 0.023, 0.0, 0.04),
        (0.027, 0.073, 0.0, 0.04),
        (0.077, 0.1, 0.0, 0.04),
    )
    for child, expected in zip((left, middle, right), expected_bounds):
        assert child.bounds == pytest.approx(expected)


@pytest.mark.parametrize("kwargs", [
    {"count": 1},
    {"count": 3, "positions_percent": (50,)},
    {"positions_percent": ()},
    {"positions_percent": (50, 25)},
    {"positions_percent": (0,)},
])
def test_invalid_split_specification_is_rejected(kwargs):
    with pytest.raises(ValueError):
        split_chamber(
            Chamber("window", 0.0, 0.1, 0.0, 0.1),
            split_dir="vertical",
            split_ins=0.0,
            **kwargs,
        )


def test_split_rejects_insulation_that_collapses_a_child():
    with pytest.raises(ValueError, match="positive child"):
        split_chamber(
            Chamber("window", 0.0, 0.1, 0.0, 0.1),
            split_dir="vertical",
            positions_percent=(1, 99),
            split_ins=0.01,
        )


def small_core():
    return CoreSpec(
        "core", "core", outer_radius=0.03, outer_half_height=0.01,
        window_inner_radius=0.01, window_outer_radius=0.02, window_half_height=0.004,
    )


def test_core_for_window_preserves_equal_area_core_sections():
    from math import pi

    core = core_for_window(
        core_area=100e-6,
        window_outer_radius=0.02,
        window_half_height=0.01,
    )
    center_radius = core.window_inner_radius
    assert pi * center_radius**2 == pytest.approx(100e-6)
    assert pi * (core.outer_radius**2 - core.window_outer_radius**2) == pytest.approx(100e-6)
    yoke_height = core.outer_half_height - core.window_half_height
    assert 2.0 * pi * center_radius * yoke_height == pytest.approx(100e-6)


def test_initial_chamber_applies_core_to_winding_insulation_without_changing_window():
    core = small_core()
    insulation = InsulationSpec(
        c2w_left=0.001, c2w_right=0.002, c2w_bottom=0.0015, c2w_top=0.0005,
    )

    chamber = initial_chamber(core, insulation)
    model = build_chambered_model(
        core=core,
        windings=(WindingDefinition("primary", RoundConductor("wire", radius=0.001), 1),),
        root_chamber=chamber,
        assignments=(ChamberWindingAssignment(chamber, "primary", 1),),
        insulation=insulation,
    )

    assert chamber.bounds == pytest.approx((0.011, 0.018, -0.0025, 0.0035))
    assert model.geometry.winding_window.corners == core.regions()[1].corners
    assert model.windings[0].turns[0].center == pytest.approx([0.0145, 0.0005])


def test_turns_can_span_chambers_and_parallel_path_metadata_is_preserved():
    core = small_core()
    insulation = InsulationSpec(c2w_left=0.001, c2w_right=0.001)
    root = initial_chamber(core, insulation)
    left, right = split_chamber(
        root, split_dir="horizontal", count=2, split_ins=0.001,
    )
    model = build_chambered_model(
        core=core,
        windings=(WindingDefinition("primary", RoundConductor("wire", radius=0.001), 3),),
        root_chamber=root,
        assignments=(
            ChamberWindingAssignment(left, "primary", 1, parallel_paths=2),
            ChamberWindingAssignment(right, "primary", 2, parallel_paths=2),
        ),
        insulation=insulation,
    )

    winding = model.windings[0]
    assert len(winding.turns) == 6
    assert winding.parallel_paths == 2
    assert model.circuits[0].turns == 3
    assert [turn.parallel_path for turn in winding.turns] == [0, 1, 0, 1, 0, 1]
    assert [turn.metadata["turn_index"] for turn in winding.turns] == [0, 0, 1, 1, 2, 2]
    assert [turn.metadata["chamber"] for turn in winding.turns] == [
        "winding_window.1", "winding_window.1", "winding_window.2",
        "winding_window.2", "winding_window.2", "winding_window.2",
    ]
    assert model.legacy_geometry()["conductor_current_fractions"] == [0.5] * 6
    restored = normalize(asdict(model))
    assert restored.geometry.extra["chamber_layout"]["assignments"][0]["turns"] == 1


def test_chamber_assignment_rejects_wrong_turn_total_and_mismatched_paths():
    core = small_core()
    insulation = InsulationSpec(
        c2w_left=0.001, c2w_right=0.001, c2w_bottom=0.001, c2w_top=0.001,
    )
    root = initial_chamber(core, insulation)
    left, right = split_chamber(root, split_dir="horizontal", count=2, split_ins=0.001)
    winding = WindingDefinition("primary", RoundConductor("wire", radius=0.001), 3)

    with pytest.raises(ValueError, match="expected 3"):
        build_chambered_model(
            core=core,
            windings=(winding,),
            root_chamber=root,
            assignments=(
                ChamberWindingAssignment(left, "primary", 1),
                ChamberWindingAssignment(right, "primary", 1),
            ),
            insulation=insulation,
        )
    with pytest.raises(ValueError, match="parallel_paths must match"):
        build_chambered_model(
            core=core,
            windings=(winding,),
            root_chamber=root,
            assignments=(
                ChamberWindingAssignment(left, "primary", 1, parallel_paths=1),
                ChamberWindingAssignment(right, "primary", 2, parallel_paths=2),
            ),
            insulation=insulation,
        )


def test_every_leaf_chamber_requires_an_assignment():
    core = small_core()
    insulation = InsulationSpec(c2w_left=0.001, c2w_right=0.001)
    root = initial_chamber(core, insulation)
    left, _right = split_chamber(root, split_dir="horizontal", count=2, split_ins=0.001)
    with pytest.raises(ValueError, match="leaf chambers need winding assignments"):
        build_chambered_model(
            core=core,
            windings=(WindingDefinition("primary", RoundConductor("wire", radius=0.001), 1),),
            root_chamber=root,
            assignments=(ChamberWindingAssignment(left, "primary", 1),),
            insulation=insulation,
        )


def test_serpentine_progression_and_location_modes_generate_bundle_centers():
    core = small_core()
    insulation = InsulationSpec(
        c2w_left=0.001, c2w_right=0.001, c2w_bottom=0.001, c2w_top=0.001,
    )
    chamber = initial_chamber(core, insulation)
    model = build_chambered_model(
        core=core,
        windings=(WindingDefinition("primary", RoundConductor("wire", radius=0.001), 7),),
        root_chamber=chamber,
        assignments=(ChamberWindingAssignment(
            chamber,
            "primary",
            7,
            h_loc=Align.Left,
            v_loc=Align.Bot,
            start=WindingStart.Bottom,
        ),),
        insulation=insulation,
    )

    centers = [turn.center for turn in model.windings[0].turns]
    assert [center[0] for center in centers] == pytest.approx([0.012] * 3 + [0.014] * 3 + [0.016])
    assert [center[1] for center in centers[:3]] == pytest.approx([-.002, 0.0, 0.002])
    assert [center[1] for center in centers[3:6]] == pytest.approx([0.002, 0.0, -.002])

    distributed = build_chambered_model(
        core=core,
        windings=(WindingDefinition(
            "secondary", RectangularConductor("bar", width=0.001, rectangular_height=0.001), 7,
        ),),
        root_chamber=chamber,
        assignments=(ChamberWindingAssignment(
            chamber,
            "secondary",
            7,
            h_loc=Distribute.Horizontally,
            v_loc=Distribute.Vertically,
        ),),
        insulation=insulation,
    )
    assert distributed.conductors[0].shape == "rectangular"
    assert len({turn.center[0] for turn in distributed.windings[0].turns}) == 2


@pytest.mark.parametrize(
    ("layout", "secondary_paths", "interwinding_insulation"),
    [("concentric", 3, 0.001), ("concentric", 2, 0.001), ("vertical", 3, 0.0005)],
)
def test_d_series_notebook_chamber_layouts_fit(
    layout, secondary_paths, interwinding_insulation,
):
    core = core_for_window(
        core_area=301.23e-6,
        window_outer_radius=0.028,
        window_half_height=0.015,
    )
    insulation = InsulationSpec(
        c2w_left=0.0019,
        c2w_right=0.003,
        c2w_top=0.0019,
        c2w_bottom=0.0019,
        w2w_matrix=[
            [0.0005, interwinding_insulation],
            [interwinding_insulation, 0.0005],
        ],
    )
    root = initial_chamber(core, insulation)
    split_direction = "vertical" if layout == "vertical" else "horizontal"
    primary_chamber, secondary_chamber = split_chamber(
        root,
        split_dir=split_direction,
        count=2,
        split_ins=interwinding_insulation,
    )
    primary = WindingDefinition(
        "primary",
        RoundConductor("primary_litz", radius=0.00275 / 2, construction="litz",
                       strands=400, strand_diameter=0.0001),
        8,
        turn_insulation=0.0005,
    )
    secondary = WindingDefinition(
        "secondary",
        RoundConductor("secondary_litz", radius=0.00275 / 2, construction="litz",
                       strands=400, strand_diameter=0.0001),
        2,
        turn_insulation=0.0005,
    )
    component = build_chambered_model(
        core=core,
        windings=(primary, secondary),
        root_chamber=root,
        assignments=(
            ChamberWindingAssignment(
                primary_chamber,
                "primary",
                8,
                h_loc=Align.Center if layout == "vertical" else Align.Left,
            ),
            ChamberWindingAssignment(
                secondary_chamber,
                "secondary",
                2,
                parallel_paths=secondary_paths,
                h_loc=Align.Center if layout == "vertical" else Align.Right,
            ),
        ),
        insulation=insulation,
    )

    assert [circuit.turns for circuit in component.circuits] == [8, 2]
    assert [len(winding.turns) for winding in component.windings] == [
        8,
        2 * secondary_paths,
    ]


def test_radial_stack_derives_d2_window_and_orders_primary_inside_secondary():
    insulation = InsulationSpec(
        c2w_left=0.0019,
        c2w_right=0.003,
        c2w_top=0.0019,
        c2w_bottom=0.0019,
        w2w_matrix=[[0.0005, 0.001], [0.001, 0.0005]],
    )
    primary = WindingDefinition(
        "primary", RoundConductor("primary", radius=0.00275 / 2), 8,
    )
    secondary = WindingDefinition(
        "secondary", RoundConductor("secondary", radius=0.00275 / 2), 2,
    )
    layout = RadialStack(
        sections=(
            WindingSection("primary", 8, layers=1, h_loc=Align.Left),
            WindingSection("secondary", 2, parallel_paths=2, layers=1, h_loc=Align.Right),
        ),
        split_ins=0.001,
    ).layout(
        core_area=301.23e-6,
        windings=(primary, secondary),
        insulation=insulation,
    )

    assert layout.scheme == "radial_inside_to_outside"
    assert layout.chamber_width_m == pytest.approx(0.0065)
    assert layout.chamber_height_m == pytest.approx(0.0255)
    assert layout.window_outer_radius_m == pytest.approx(
        (301.23e-6 / 3.141592653589793) ** 0.5 + 0.0019 + 0.0065 + 0.003
    )
    assert layout.window_height_m == pytest.approx(0.0293)
    assert layout.assignments[0].chamber.r_max < layout.assignments[1].chamber.r_min
    component = layout.build_model()
    assert [turn.center[0] for turn in component.windings[0].turns] == sorted(
        turn.center[0] for turn in component.windings[0].turns
    )


def test_axial_stack_derives_d3_window_with_primary_above_secondary():
    insulation = InsulationSpec(
        c2w_left=0.0019,
        c2w_right=0.003,
        c2w_top=0.0019,
        c2w_bottom=0.0019,
        w2w_matrix=[[0.0005, 0.0005], [0.0005, 0.0005]],
    )
    primary = WindingDefinition(
        "primary", RoundConductor("primary", radius=0.00275 / 2), 8,
    )
    secondary = WindingDefinition(
        "secondary", RoundConductor("secondary", radius=0.00275 / 2), 2,
    )
    layout = AxialStack(
        sections=(
            WindingSection("primary", 8, layers=2, h_loc=Align.Center),
            WindingSection("secondary", 2, parallel_paths=3, layers=2, h_loc=Align.Center),
        ),
        split_ins=0.0005,
    ).layout(
        core_area=301.23e-6,
        windings=(primary, secondary),
        insulation=insulation,
    )

    assert layout.scheme == "axial_top_to_bottom"
    assert layout.chamber_width_m == pytest.approx(0.006)
    assert layout.chamber_height_m == pytest.approx(0.02225)
    assert layout.window_height_m == pytest.approx(0.02605)
    component = layout.build_model()
    primary_z = [turn.center[1] for turn in component.windings[0].turns]
    secondary_z = [turn.center[1] for turn in component.windings[1].turns]
    assert min(primary_z) > max(secondary_z)
    assert [assignment.layers for assignment in layout.assignments] == [2, 2]