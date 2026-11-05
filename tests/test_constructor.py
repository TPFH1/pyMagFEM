from dataclasses import replace

import pytest

from model import Constructor, CoreSpec, InsulationSpec, RoundConductor


def test_concentric_insulation_matrix_controls_diagonal_and_interwinding_gaps():
    from dataclasses import asdict

    from model import RadialWindingSpec, build_concentric_model, normalize

    insulation = InsulationSpec(0.001, 0.001, 0.001, 0.001,
                               w2w_matrix=[[0.0002, 0.0012], [0.0012, 0.0007]])
    component = build_concentric_model(
        core_area=100e-6,
        windings=(
            RadialWindingSpec("primary", RoundConductor("primary", radius=0.001), 3),
            RadialWindingSpec("secondary", RoundConductor("secondary", radius=0.001), 2),
        ),
        insulation=insulation,
    )
    primary, secondary = component.windings
    assert primary.turns[1].center[1] - primary.turns[0].center[1] == pytest.approx(0.0022)
    assert secondary.turns[1].center[1] - secondary.turns[0].center[1] == pytest.approx(0.0027)
    assert secondary.turns[0].center[0] - primary.turns[0].center[0] - 0.002 == pytest.approx(0.0012)
    stored = component.geometry.core.extra["insulation_mm"]["w2w_matrix"]
    assert stored == [[0.2, 1.2], [1.2, 0.7]]
    assert normalize(asdict(component)).geometry.core.extra["insulation_mm"]["w2w_matrix"] == stored


@pytest.mark.parametrize("matrix", [
    [[0.001, 0.002], [0.003, 0.001]],
    [[0.001, -0.001], [-0.001, 0.001]],
    [[0.001], [0.001, 0.001]],
    [[float("nan")]],
    [],
])
def test_insulation_matrix_must_be_symmetric_square_finite_and_nonnegative(matrix):
    with pytest.raises(ValueError):
        InsulationSpec(w2w_matrix=matrix)


@pytest.mark.parametrize("core_area", [0.0, -1.0, float("nan")])
def test_concentric_factory_rejects_invalid_core_area(core_area):
    from model import build_concentric_model

    with pytest.raises(ValueError, match="core_area"):
        build_concentric_model(core_area=core_area, windings=(), insulation=InsulationSpec())


@pytest.mark.parametrize("alignment,bottom", [("bottom", 0.001), ("center", 0.0085), ("top", 0.016)])
@pytest.mark.parametrize("use_matrix", [False, True])
def test_concentric_factory_preserves_core_and_winding_geometry(alignment, bottom, use_matrix):
    from math import pi

    from model import RadialWindingSpec, build_concentric_model

    component = build_concentric_model(
        core_area=455.58e-6,
        windings=(
            RadialWindingSpec("primary", RoundConductor("primary", radius=0.0011), 14),
            RadialWindingSpec("secondary", RoundConductor("secondary", radius=0.00165),
                              2, parallel_paths=3, alignment=alignment),
        ),
        insulation=InsulationSpec(
            0.001, 0.001, 0.001, 0.001, 0.0005,
            w2w_matrix=[[0.0005, 0.0005], [0.0005, 0.0005]] if use_matrix else None,
        ),
        air_gap_length=0.00005,
    )
    window = component.geometry.winding_window.corners
    outer = component.geometry.core.corners
    assert window[2][1] - window[0][1] == pytest.approx(0.0393)
    assert window[1][0] - window[0][0] == pytest.approx(0.008)
    assert pi * window[0][0]**2 == pytest.approx(455.58e-6)
    assert pi * (outer[1][0]**2 - window[1][0]**2) == pytest.approx(455.58e-6)
    assert 2 * pi * window[0][0] * (outer[2][1] - window[2][1]) == pytest.approx(455.58e-6)
    assert [len(winding.turns) for winding in component.windings] == [14, 6]
    assert [circuit.turns for circuit in component.circuits] == [14, 2]
    assert (component.windings[1].turns[0].center[1] - 0.00165 - window[0][1]
            == pytest.approx(bottom))


def test_vertical_layout_places_two_column_windings_above_and_below_gap():
    from model import RadialWindingSpec, build_concentric_model

    conductor = RoundConductor("wire", radius=0.001375)
    component = build_concentric_model(
        core_area=301.23e-6,
        windings=(
            RadialWindingSpec("primary", conductor, 8, radial_columns=2),
            RadialWindingSpec(
                "secondary", conductor, 2, parallel_paths=3, radial_columns=2,
            ),
        ),
        insulation=InsulationSpec(
            0.0019, 0.003, 0.0019, 0.0019,
            w2w_matrix=[[0.0005, 0.001], [0.001, 0.0005]],
        ),
        winding_layout="vertical",
    )

    primary, secondary = component.windings
    assert [len(winding.turns) for winding in component.windings] == [8, 6]
    assert [circuit.turns for circuit in component.circuits] == [8, 2]
    assert [winding.parallel_paths for winding in component.windings] == [1, 3]

    primary_radii = sorted({turn.center[0] for turn in primary.turns})
    secondary_radii = sorted({turn.center[0] for turn in secondary.turns})
    assert len(primary_radii) == 2
    assert secondary_radii == pytest.approx(primary_radii)
    assert len({turn.center[1] for turn in primary.turns}) == 4
    assert len({turn.center[1] for turn in secondary.turns}) == 3

    primary_top = max(turn.center[1] for turn in primary.turns) + conductor.radius
    secondary_bottom = min(turn.center[1] for turn in secondary.turns) - conductor.radius
    assert secondary_bottom - primary_top == pytest.approx(0.001)


def test_insulation_matrix_size_must_match_winding_count():
    from model import RadialWindingSpec, build_concentric_model

    with pytest.raises(ValueError, match="number of windings"):
        build_concentric_model(
            core_area=100e-6,
            windings=(RadialWindingSpec("primary", RoundConductor("wire", radius=0.001), 1),),
            insulation=InsulationSpec(w2w_matrix=[[0, 0.001], [0.001, 0]]),
        )


def test_constructor_enforces_offdiagonal_insulation_clearance():
    core = CoreSpec("core", "core", 0.025, 0.025, 0.010, 0.020, 0.020)
    wire = RoundConductor("wire", radius=0.001)
    windings = tuple(
        Constructor.place_winding(
            core=core, insulation=InsulationSpec(c2w_left=radial_offset),
            name=name, conductor=wire, circuit=name, turns=1, radial_columns=1,
        )
        for name, radial_offset in (("primary", 0.001), ("secondary", 0.0035))
    )
    with pytest.raises(ValueError, match="clearance"):
        Constructor(core, windings, insulation=InsulationSpec(
            w2w_matrix=[[0, 0.001], [0.001, 0]],
        )).build()


def test_place_winding_uses_selected_matrix_diagonal():
    core = CoreSpec("core", "core", 0.025, 0.025, 0.010, 0.020, 0.020)
    winding = Constructor.place_winding(
        core=core, insulation=InsulationSpec(w2w_matrix=[[0.0002, 0.001], [0.001, 0.0007]]),
        name="secondary", conductor=RoundConductor("wire", radius=0.001),
        circuit="secondary", turns=2, radial_columns=1, winding_index=1,
    )
    assert winding.layers[1].z - winding.layers[0].z == pytest.approx(0.0027)


@pytest.mark.parametrize("parallel_paths", [0, -1, 1.5, True])
def test_parallel_paths_must_be_positive_integer(parallel_paths):
    from model import Winding

    with pytest.raises(ValueError, match="positive integer"):
        Winding("secondary", parallel_paths=parallel_paths)


def test_parallel_winding_keeps_electrical_turn_count():
    from dataclasses import asdict

    from model import normalize

    core = CoreSpec("core", "core", 0.025, 0.025, 0.012, 0.020, 0.020)
    winding = Constructor.place_winding(
        core=core, insulation=InsulationSpec(w2w=0.0005), name="secondary",
        conductor=RoundConductor("litz", radius=0.00165, construction="litz",
                                 strands=600, strand_diameter=0.0001),
        circuit="secondary", turns=2, radial_columns=1, parallel_paths=4,
    )
    component = Constructor(core, (winding,)).build()
    assert winding.turns_count == 2
    assert len(component.windings[0].turns) == 8
    assert component.windings[0].parallel_paths == 4
    assert component.circuits[0].turns == 2
    geometry = component.legacy_geometry()
    assert geometry["conductor_current_fractions"] == [0.25] * 8
    restored = normalize(asdict(component))
    assert restored.windings[0].parallel_paths == 4
    assert restored.legacy_geometry()["conductor_current_fractions"] == [0.25] * 8


@pytest.mark.parametrize("radial_offset", [0.0, 0.00285, 0.0031])
def test_radial_transformer_winding_clearance(radial_offset):
    core = CoreSpec("core", "core", 0.019943733351606967, 0.018475,
                    0.00745, 0.0185, 0.01475)
    conductor = RoundConductor("copper", radius=0.0013)
    insulation = InsulationSpec(0.003, 0.00235, 0.0026, 0.0026, 0.0005)
    primary = Constructor.place_winding(
        core=core, insulation=insulation, name="primary", conductor=conductor,
        circuit="primary", turns=8, radial_columns=1,
    )
    secondary = Constructor.place_winding(
        core=core, insulation=replace(insulation, c2w_left=0.003 + radial_offset),
        name="secondary", conductor=conductor, circuit="secondary", turns=8,
        radial_columns=1,
    )
    constructor = Constructor(core, (primary, secondary), insulation=insulation)
    if radial_offset < 0.0031:
        with pytest.raises(ValueError, match="clearance"):
            constructor.build()
    else:
        component = constructor.build()
        assert [len(winding.turns) for winding in component.windings] == [8, 8]
        assert component.geometry.core.corners == core.regions()[0].corners