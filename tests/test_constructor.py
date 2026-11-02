from dataclasses import replace

import pytest

from model import Constructor, CoreSpec, InsulationSpec, RoundConductor


@pytest.mark.parametrize("core_area", [0.0, -1.0, float("nan")])
def test_concentric_factory_rejects_invalid_core_area(core_area):
    from model import build_concentric_model

    with pytest.raises(ValueError, match="core_area"):
        build_concentric_model(core_area=core_area, windings=(), insulation=InsulationSpec())


@pytest.mark.parametrize("alignment,bottom", [("bottom", 0.001), ("center", 0.0085), ("top", 0.016)])
def test_concentric_factory_preserves_core_and_winding_geometry(alignment, bottom):
    from math import pi

    from model import RadialWindingSpec, build_concentric_model

    component = build_concentric_model(
        core_area=455.58e-6,
        windings=(
            RadialWindingSpec("primary", RoundConductor("primary", radius=0.0011), 14),
            RadialWindingSpec("secondary", RoundConductor("secondary", radius=0.00165),
                              2, parallel_paths=3, alignment=alignment),
        ),
        insulation=InsulationSpec(0.001, 0.001, 0.001, 0.001, 0.0005),
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