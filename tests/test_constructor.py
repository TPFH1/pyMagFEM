from dataclasses import asdict

import pytest

from model import (
    Constructor,
    CoreSpec,
    InsulationSpec,
    RoundConductor,
    WindingLayer,
    WindingSpec,
    normalize,
)


def winding_from_centers(name, conductor, centers, *, parallel_paths=1):
    layers = tuple(
        WindingLayer(name, conductor.name, z, (radius,), name)
        for radius, z in centers
    )
    return WindingSpec(name, conductor, name, layers, parallel_paths=parallel_paths)


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


def test_constructor_enforces_offdiagonal_insulation_clearance():
    core = CoreSpec("core", "core", 0.025, 0.025, 0.010, 0.020, 0.020)
    wire = RoundConductor("wire", radius=0.001)
    windings = (
        winding_from_centers("primary", wire, [(0.011, 0.0)]),
        winding_from_centers("secondary", wire, [(0.0135, 0.0)]),
    )
    with pytest.raises(ValueError, match="clearance"):
        Constructor(core, windings, insulation=InsulationSpec(
            w2w_matrix=[[0, 0.001], [0.001, 0]],
        )).build()


def test_explicit_winding_layers_respect_selected_matrix_diagonal():
    core = CoreSpec("core", "core", 0.025, 0.025, 0.010, 0.020, 0.020)
    wire = RoundConductor("wire", radius=0.001)
    winding = winding_from_centers(
        "secondary", wire, [(0.015, 0.0), (0.015, 0.0027)],
    )
    component = Constructor(
        core,
        (winding,),
        insulation=InsulationSpec(w2w_matrix=[[0.0007]]),
    ).build()
    assert component.windings[0].turns[1].center[1] - component.windings[0].turns[0].center[1] == pytest.approx(0.0027)


@pytest.mark.parametrize("parallel_paths", [0, -1, 1.5, True])
def test_parallel_paths_must_be_positive_integer(parallel_paths):
    from model import Winding

    with pytest.raises(ValueError, match="positive integer"):
        Winding("secondary", parallel_paths=parallel_paths)


def test_parallel_winding_keeps_electrical_turn_count():
    core = CoreSpec("core", "core", 0.025, 0.025, 0.012, 0.020, 0.020)
    conductor = RoundConductor(
        "litz", radius=0.00165, construction="litz",
        strands=600, strand_diameter=0.0001,
    )
    stack_height = 8 * conductor.height + 7 * 0.0005
    first_z = -stack_height / 2.0 + conductor.height / 2.0
    centers = [(0.015, first_z + index * (conductor.height + 0.0005)) for index in range(8)]
    winding = winding_from_centers("secondary", conductor, centers, parallel_paths=4)
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
    stack_height = 8 * conductor.height + 7 * insulation.w2w
    first_z = -stack_height / 2.0 + conductor.height / 2.0
    z_centers = [first_z + index * (conductor.height + insulation.w2w) for index in range(8)]
    primary_radius = core.window_inner_radius + insulation.c2w_left + conductor.radius
    primary = winding_from_centers(
        "primary", conductor, [(primary_radius, z) for z in z_centers],
    )
    secondary = winding_from_centers(
        "secondary", conductor, [(primary_radius + radial_offset, z) for z in z_centers],
    )
    constructor = Constructor(core, (primary, secondary), insulation=insulation)
    if radial_offset < 0.0031:
        with pytest.raises(ValueError, match="clearance"):
            constructor.build()
    else:
        component = constructor.build()
        assert [len(winding.turns) for winding in component.windings] == [8, 8]
        assert component.geometry.core.corners == core.regions()[0].corners