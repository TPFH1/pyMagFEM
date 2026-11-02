from dataclasses import replace

import pytest

from model import Constructor, CoreSpec, InsulationSpec, RoundConductor


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