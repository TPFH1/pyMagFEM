"""PQ4040 400 V/48 V, 2 kW transformer study."""
from __future__ import annotations

from model import (
    AirGapSpec,
    Constructor,
    CoreSpec,
    InsulationSpec,
    RectangularConductor,
    RoundConductor,
    WindingLayer,
    WindingSpec,
)
from model.schema import FEMModel


def make_pq4040_2kw_transformer() -> FEMModel:
    """Build the application-specific PQ4040 winding arrangement."""
    millimetre = 1e-3
    primary = RoundConductor(
        "primary_litz",
        radius=0.564 * millimetre,
        construction="litz",
        strands=64,
        strand_diameter=0.14 * millimetre,
    )
    secondary = RectangularConductor(
        "secondary_foil",
        width=10.0 * millimetre,
        rectangular_height=0.833 * millimetre,
    )
    primary_radii = tuple(value * millimetre for value in (8.3, 11.1, 13.9, 16.7))
    insulation = InsulationSpec(
        c2w_left=2.0 * millimetre,
        c2w_right=2.0 * millimetre,
        c2w_top=2.0 * millimetre,
        c2w_bottom=2.0 * millimetre,
        w2w=0.5 * millimetre,
    )
    layer_heights = (
        2 * primary.radius,
        secondary.height,
        2 * primary.radius,
        2 * primary.radius,
        secondary.height,
        2 * primary.radius,
    )
    stack_height = sum(layer_heights) + 5 * insulation.w2w
    z = -stack_height / 2.0
    centers: list[float] = []
    for height in layer_heights:
        centers.append(z + height / 2.0)
        z += height + insulation.w2w
    primary_layers = tuple(
        WindingLayer("primary", primary.name, centers[index], primary_radii, "primary")
        for index in (0, 2, 3, 5)
    )
    secondary_layers = tuple(
        WindingLayer("secondary", secondary.name, centers[index], (12.975 * millimetre,), "secondary")
        for index in (1, 4)
    )
    return Constructor(
        core=CoreSpec(
            name="core",
            material="core",
            outer_radius=19.943733351606967 * millimetre,
            outer_half_height=18.475 * millimetre,
            window_inner_radius=7.45 * millimetre,
            window_outer_radius=18.5 * millimetre,
            window_half_height=14.75 * millimetre,
            core_type="PQ4040",
        ),
        windings=(
            WindingSpec("primary", primary, "primary", primary_layers, current=5.0),
            WindingSpec(
                "secondary", secondary, "secondary", secondary_layers,
                current=-2000.0 / 48.0,
            ),
        ),
        air_gaps=(AirGapSpec("center_gap", 3.725 * millimetre, 0.0, 7.45 * millimetre, 0.5 * millimetre),),
        insulation=insulation,
        metadata={"name": "PQ4040 2 kW transformer"},
    ).build()
