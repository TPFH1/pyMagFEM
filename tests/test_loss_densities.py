from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.figure import Figure

from fem import (
    MU0,
    AxisymmetricMesh,
    FrequencyDomainResult,
    FrequencyDomainSimulation,
    LitzProperties,
    NonlinearCoreMaterial,
    PhysicalField,
    loss_densities,
    magnetic_flux_density,
    meeker_litz_properties,
    solve_frequency_domain,
    solve_nonlinear_frequency_domain,
    solve_voltage_excitation,
)
from fem.losses import element_volumes
from model import Circuit, Conductor, FEMModel, Material, TurnPlacement, Winding


@pytest.fixture
def mixed_case():
    points = np.array([[r, z] for z in (0., 1., 2.) for r in (1., 2., 3.)])
    triangles = np.array([
        [0, 1, 4], [0, 4, 3], [1, 2, 5], [1, 5, 4],
        [3, 4, 7], [3, 7, 6], [4, 5, 8], [4, 8, 7],
    ])
    materials = [
        Material("core", relative_permeability=1000., permeability_loss_angle_deg=30.,
                 relative_permittivity=20., permittivity_loss_angle_deg=10.),
        Material("copper", conductivity=1e6),
        Material("other", conductivity=2e6),
    ]
    mesh = AxisymmetricMesh(
        points, triangles, np.array(["core"] * 2 + ["winding"] * 4 + ["air"] * 2),
        {"outer": np.array([0, 1, 2, 3, 5, 6, 7, 8]), "r0": np.array([], dtype=int)},
        {"p_cond_center_1": [[2.5, .5], [1.5, 1.5]],
         "conductor_windings": ["primary", "secondary"],
         "conductor_circuits": ["primary", "secondary"]},
        {"conductor_0": np.array([2, 3]), "conductor_1": np.array([4, 5])},
        {material.name: material for material in materials},
    )
    component = FEMModel(
        materials=materials,
        conductors=[
            Conductor("litz", construction="litz", radius=.005, strands=4,
                      strand_diameter=.001, material="copper"),
            Conductor("solid", radius=.005, material="other"),
        ],
        circuits=[Circuit("primary"), Circuit("secondary")],
        windings=[
            Winding("primary", conductor="litz", circuit="primary",
                    turns=[TurnPlacement([2.5, .5], "litz", "primary")]),
            Winding("secondary", conductor="solid", circuit="secondary",
                    turns=[TurnPlacement([1.5, 1.5], "solid", "secondary")]),
        ],
    )
    component.excitation.frequency = 50e3
    component.excitation.circuits = {"primary": 2., "secondary": 3.}
    return component, mesh


@pytest.mark.parametrize("nonlinear", [False, True])
def test_loss_plots_and_integrals_share_linear_and_nonlinear_decomposition(mixed_case, nonlinear):
    component, mesh = mixed_case
    material = NonlinearCoreMaterial(
        lambda b: 2000. / (1. + b) - 1j * (100. + 200. * b),
        2. + 3.j, 90., 50e3,
    ) if nonlinear else None
    a_phi = np.arange(len(mesh.points), dtype=complex) * .01
    result = FrequencyDomainResult.from_solution(
        component, mesh, a_phi,
        {"conductor_0": np.full(8, 1000.), "conductor_1": np.full(8, 2000.)},
        nonlinear_materials={"core": material} if material else None,
    )
    densities = result.loss_densities()
    np.testing.assert_array_equal(densities.total, densities.electrical + densities.magnetic)
    for field, expected in (
        (PhysicalField.LOSS_DENSITY, densities.total),
        (PhysicalField.MAG_LOSS_DENSITY, densities.magnetic),
        (PhysicalField.ELECTRIC_LOSS_DENSITY, densities.electrical),
    ):
        ax = Figure().subplots()
        result.plot_physical_field(field, ax=ax)
        np.testing.assert_allclose(ax.collections[0].get_array(), expected)
    properties = result.litz_properties["conductor_0"]
    np.testing.assert_allclose(densities.electrical[2:4], .5 * properties.resistivity.real * 1e6)
    np.testing.assert_allclose(densities.electrical[4:6], .5 * 2000.**2 / 2e6)
    h = result.magnetic_field_intensity()
    np.testing.assert_allclose(
        densities.magnetic[2:4],
        -.5 * 2*np.pi*result.frequency_hz * properties.absolute_permeability.imag
        * np.linalg.norm(h[2:4], axis=1)**2,
    )
    np.testing.assert_array_equal(densities.magnetic[4:], 0.)
    np.testing.assert_array_equal(densities.electrical[6:], 0.)
    core_mu = (
        material.absolute_permeability_at(np.linalg.norm(magnetic_flux_density(mesh, a_phi)[:2],
                                                       axis=1))
        if material else MU0 * 1000. * np.exp(-1j*np.deg2rad(30.))
    )
    np.testing.assert_allclose(
        densities.magnetic[:2],
        -.5 * 2*np.pi*result.frequency_hz * np.imag(core_mu)
        * np.linalg.norm(h[:2], axis=1)**2,
    )
    assert np.all(densities.electrical[:2] > 0.)
    volumes = element_volumes(mesh)
    magnetic, electrical, total = result.core_losses()
    assert magnetic == pytest.approx(np.dot(volumes[:2], densities.magnetic[:2]))
    assert electrical == pytest.approx(np.dot(volumes[:2], densities.electrical[:2]))
    assert total == pytest.approx(magnetic + electrical)
    winding_losses = result.winding_losses()
    for name, indices in (("primary", [2, 3]), ("secondary", [4, 5])):
        assert winding_losses[name].transport == pytest.approx(
            np.dot(volumes[indices], densities.electrical[indices]),
        )
        assert winding_losses[name].proximity == pytest.approx(
            np.dot(volumes[indices], densities.magnetic[indices]),
        )
    assert result.loss_summary()["total"] == pytest.approx(np.dot(volumes, densities.total))
    with pytest.raises(ValueError, match="solid-only"):
        result.winding_copper_losses()


def test_conductor_properties_are_resolved_once_and_preserved(mixed_case):
    component, mesh = mixed_case
    result = FrequencyDomainResult.from_solution(
        component, mesh, np.zeros(9), {"conductor_0": np.ones(8), "conductor_1": np.ones(8)},
    )
    expected = meeker_litz_properties(
        50e3, strand_diameter=.001, fill_factor=.04, conductivity=1e6,
    )
    assert result.litz_properties["conductor_0"].resistivity == pytest.approx(expected.resistivity)
    assert result.litz_properties["conductor_0"].absolute_permeability == pytest.approx(
        expected.absolute_permeability,
    )
    assert result.conductor_conductivities["conductor_1"] == 2e6
    before = result.loss_densities()
    component.materials[1].conductivity = 1.
    component.materials[2].conductivity = 2.
    after = result.loss_densities()
    np.testing.assert_array_equal(before.total, after.total)
    result.litz_properties["conductor_0"] = LitzProperties(MU0, 10.+2.j, 1/(10.+2.j), 1.)
    np.testing.assert_allclose(result.loss_densities().electrical[2:4], 5.)


def test_invalid_litz_metadata_is_not_treated_as_solid(mixed_case):
    component, mesh = mixed_case
    component.conductors[0].strand_diameter = None
    with pytest.raises(ValueError, match="Litz conductors require"):
        FrequencyDomainResult.from_solution(component, mesh, np.zeros(9), {})


def test_high_level_solve_forwards_per_conductor_properties(mixed_case):
    component, mesh = mixed_case
    with patch("fem.simulation.solve_frequency_domain",
               return_value=(np.zeros(9, complex), {}, {})) as solve:
        result = FrequencyDomainSimulation(component, mesh=mesh).solve()
    options = solve.call_args.kwargs
    assert options["conductor_conductivities"]["conductor_1"] == 2e6
    assert options["litz_properties"] == result.litz_properties
    assert options["current_modes"] == {"conductor_0": "fixed_current_density"}
    assert options["material_conductivities"]["core"].real > 0.


def test_nonlinear_electrical_properties_reach_each_iteration(mixed_case):
    _, mesh = mixed_case
    material = NonlinearCoreMaterial(lambda b: np.ones_like(b) * (1000.-10.j),
                                    123.+4.j, 90., 50e3)
    with patch("fem.nonlinear_solver.solve_frequency_domain",
               return_value=(np.zeros(9, complex), {})) as solve:
        solve_nonlinear_frequency_domain(
            mesh, 50e3, {}, nonlinear_materials={"core": material},
            material_conductivities={"core": 1.}, progress=False,
        )
    assert solve.call_args.kwargs["material_conductivities"]["core"] == material.conductivity


def test_ac_material_overrides_match_explicit_materials(mixed_case):
    _, mesh = mixed_case
    frequency = 100.
    sigmas = {"conductor_0": 10., "conductor_1": 20.}
    material = NonlinearCoreMaterial(lambda b: np.full_like(b, 1000.-10.j, dtype=complex),
                                    50.+7.j, 90., frequency)
    options = {
        "conductor_conductivities": sigmas, "conductivity": 10., "return_voltages": True,
    }
    nonlinear = solve_frequency_domain(
        mesh, frequency, {"primary": 2., "secondary": 3.},
        nonlinear_materials={"core": material}, **options,
    )
    mesh.materials["core"] = Material(
        "core", conductivity=50.,
        relative_permittivity=7. / (2*np.pi*frequency*8.8541878128e-12),
    )
    reference = solve_frequency_domain(
        mesh, frequency, {"primary": 2., "secondary": 3.},
        relative_permeability={"core": 1000.-10.j}, **options,
    )
    np.testing.assert_allclose(nonlinear[0], reference[0])
    for name in sigmas:
        np.testing.assert_allclose(nonlinear[1][name], reference[1][name])
        assert nonlinear[2][name] == pytest.approx(reference[2][name])
    _, _, uniform = solve_frequency_domain(
        mesh, frequency, {"primary": 2., "secondary": 3.},
        relative_permeability={"core": 1000.-10.j}, conductivity=10., return_voltages=True,
    )
    assert not np.isclose(reference[2]["conductor_1"], uniform["conductor_1"])


def test_dc_and_voltage_solver_use_effective_litz_conductivity(mixed_case):
    _, mesh = mixed_case
    litz = LitzProperties(MU0, 4/58e6, 58e6/4, 0.)
    args = (mesh, 0., {"conductor_0": 1., "conductor_1": 1.})
    _, _, baseline = solve_frequency_domain(*args, return_voltages=True)
    _, _, effective = solve_frequency_domain(
        *args, return_voltages=True, litz_properties={"conductor_0": litz},
        conductor_conductivities={"conductor_1": 58e6/2},
    )
    assert effective["conductor_0"] == pytest.approx(4 * baseline["conductor_0"])
    assert effective["conductor_1"] == pytest.approx(2 * baseline["conductor_1"])
    with patch("fem.voltage_solver.solve_frequency_domain",
               return_value=(np.zeros(9, complex), {}, {"conductor_0": 1.})) as solve:
        solve_voltage_excitation(mesh, 0., 2., winding="primary",
                                 litz_properties={"conductor_0": litz})
    assert solve.call_args.kwargs["litz_properties"] == {"conductor_0": litz}


@pytest.mark.parametrize(("frequency", "current_mode"), [
    (0., "total_current"),
    (0., "fixed_current_density"),
    (100., "fixed_current_density"),
])
def test_sources_not_using_ohms_law_allow_zero_conductivity(mixed_case, frequency, current_mode):
    _, mesh = mixed_case
    options = {
        "current_modes": current_mode,
        "fixed_current_density": {"conductor_0": 2., "conductor_1": 3.},
    }
    reference = solve_frequency_domain(
        mesh, frequency, {"primary": 2., "secondary": 3.}, **options,
    )
    actual = solve_frequency_domain(
        mesh, frequency, {"primary": 2., "secondary": 3.}, conductivity=0., **options,
    )
    np.testing.assert_allclose(actual[0], reference[0])
    for name in actual[1]:
        np.testing.assert_allclose(actual[1][name], reference[1][name])


def test_missing_nonlinear_field_and_loss_inputs_are_explicit(mixed_case):
    _, mesh = mixed_case
    nonlinear = NonlinearCoreMaterial(lambda b: np.ones_like(b) * 1000., 1.j, 90., 50e3)
    with pytest.raises(ValueError, match="a_phi"):
        loss_densities(mesh, 50e3, magnetic_field=np.zeros((8, 2)),
                       nonlinear_material=nonlinear, conductor_names=[])
    with pytest.raises(ValueError, match="current density"):
        loss_densities(mesh, 50e3)
