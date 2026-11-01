import numpy as np

from fem import (
    MU0,
    AxisymmetricMesh,
    absolute_to_relative_permeability,
    average_abs_current_density,
    build_mesh,
    build_result_report,
    compare_domainc_blocks,
    complex_conductivity,
    complex_relative_permeability,
    copper_loss,
    electric_loss_density,
    integrated_current,
    litz_loss,
    magnetic_field_intensity,
    magnetic_flux_density,
    magnetic_flux_density_nodal,
    magnetic_loss_density,
    meeker_litz_properties,
    solve_axisymmetric,
    solve_combined,
    solve_frequency_domain,
    total_litz_loss_density,
    winding_flux_linkages,
    winding_litz_losses,
    winding_voltages,
)
from fem.materials import relative_to_absolute_permeability
from model import (
    Circuit,
    Conductor,
    FEMModel,
    Geometry,
    Region,
    RoundConductor,
    TurnPlacement,
    Winding,
)


def _mesh():
    points = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0], [1.5, 0.5]])
    triangles = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    return AxisymmetricMesh(
        points,
        triangles,
        np.array(["winding"] * 4),
        {"r0": np.array([], dtype=int), "outer": np.array([0, 1, 2, 3])},
        {"radius_cond_1": 1.0, "p_cond_center_1": [[1.5, 0.5]]},
        {"conductor_0": np.arange(4)},
    )


def test_permeability_conversion_round_trip():
    relative = np.array([1.0, 1500.0, 2.5 + 0.1j])
    absolute = relative_to_absolute_permeability(relative)
    assert np.allclose(absolute, MU0 * relative)
    assert np.allclose(absolute_to_relative_permeability(absolute), relative)


def test_magnetic_field_uses_relative_permeability_once():
    mesh = _mesh()
    a_phi = np.array([0.0, 0.1, 0.2, 0.3, 0.15])
    flux_density = magnetic_flux_density(mesh, a_phi)
    field = magnetic_field_intensity(
        mesh,
        a_phi,
        relative_permeability={"winding": 1.0},
    )
    assert np.allclose(field, flux_density / MU0)


def test_low_frequency_current_and_finite_field():
    mesh = _mesh()
    current = 2.0
    a_phi, densities = solve_frequency_domain(mesh, frequency=0.0, currents=current)
    density = densities["conductor_0"]
    assert np.isclose(integrated_current(mesh, density, "conductor_0"), current)
    assert np.isclose(average_abs_current_density(mesh, density, "conductor_0"), current, rtol=0.05)
    assert np.isfinite(a_phi).all()
    assert np.isfinite(magnetic_flux_density(mesh, a_phi)).all()


def test_magnetostatic_and_zero_frequency_match():
    mesh = _mesh()
    static = solve_axisymmetric(mesh, current_density={"winding": 2.0})
    dynamic, _ = solve_frequency_domain(mesh, frequency=0.0, currents=2.0)
    assert np.allclose(dynamic, static)


def test_nodal_flux_density_is_continuous_field():
    mesh = _mesh()
    a_phi, _ = solve_frequency_domain(mesh, frequency=1.0e4, currents=2.0)
    element = magnetic_flux_density(mesh, a_phi)
    nodal = magnetic_flux_density_nodal(mesh, a_phi)
    assert element.shape == (len(mesh.triangles), 2)
    assert nodal.shape == (len(mesh.points), 2)
    assert np.isfinite(nodal).all()


def test_finite_frequency_preserves_integrated_current():
    mesh = _mesh()
    _, densities = solve_frequency_domain(mesh, frequency=1.0e4, currents=2.0)
    assert np.isclose(
        integrated_current(mesh, densities["conductor_0"], "conductor_0"), 2.0, rtol=1e-8
    )


def test_result_report_is_json_safe_and_supports_named_losses():
    mesh = _mesh()
    model = FEMModel()
    density = np.full(len(mesh.triangles), 2.0 + 1.0j)
    report = build_result_report(
        model,
        mesh,
        frequency_hz=900_000.0,
        currents_a={"primary": 2.0},
        current_densities={"conductor_0": density},
        flux_density=np.ones((len(mesh.triangles), 2)),
        inductance_h={"primary": 1.0e-6 + 2.0e-6j},
        core_losses={
            "core_a": (1.0, 2.0, 3.0),
            "core_b": (4.0, 5.0, 9.0),
        },
        turn_voltages={"conductor_0": 1.0 + 2.0j},
    )
    data = report.to_dict()
    assert data["schema"] == "pyMagBEM.result"
    assert set(data["losses"]["cores"]) == {"core_a", "core_b"}
    assert data["inductance"]["primary"]["magnitude"] == np.sqrt(5.0) * 1e-6
    assert data["voltages"]["turns"]["conductor_0"]["phase_deg"] == np.degrees(np.arctan2(2.0, 1.0))


def test_complex_field_and_loss_are_finite():
    mesh = _mesh()
    a_phi, densities = solve_frequency_domain(mesh, frequency=1.0e4, currents=2.0j)
    assert np.iscomplexobj(a_phi)
    assert np.isfinite(magnetic_flux_density(mesh, a_phi)).all()
    assert np.isfinite(copper_loss(mesh, densities))


def test_complex_dirichlet_boundary_is_zero():
    mesh = _mesh()
    a_phi, _ = solve_frequency_domain(mesh, frequency=1.0e4, currents=2.0)
    fixed = mesh.boundaries["outer"]
    assert np.iscomplexobj(a_phi)
    assert np.all(a_phi[fixed] == 0.0 + 0.0j)


def test_dc_density_is_uniform_and_matches_static_source():
    mesh = _mesh()
    a_phi, densities = solve_frequency_domain(mesh, frequency=0.0, currents=2.0)
    assert np.allclose(densities["conductor_0"][mesh.conductor_regions["conductor_0"]], 2.0)
    assert np.allclose(a_phi, solve_axisymmetric(mesh, current_density={"winding": 2.0}))


def test_combined_solver_supports_fixed_and_total_current_conductors():
    mesh = _mesh()
    mesh.conductor_regions = {
        "litz": np.array([0, 1]),
        "solid": np.array([2, 3]),
    }
    mesh.geometry["conductor_windings"] = ["litz", "solid"]
    mesh.geometry["conductor_circuits"] = ["litz", "solid"]
    _, densities = solve_combined(
        mesh,
        frequency=1.0e4,
        currents={"solid": 2.0},
        current_modes={"litz": "fixed_current_density", "solid": "total_current"},
        fixed_current_density={"litz": 3.0},
    )
    assert np.allclose(densities["litz"][[0, 1]], 3.0)
    assert np.isclose(integrated_current(mesh, densities["solid"], "solid"), 2.0, rtol=1e-8)


def test_winding_litz_losses_returns_per_turn_and_total():
    mesh = _mesh()
    losses = winding_litz_losses(
        mesh,
        {"conductor_0": np.ones(4, dtype=complex)},
        np.ones((4, 2), dtype=complex),
        200e3,
        strand_diameter=0.14e-3,
        fill_factor=0.9,
    )
    assert len(losses.per_turn) == 1
    assert losses.total == losses.per_turn[0].total
    assert losses.transport > 0.0
    assert losses.proximity > 0.0


def test_complex_conductivity_includes_dielectric_loss_angle():
    value = complex_conductivity(0.6, 50_000.0, 25.0, 270_000.0)
    assert np.isclose(
        value.real,
        0.6 + 2.0 * np.pi * 270e3 * 8.8541878128e-12 * 50_000.0 * np.sin(np.deg2rad(25.0)),
    )
    assert value.imag > 0.0


def test_complex_relative_permeability_uses_negative_loss_angle():
    permeability = complex_relative_permeability(1500.0, 10.0)
    assert permeability.real > 0.0
    assert permeability.imag < 0.0
    assert np.isclose(abs(permeability), 1500.0)


def test_physics_loss_density_functions():
    density = np.array([2.0 + 0.0j])
    field = np.array([[3.0, 4.0]])
    electric = electric_loss_density(density, 2.0)
    magnetic = magnetic_loss_density(field, 1.0 - 0.5j, 10.0)
    total = total_litz_loss_density(density, field, 0.5 + 0.0j, 1.0 - 0.5j, 10.0)
    assert np.allclose(electric, [1.0])
    assert np.allclose(magnetic, [12.5 * np.pi * 10.0])
    assert np.allclose(total, [1.0 + 12.5 * np.pi * 10.0])


def test_meeker_litz_loss_has_positive_components():
    properties = meeker_litz_properties(200e3, strand_diameter=0.14e-3, fill_factor=0.9)
    assert properties.absolute_permeability.imag < 0.0
    loss = litz_loss(
        _mesh(),
        {"conductor_0": np.ones(4, dtype=complex)},
        np.ones((4, 2), dtype=complex),
        "conductor_0",
        200e3,
        strand_diameter=0.14e-3,
        fill_factor=0.9,
    )
    assert loss.transport > 0.0
    assert loss.proximity > 0.0


def test_frequency_solver_uses_litz_effective_properties():
    mesh = _mesh()
    properties = meeker_litz_properties(200e3, strand_diameter=0.14e-3, fill_factor=0.9)
    reference, _ = solve_frequency_domain(mesh, frequency=200e3, currents=2.0)
    coupled, _ = solve_frequency_domain(
        mesh,
        frequency=200e3,
        currents=2.0,
        litz_properties={"conductor_0": properties},
    )
    assert np.linalg.norm(coupled - reference) > 1e-15


def test_litz_fixed_density_uses_homogenized_current_excitation():
    mesh = _mesh()
    properties = meeker_litz_properties(200e3, strand_diameter=0.14e-3, fill_factor=0.9)
    density = 2.0
    _reference, reference_density = solve_frequency_domain(
        mesh,
        frequency=200e3,
        currents=2.0,
        current_modes={"conductor_0": "fixed_current_density"},
        fixed_current_density={"conductor_0": density},
    )
    coupled, coupled_density = solve_frequency_domain(
        mesh,
        frequency=200e3,
        currents=2.0,
        current_modes={"conductor_0": "fixed_current_density"},
        fixed_current_density={"conductor_0": density},
        litz_properties={"conductor_0": properties},
    )
    assert np.allclose(reference_density["conductor_0"], density * np.array([1, 1, 1, 1]))
    assert np.allclose(coupled_density["conductor_0"], density * np.array([1, 1, 1, 1]))
    assert np.isfinite(coupled).all()


def test_terminal_voltage_has_resistive_and_inductive_parts():
    mesh = _mesh()
    a_phi, densities, source_voltages = solve_frequency_domain(
        mesh, frequency=1.0e4, currents=2.0, return_voltages=True
    )
    voltages = winding_voltages(mesh, a_phi, densities, 2.0, 1.0e4)
    assert set(voltages) == {"conductor_0"}
    assert voltages["conductor_0"].real > 0.0
    assert np.isfinite(voltages["conductor_0"])
    assert np.isclose(voltages["conductor_0"], source_voltages["conductor_0"], rtol=1e-2)


def test_litz_conductor_derives_fill_factor_from_strands():
    conductor = RoundConductor(
        "litz",
        radius=1.3e-3,
        construction="litz",
        strands=150,
        strand_diameter=200e-6,
    )
    expected = 150.0 * (100e-6 / 1.3e-3) ** 2
    assert np.isclose(conductor.fill_factor, expected)


def test_domainc_block_comparison_shows_radial_basis_difference():
    comparison = compare_domainc_blocks(_mesh(), conductivity=58e6)
    assert comparison["relative_difference"]["K"] == 0.0
    assert comparison["relative_difference"]["M"] == 0.0
    assert comparison["relative_difference"]["B"] > 0.1
    assert comparison["relative_difference"]["D"] == 0.0
    assert comparison["relative_difference"]["H"] > 0.1


def test_ring_voltage_constraint_uses_radial_quadrature():
    comparison = compare_domainc_blocks(_mesh(), conductivity=58e6)
    ring_h = comparison["ring"]["H"][0, 0]
    area = 1.0
    centroid_inverse_radius = 1.0 / 1.5
    expected_centroid = 58e6 * area * centroid_inverse_radius / (2.0 * np.pi)
    assert not np.isclose(ring_h, expected_centroid)


def test_normalized_turn_coordinates_are_conductor_centers():
    model = FEMModel(
        geometry=Geometry(
            core=Region("core", corners=[[0.0, -1.0], [2.0, -1.0], [0.0, 1.0], [2.0, 1.0]]),
            winding_window=Region(
                "window", corners=[[1.0, -0.8], [2.0, -0.8], [1.0, 0.8], [2.0, 0.8]]
            ),
        ),
        conductors=[Conductor("copper", radius=0.1)],
        windings=[Winding("primary", [TurnPlacement([1.5, 0.25])])],
    )
    assert model.legacy_geometry()["p_cond_center_1"] == [[1.5, 0.25]]


def test_two_winding_model_preserves_turn_ownership():
    model = FEMModel(
        geometry=Geometry(
            core=Region("core", corners=[[0.0, -1.0], [3.0, -1.0], [0.0, 1.0], [3.0, 1.0]]),
            winding_window=Region(
                "window", corners=[[1.0, -0.8], [3.0, -0.8], [1.0, 0.8], [3.0, 0.8]]
            ),
        ),
        conductors=[Conductor("copper", radius=0.1)],
        windings=[
            Winding("primary", [TurnPlacement([1.5, -0.3], circuit="primary")]),
            Winding("secondary", [TurnPlacement([2.0, 0.3], circuit="secondary")]),
        ],
        circuits=[
            Circuit("primary", turns=1, current=2.0),
            Circuit("secondary", turns=1, current=-1.0),
        ],
    )
    geometry = model.legacy_geometry()
    assert geometry["conductor_windings"] == ["primary", "secondary"]
    assert geometry["conductor_circuits"] == ["primary", "secondary"]
    mesh = build_mesh(model, max_size=0.25)
    a_phi, densities = solve_frequency_domain(
        mesh,
        frequency=0.0,
        currents={"primary": 2.0, "secondary": -1.0},
    )
    assert np.isclose(integrated_current(mesh, densities["conductor_0"], "conductor_0"), 2.0)
    assert np.isclose(integrated_current(mesh, densities["conductor_1"], "conductor_1"), -1.0)
    assert set(winding_flux_linkages(mesh, a_phi)) == {"primary", "secondary"}
