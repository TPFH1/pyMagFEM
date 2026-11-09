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
    winding_complex_powers,
    winding_litz_losses,
    winding_terminal_voltages,
    winding_voltages,
)
from fem.materials import relative_to_absolute_permeability
from model import (
    Circuit,
    Conductor,
    FEMModel,
    Geometry,
    Material,
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


def test_operating_point_uses_nonlinear_material_for_core_loss_only_when_requested(
    monkeypatch,
):
    from fem import NonlinearCoreMaterial, solve_operating_point

    mesh = _mesh()
    nonlinear_core = NonlinearCoreMaterial(
        relative_permeability=lambda values: np.ones_like(values, dtype=complex),
        conductivity=1.0 + 0.0j,
        temperature=90.0,
        frequency=250e3,
    )
    component = FEMModel(
        materials=[Material("core", relative_permeability=1000.0)],
        circuits=[Circuit("primary")],
    )
    solve_options = {}

    def fake_frequency_solve(mesh, **kwargs):
        solve_options.update(kwargs)
        return np.zeros(len(mesh.points), dtype=complex), {}, {}

    monkeypatch.setattr("fem.simulation.solve_frequency_domain", fake_frequency_solve)
    result = solve_operating_point(
        component,
        mesh=mesh,
        frequency_hz=250e3,
        currents={"primary": 2.0 + 0.0j},
        nonlinear_materials={"core": nonlinear_core},
    )

    assert solve_options["nonlinear_materials"] == {"core": nonlinear_core}
    assert result.nonlinear_material is nonlinear_core
    assert result.nonlinear_materials == {"core": nonlinear_core}

    loss_options = {}

    def fake_core_losses(*_args, **kwargs):
        loss_options.update(kwargs)
        return 0.0, 0.0, 0.0

    monkeypatch.setattr("fem.simulation.core_losses", fake_core_losses)
    result.core_losses()
    assert loss_options["nonlinear_material"] is nonlinear_core


def test_voltage_simulation_converges_and_forwards_nonlinear_materials(monkeypatch):
    from types import SimpleNamespace

    from fem import FrequencyDomainSimulation
    from fem import NonlinearCoreMaterial

    component = FEMModel(
        circuits=[Circuit("primary")],
        windings=[Winding("primary", circuit="primary")],
    )
    simulation = FrequencyDomainSimulation(component, mesh=_mesh())
    nonlinear_core = NonlinearCoreMaterial(
        relative_permeability=lambda values: np.ones_like(values, dtype=complex),
        conductivity=0.0 + 0.0j,
        temperature=25.0,
        frequency=1e3,
    )
    trial_currents = []
    forwarded_materials = []

    def fake_solve(trial_simulation, *, nonlinear_materials=None):
        trial_current = trial_simulation.model.excitation.circuits["primary"]
        trial_currents.append(trial_current)
        forwarded_materials.append(nonlinear_materials)
        return SimpleNamespace(
            winding_terminal_voltages=lambda: {"primary": 2.0 * trial_current}
        )

    monkeypatch.setattr(FrequencyDomainSimulation, "solve", fake_solve)
    result = simulation.solve_voltage(
        10.0,
        winding="primary",
        nonlinear_materials={"core": nonlinear_core},
        relaxation=1.0,
    )

    assert result.winding_terminal_voltages()["primary"] == 10.0
    assert trial_currents == [1.0 + 0.0j, 5.0 + 0.0j]
    assert all(materials == {"core": nonlinear_core} for materials in forwarded_materials)


def test_voltage_simulation_rejects_zero_or_unknown_winding():
    import pytest

    from fem import FrequencyDomainSimulation

    component = FEMModel(
        circuits=[Circuit("primary")],
        windings=[Winding("primary", circuit="primary")],
    )
    simulation = FrequencyDomainSimulation(component, mesh=_mesh())

    with pytest.raises(ValueError, match="non-zero"):
        simulation.solve_voltage(0.0, winding="primary")
    with pytest.raises(ValueError, match="unknown winding"):
        simulation.solve_voltage(1.0, winding="secondary")


def test_transformer_report_separates_losses_and_skips_zero_current_inductance(monkeypatch):
    from fem.simulation import FrequencyDomainResult

    mesh = _mesh()
    mesh.conductor_regions["conductor_1"] = np.arange(4)
    component = FEMModel(
        windings=[
            Winding("primary", turns=[TurnPlacement(center=[1.5, 0.5])]),
            Winding("secondary", turns=[TurnPlacement(center=[1.5, 0.5])]),
        ],
    )
    component.excitation.circuits = {"primary": 4.5, "secondary": 0.0}
    result = FrequencyDomainResult.from_solution(
        component, mesh, np.zeros(5),
        {"conductor_0": np.ones(4), "conductor_1": np.full(4, 2.0)},
    )
    monkeypatch.setattr(
        FrequencyDomainResult, "winding_flux_linkages",
        lambda self: {"primary": 9e-6, "secondary": 9e-6},
    )
    monkeypatch.setattr(FrequencyDomainResult, "winding_voltages", lambda self: {})
    monkeypatch.setattr("fem.report.build_result_report", lambda *args, **kwargs: kwargs)
    report = result.report()
    assert report["inductance_h"] == {"primary": 2e-6}
    losses = report["solid_losses"]
    assert set(losses) == {"primary", "secondary"}
    assert len(losses["primary"].per_turn) == len(losses["secondary"].per_turn) == 1
    assert np.isclose(losses["secondary"].total, 4.0 * losses["primary"].total)


def test_result_report_forwards_complex_terminal_current(monkeypatch):
    from fem.simulation import FrequencyDomainResult

    component = FEMModel(
        circuits=[Circuit("primary")],
        windings=[
            Winding(
                "primary",
                turns=[TurnPlacement(center=[1.5, 0.5])],
                circuit="primary",
            )
        ],
    )
    component.excitation.circuits = {"primary": 1.0 + 2.0j}
    result = FrequencyDomainResult.from_solution(
        component,
        _mesh(),
        np.zeros(5),
        {"conductor_0": np.ones(4, dtype=complex)},
    )
    monkeypatch.setattr(result, "winding_flux_linkages", lambda: {"primary": 3.0 + 4.0j})
    monkeypatch.setattr(result, "winding_voltages", lambda: {})
    monkeypatch.setattr(result, "winding_losses", lambda: {"primary": object()})
    monkeypatch.setattr(result, "core_losses", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(result, "magnetic_flux_density", lambda: np.zeros((4, 2)))
    monkeypatch.setattr("fem.report.build_result_report", lambda *args, **kwargs: kwargs)

    report = result.report()

    assert report["currents_a"] == {"primary": 1.0 + 2.0j}


def test_loss_summary_does_not_double_count_transport_and_proximity(monkeypatch):
    from fem import FrequencyDomainResult, LitzLoss, WindingLosses

    result = FrequencyDomainResult.from_solution(FEMModel(), _mesh(), np.zeros(5), {})
    monkeypatch.setattr(FrequencyDomainResult, "winding_losses", lambda self: {
        "primary": WindingLosses((LitzLoss(1.0, 2.0),)),
        "secondary": WindingLosses((LitzLoss(3.0, 4.0),)),
    })
    monkeypatch.setattr(FrequencyDomainResult, "core_losses", lambda self: (5.0, 6.0, 11.0))
    summary = result.loss_summary()
    assert summary["primary_copper"] == 3.0
    assert summary["secondary_copper"] == 7.0
    assert summary["primary_transport"] == 1.0
    assert summary["primary_proximity"] == 2.0
    assert summary["total"] == 21.0


def test_analytical_dc_winding_resistances_sums_bundle_turns_and_parallel_paths():
    from fem import analytical_dc_winding_resistances

    conductivity_s_per_m = 58e6
    component = FEMModel(
        materials=[Material("copper", conductivity=conductivity_s_per_m)],
        conductors=[
            Conductor(
                "litz",
                construction="litz",
                strands=10,
                strand_diameter=0.001,
            ),
        ],
        windings=[
            Winding(
                "primary",
                conductor="litz",
                parallel_paths=2,
                turns=[
                    TurnPlacement(center=[0.01, 0.0], conductor="litz"),
                    TurnPlacement(center=[0.02, 0.0], conductor="litz"),
                ],
            ),
        ],
    )

    resistance_ohm = analytical_dc_winding_resistances(
        component,
    )
    copper_area_m2 = 10 * np.pi * 0.001**2 / 4.0
    expected_ohm = (2.0 * np.pi * (0.01 + 0.02)) / (
        conductivity_s_per_m * copper_area_m2 * 2**2
    )
    assert np.isclose(resistance_ohm["primary"], expected_ohm)


def test_analytical_dc_winding_resistances_requires_litz_data():
    import pytest
    from fem import analytical_dc_winding_resistances

    component = FEMModel(
        materials=[Material("copper", conductivity=58e6)],
        conductors=[Conductor("copper", construction="solid", radius=0.001)],
        windings=[
            Winding(
                "primary",
                conductor="copper",
                turns=[TurnPlacement(center=[0.01, 0.0], conductor="copper")],
            ),
        ],
    )
    with pytest.raises(ValueError, match="requires defined Litz strand data"):
        analytical_dc_winding_resistances(component)


def test_analytical_dc_winding_resistances_requires_material_conductivity():
    import pytest

    from fem import analytical_dc_winding_resistances

    component = FEMModel(
        materials=[Material("copper", conductivity=0.0)],
        conductors=[
            Conductor(
                "litz",
                construction="litz",
                strands=10,
                strand_diameter=0.001,
                material="copper",
            ),
        ],
        windings=[
            Winding(
                "primary",
                conductor="litz",
                turns=[TurnPlacement(center=[0.01, 0.0], conductor="litz")],
            ),
        ],
    )
    with pytest.raises(ValueError, match="requires a material with finite, positive conductivity"):
        analytical_dc_winding_resistances(component)


def test_conductor_interior_growth_rejects_invalid_factors():
    import pytest

    for growth in (0.5, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="conductor_max_growth"):
            build_mesh(FEMModel(), conductor_max_growth=growth)


def test_conductor_interior_mesh_growth_is_limited_without_refining_core():
    from model import InsulationSpec, RadialWindingSpec, build_concentric_model

    conductor_radius = 0.002
    boundary_size = 0.00025
    component = build_concentric_model(
        core_area=100e-6,
        windings=(RadialWindingSpec("primary", RoundConductor("copper", radius=conductor_radius), 1),),
        insulation=InsulationSpec(0.001, 0.001, 0.001, 0.001),
    )
    mesh = build_mesh(component, max_size=0.003, conductor_geometry_size=boundary_size,
                      physics_adaptive=False, conductor_max_growth=1.5)
    centers = mesh.points[mesh.triangles].mean(axis=1)
    conductor_center = component.windings[0].turns[0].center
    central_elements = np.linalg.norm(centers - conductor_center, axis=1) < conductor_radius / 2
    edge_lengths = np.linalg.norm(
        mesh.points[mesh.triangles] - np.roll(mesh.points[mesh.triangles], 1, axis=1), axis=2,
    )
    assert central_elements.any()
    assert np.quantile(edge_lengths[central_elements], 0.95) < 1.8 * boundary_size
    assert edge_lengths[np.asarray(mesh.regions) == "core"].max() > 3 * boundary_size


def test_permeability_conversion_round_trip():
    relative = np.array([1.0, 1500.0, 2.5 + 0.1j])
    absolute = relative_to_absolute_permeability(relative)
    assert np.allclose(absolute, MU0 * relative)
    assert np.allclose(absolute_to_relative_permeability(absolute), relative)


def test_high_level_litz_solver_respects_parallel_current_share():
    from fem import FrequencyDomainSimulation

    mesh = _mesh()
    mesh.geometry["conductor_windings"] = ["secondary"]
    mesh.geometry["conductor_circuits"] = ["secondary"]
    mesh.geometry["conductor_signs"] = [0.25]
    component = FEMModel(
        conductors=[Conductor(
            "litz", construction="litz", radius=0.4, strands=100, strand_diameter=0.01,
        )],
        windings=[Winding(
            "secondary", circuit="secondary", conductor="litz",
            turns=[TurnPlacement([1.5, 0.5], "litz", "secondary", current_sign=0.25)],
        )],
    )
    component.excitation.frequency = 10_000.0
    component.excitation.circuits = {"secondary": 4.0}
    result = FrequencyDomainSimulation(component, mesh=mesh).solve()
    assert np.isclose(integrated_current(mesh, result.current_densities["conductor_0"], "conductor_0"), 1.0)
    assert set(result.litz_properties) == {"conductor_0"}
    assert result.turn_voltages == {}
    np.testing.assert_allclose(
        result.magnetic_field_intensity(),
        result.magnetic_flux_density() / result.litz_properties["conductor_0"].absolute_permeability,
    )


def test_result_plot_exposes_and_forwards_color_scale():
    from inspect import Parameter, signature

    import pytest
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

    from fem import FrequencyDomainResult, PhysicalField

    parameter = signature(FrequencyDomainResult.plot_physical_field).parameters["color_scale"]
    assert parameter.kind is Parameter.KEYWORD_ONLY
    assert parameter.default == "linear"
    result = FrequencyDomainResult.from_solution(
        FEMModel(), _mesh(), np.zeros(5),
        {"conductor_0": np.array([1.0, 2.0, 4.0, 8.0])},
    )
    for scale in ("linear", "log"):
        figure = Figure()
        axis = figure.subplots()
        result.plot_physical_field(PhysicalField.J, ax=axis, color_scale=scale)
        assert isinstance(axis.collections[0].norm, LogNorm) == (scale == "log")
        assert axis.get_xscale() == axis.get_yscale() == "linear"
    with pytest.raises(ValueError, match="color_scale"):
        result.plot_physical_field(PhysicalField.J, color_scale="invalid")
    result.current_densities["conductor_0"][:] = 0.0
    with pytest.raises(ValueError, match="positive value"):
        result.plot_physical_field(PhysicalField.J, color_scale="log")


def test_result_plot_uses_per_conductor_litz_properties_for_loss_density():
    from matplotlib.figure import Figure

    from fem import FrequencyDomainResult, LitzProperties, PhysicalField

    mesh = _mesh()
    result = FrequencyDomainResult.from_solution(
        FEMModel(),
        mesh,
        np.zeros(5),
        {"conductor_0": np.ones(4, dtype=complex)},
    )
    result.frequency_hz = 100.0
    result.litz_properties = {
        "conductor_0": LitzProperties(
            absolute_permeability=1.0 - 0.1j,
            resistivity=2.0 + 0.0j,
            conductivity=0.5 + 0.0j,
            dimensionless_frequency=1.0,
        ),
    }
    magnetic_field = np.asarray([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ])

    proximity_figure = Figure()
    proximity_axis = proximity_figure.subplots()
    result.plot_physical_field(
        PhysicalField.MAG_LOSS_DENSITY,
        magnetic_field=magnetic_field,
        ax=proximity_axis,
    )
    proximity_density = np.asarray(proximity_axis.collections[0].get_array())
    assert np.all(proximity_density > 0.0)
    assert np.all(np.diff(proximity_density) > 0.0)

    total_figure = Figure()
    total_axis = total_figure.subplots()
    result.plot_physical_field(
        PhysicalField.LOSS_DENSITY,
        magnetic_field=magnetic_field,
        ax=total_axis,
    )
    total_density = np.asarray(total_axis.collections[0].get_array())
    assert np.all(total_density > proximity_density)
    assert np.all(np.diff(total_density) > 0.0)


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


def test_result_report_preserves_complex_terminal_current(capsys):
    report = build_result_report(
        FEMModel(),
        _mesh(),
        frequency_hz=900_000.0,
        currents_a={"primary": 1.0 + 2.0j},
        current_densities={"conductor_0": np.full(4, 1.0 + 2.0j)},
        flux_density=np.ones((4, 2)),
    )

    current = report.to_dict()["simulation"]["currents_a"]["primary"]
    assert current == {
        "real": 1.0,
        "imaginary": 2.0,
        "magnitude": np.sqrt(5.0),
        "phase_deg": np.degrees(np.arctan2(2.0, 1.0)),
    }
    report.print()
    assert "Current primary: 1.000 + j2.000 A" in capsys.readouterr().out


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


def test_parallel_litz_report_uses_litz_losses():
    from fem import FrequencyDomainSimulation
    from model import Constructor, CoreSpec, InsulationSpec

    core = CoreSpec("core", "core", 0.025, 0.025, 0.012, 0.020, 0.020)
    winding = Constructor.place_winding(
        core=core, insulation=InsulationSpec(w2w=0.0005), name="secondary",
        conductor=RoundConductor("litz", radius=0.00165, construction="litz",
                                 strands=600, strand_diameter=0.0001),
        circuit="secondary", turns=1, radial_columns=1, parallel_paths=4,
    )
    component = Constructor(core, (winding,)).build()
    component.excitation.frequency = 200e3
    component.excitation.circuits = {"secondary": 4.0}
    simulation = FrequencyDomainSimulation(component)
    simulation.create_mesh(max_size=0.003)
    result = simulation.solve()
    report = result.report().to_dict()
    losses = report["losses"]["windings"]["secondary"]
    assert losses["litz"]["total_w"] == result.winding_losses()["secondary"].total
    assert not losses.get("solid")
    assert all(item["current_error_a"] < 1e-8 for item in report["conductors"].values())


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


def test_litz_turn_voltage_uses_effective_conductor_conductivity():
    mesh = _mesh()
    field = np.zeros(5, dtype=complex)
    densities = {"conductor_0": np.ones(4, dtype=complex)}
    bulk_conductivity_voltage = winding_voltages(
        mesh, field, densities, 1.0, 0.0, conductivity=10.0,
    )["conductor_0"]
    litz_conductivity_voltage = winding_voltages(
        mesh,
        field,
        densities,
        1.0,
        0.0,
        conductivity=10.0,
        conductor_conductivities={"conductor_0": 5.0 + 1.0j},
    )["conductor_0"]
    assert np.isclose(
        litz_conductivity_voltage,
        bulk_conductivity_voltage * 10.0 / (5.0 + 1.0j),
    )


def test_winding_terminal_voltage_and_complex_power_for_parallel_paths(monkeypatch):
    from fem import FrequencyDomainResult

    mesh = _mesh()
    mesh.geometry.update({
        "p_cond_center_1": [[1.5, 0.5]] * 4,
        "conductor_windings": ["secondary"] * 4,
        "conductor_current_fractions": [0.5] * 4,
    })
    component = FEMModel(windings=[
        Winding(
            "secondary",
            turns=[TurnPlacement(center=[1.5, 0.5]) for _ in range(4)],
            parallel_paths=2,
        ),
    ])
    component.excitation.circuits = {"circuit_1": 3.0 + 1.0j}
    result = FrequencyDomainResult.from_solution(
        component, mesh, np.zeros(5), {},
    )
    conductor_voltages = {
        f"conductor_{index}": complex(index + 2.0) for index in range(4)
    }
    monkeypatch.setattr(result, "winding_voltages", lambda: conductor_voltages)

    terminal_voltages = result.winding_terminal_voltages()
    assert terminal_voltages == {"secondary": 7.0 + 0.0j}
    assert result.winding_complex_powers() == {"secondary": 10.5 - 3.5j}
    assert winding_terminal_voltages(mesh, conductor_voltages) == terminal_voltages
    assert winding_complex_powers(terminal_voltages, {"secondary": 3.0 + 1.0j}) == {
        "secondary": 10.5 - 3.5j,
    }


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


def test_parallel_current_targets_and_flux_linkages():
    mesh = _mesh()
    mesh.geometry["conductor_windings"] = ["secondary"]
    field, _ = solve_frequency_domain(mesh, frequency=1e4, currents=4.0)
    reference = winding_flux_linkages(mesh, field)["secondary"]
    mesh.geometry["conductor_current_fractions"] = [0.25]
    for currents in (4.0, {"secondary": 4.0}):
        _, densities = solve_frequency_domain(mesh, frequency=1e4, currents=currents)
        assert np.isclose(integrated_current(mesh, densities["conductor_0"], "conductor_0"), 1.0)
    assert np.isclose(winding_flux_linkages(mesh, field)["secondary"], reference / 4.0)


def test_parallel_litz_simulation_splits_terminal_current():
    from fem import FrequencyDomainSimulation
    from model import Constructor, CoreSpec, InsulationSpec

    core = CoreSpec("core", "core", 0.025, 0.025, 0.012, 0.020, 0.020)
    winding = Constructor.place_winding(
        core=core, insulation=InsulationSpec(w2w=0.0005), name="secondary",
        conductor=RoundConductor("litz", radius=0.00165, construction="litz",
                                 strands=600, strand_diameter=0.0001),
        circuit="secondary", turns=1, radial_columns=1, parallel_paths=4,
    )
    component = Constructor(core, (winding,)).build()
    component.excitation.frequency = 200e3
    component.excitation.circuits = {"secondary": 4.0}
    simulation = FrequencyDomainSimulation(component)
    mesh = simulation.create_mesh(max_size=0.003)
    result = simulation.solve()
    assert len(result.litz_properties) == 4
    for name, density in result.current_densities.items():
        assert np.isclose(integrated_current(mesh, density, name), 1.0)
    losses = result.winding_losses()["secondary"]
    assert len(losses.per_turn) == 4
    assert losses.transport > 0.0
    assert losses.proximity > 0.0


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
