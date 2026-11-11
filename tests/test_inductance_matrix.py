from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from fem import FrequencyDomainSimulation, inductance_matrix
from model import FEMModel, Winding


@pytest.fixture
def matrix_case(monkeypatch):
    component = FEMModel(windings=[
        Winding("primary", circuit="primary_circuit"),
        Winding("secondary", circuit="secondary_circuit"),
    ])
    component.excitation.circuits = {"primary_circuit": 4.5, "secondary_circuit": 0.0}
    expected = np.array([[3 - 0.1j, 2 - 0.2j], [2.01 - 0.21j, 4 - 0.3j]]) * 1e-6
    shared_mesh = object()
    calls = []

    def solve(simulation):
        assert simulation.mesh is shared_mesh
        currents = simulation.model.excitation.circuits
        calls.append(dict(currents))
        linkages = expected @ np.array([
            currents["primary_circuit"], currents["secondary_circuit"],
        ])
        return SimpleNamespace(winding_flux_linkages=lambda: dict(zip(
            ["primary", "secondary"], linkages,
        )))

    monkeypatch.setattr(FrequencyDomainSimulation, "solve", solve)
    monkeypatch.setattr(FrequencyDomainSimulation, "create_mesh", lambda self: shared_mesh)
    return component, shared_mesh, expected, calls


def test_inductance_matrix_normalizes_excitation_and_preserves_model(matrix_case):
    component, shared_mesh, expected, calls = matrix_case
    original = deepcopy(component)
    matrix, results = inductance_matrix(
        component, mesh=shared_mesh, test_current=2.5, return_results=True,
    )
    np.testing.assert_allclose(matrix, expected)
    assert component == original
    assert list(results) == ["primary", "secondary"]
    assert calls == [
        {"primary_circuit": 2.5, "secondary_circuit": 0.0},
        {"primary_circuit": 0.0, "secondary_circuit": 2.5},
    ]


def test_inductance_matrix_creates_mesh_and_respects_order(matrix_case):
    component, _, expected, calls = matrix_case
    matrix = inductance_matrix(component, winding_order=["secondary", "primary"])
    np.testing.assert_allclose(matrix, expected[::-1, ::-1])
    assert len(calls) == 2


@pytest.mark.parametrize("test_current", [0.0, float("nan"), float("inf")])
def test_inductance_matrix_rejects_invalid_current(matrix_case, test_current):
    with pytest.raises(ValueError, match="test_current"):
        inductance_matrix(matrix_case[0], test_current=test_current)


@pytest.mark.parametrize("order", [[], ["primary"], ["primary", "primary"]])
def test_inductance_matrix_rejects_invalid_order(matrix_case, order):
    with pytest.raises(ValueError, match="winding_order"):
        inductance_matrix(matrix_case[0], winding_order=order)


def test_inductance_matrix_rejects_shared_circuits(matrix_case):
    component = matrix_case[0]
    component.windings[1].circuit = component.windings[0].circuit
    with pytest.raises(ValueError, match="distinct"):
        inductance_matrix(component)


def test_transformer_equivalent_reconstructs_matrix_and_refers_external_inductance():
    from fem import transformer_equivalent

    matrix = np.array([[100 - 2j, 18 - 0.3j], [18.2 - 0.4j, 4 - 0.1j]]) * 1e-6
    original = matrix.copy()
    equivalent = transformer_equivalent(matrix, secondary_series_inductance_h=100e-9)
    assert equivalent.turns_ratio == pytest.approx(5.0)
    assert equivalent.external_inductance_referred_primary_h == pytest.approx(2.5e-6)
    reconstructed = np.array([
        [equivalent.leakage_primary_h + equivalent.magnetizing_primary_h,
         equivalent.magnetizing_primary_h / equivalent.turns_ratio],
        [equivalent.magnetizing_primary_h / equivalent.turns_ratio,
         equivalent.leakage_secondary_h + equivalent.magnetizing_primary_h / equivalent.turns_ratio**2],
    ])
    np.testing.assert_allclose(reconstructed, (matrix.real + matrix.real.T) / 2)
    np.testing.assert_array_equal(matrix, original)
    assert equivalent.total_leakage_referred_primary_h == pytest.approx(
        equivalent.transformer_leakage_referred_primary_h + 2.5e-6
    )
    assert equivalent.to_dict()["turns_ratio"] == pytest.approx(5)


@pytest.mark.parametrize(
    ("equivalent_type", "options", "expected_ratio", "zero_leakage"),
    [
        ("physical_turns_ratio", {"physical_turns_ratio": 4.0}, 4.0, None),
        ("balanced_leakage", {}, 5.0, None),
        ("primary_concentrated_leakage", {}, 18.1 / 4, "secondary"),
        ("secondary_concentrated_leakage", {}, 100 / 18.1, "primary"),
    ],
)
def test_transformer_equivalent_supports_ratio_and_leakage_modes(
    equivalent_type, options, expected_ratio, zero_leakage,
):
    from fem import transformer_equivalent

    matrix = np.array([[100 - 2j, 18 - 0.3j], [18.2 - 0.4j, 4 - 0.1j]]) * 1e-6
    equivalent = transformer_equivalent(
        matrix, equivalent_type=equivalent_type, **options,
    )

    assert equivalent.equivalent_type == equivalent_type
    assert equivalent.turns_ratio == pytest.approx(expected_ratio)
    reconstructed = np.array([
        [equivalent.leakage_primary_h + equivalent.magnetizing_primary_h,
         equivalent.magnetizing_primary_h / equivalent.turns_ratio],
        [equivalent.magnetizing_primary_h / equivalent.turns_ratio,
         equivalent.leakage_secondary_h + equivalent.magnetizing_primary_h / equivalent.turns_ratio**2],
    ])
    np.testing.assert_allclose(reconstructed, (matrix.real + matrix.real.T) / 2)
    if zero_leakage == "primary":
        assert equivalent.leakage_primary_h == pytest.approx(0.0, abs=1e-18)
    elif zero_leakage == "secondary":
        assert equivalent.leakage_secondary_h == pytest.approx(0.0, abs=1e-18)


@pytest.mark.parametrize(
    ("equivalent_type", "options", "message"),
    [
        ("unknown", {}, "equivalent_type"),
        ("physical_turns_ratio", {}, "required"),
        ("physical_turns_ratio", {"physical_turns_ratio": 0.0}, "finite and positive"),
        ("balanced_leakage", {"physical_turns_ratio": 2.0}, "only valid"),
    ],
)
def test_transformer_equivalent_rejects_invalid_mode_options(equivalent_type, options, message):
    from fem import transformer_equivalent

    with pytest.raises(ValueError, match=message):
        transformer_equivalent(np.eye(2), equivalent_type=equivalent_type, **options)


def test_transformer_equivalent_concentrated_modes_reject_zero_mutual():
    from fem import transformer_equivalent

    for equivalent_type in ("primary_concentrated_leakage", "secondary_concentrated_leakage"):
        with pytest.raises(ValueError, match="non-zero mutual"):
            transformer_equivalent(np.diag([2.0, 3.0]), equivalent_type=equivalent_type)


@pytest.mark.parametrize("matrix", [np.eye(3), [[0, 1], [1, 1]], [[1, 0], [0, np.nan]]])
def test_transformer_equivalent_rejects_invalid_matrices(matrix):
    from fem import transformer_equivalent

    with pytest.raises(ValueError):
        transformer_equivalent(matrix)


def test_operating_point_preserves_original_excitation(matrix_case):
    from fem import solve_operating_point

    component, shared_mesh, _, calls = matrix_case
    original = deepcopy(component)
    solve_operating_point(component, mesh=shared_mesh, frequency_hz=250e3,
                          currents={"primary_circuit": 1 + 2j, "secondary_circuit": -7j})
    assert component == original
    assert calls == [{"primary_circuit": 1 + 2j, "secondary_circuit": -7j}]


@pytest.mark.parametrize("external", [-1.0, float("nan"), float("inf")])
def test_transformer_equivalent_rejects_invalid_external_inductance(external):
    from fem import transformer_equivalent

    with pytest.raises(ValueError, match="secondary_series"):
        transformer_equivalent(np.eye(2), secondary_series_inductance_h=external)


def test_operating_point_applies_frequency_and_opens_missing_circuits(matrix_case, monkeypatch):
    from fem import solve_operating_point

    component, shared_mesh, _, _ = matrix_case
    original = deepcopy(component)
    monkeypatch.setattr(FrequencyDomainSimulation, "solve",
                        lambda simulation: SimpleNamespace(model=simulation.model))
    result = solve_operating_point(component, mesh=shared_mesh, frequency_hz=250e3,
                                   currents={"primary_circuit": 1 + 2j})
    assert result.model.excitation.frequency == 250e3
    assert result.model.excitation.circuits == {"primary_circuit": 1 + 2j, "secondary_circuit": 0j}
    assert component == original


def test_operating_point_rejects_unknown_circuits(matrix_case):
    from fem import solve_operating_point

    with pytest.raises(ValueError, match="unknown"):
        solve_operating_point(matrix_case[0], frequency_hz=1e3, currents={"missing": 1.0})