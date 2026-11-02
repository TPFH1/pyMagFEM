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