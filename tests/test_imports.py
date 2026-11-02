"""Tests for the public package namespace."""

import fem
import model
import pymagfem


def test_public_namespace_preserves_existing_packages():
    assert pymagfem.fem is fem
    assert pymagfem.model is model


def test_public_namespace_exports_packages():
    from pymagfem import fem as public_fem, model as public_model

    assert public_fem.FrequencyDomainSimulation is fem.FrequencyDomainSimulation
    assert public_model.FEMModel is model.FEMModel