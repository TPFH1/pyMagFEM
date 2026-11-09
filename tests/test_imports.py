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


def test_fem_exports_copper_reference_properties():
    assert fem.COPPER_CONDUCTIVITY_AT_20C_S_PER_M == 58e6
    assert fem.COPPER_REFERENCE_TEMPERATURE_C == 20.0
    assert fem.COPPER_TEMPERATURE_COEFFICIENT_PER_C == 0.00393


def test_copper_conductivity_uses_reference_defaults():
    assert fem.copper_conductivity_at_temperature_s_per_m(20.0) == 58e6
    assert fem.copper_conductivity_at_temperature_s_per_m(140.0) == 58e6 / (
        1.0 + 0.00393 * (140.0 - 20.0)
    )


def test_copper_conductivity_accepts_custom_reference_conductivity():
    conductivity_at_20c_s_per_m = 1.0 / 1.72e-8
    actual = fem.copper_conductivity_at_temperature_s_per_m(
        140.0,
        conductivity_at_reference_s_per_m=conductivity_at_20c_s_per_m,
    )
    expected = conductivity_at_20c_s_per_m / (1.0 + 0.00393 * (140.0 - 20.0))
    assert actual == expected


def test_copper_conductivity_rejects_nonpositive_temperature_factor():
    import pytest

    with pytest.raises(ValueError, match="positive conductivity"):
        fem.copper_conductivity_at_temperature_s_per_m(
            21.0,
            temperature_coefficient_per_c=-1.0,
        )