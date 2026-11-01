"""Dependency-light 2-D axisymmetric magnetostatic FEM package."""

from .combined_solver import solve_combined
from .frequency_solver import (
                          average_abs_current_density,
                          compare_domainc_blocks,
                          copper_loss,
                          integrated_current,
                          solve_frequency_domain,
)
from .litz import LitzLoss, LitzProperties, litz_loss, meeker_litz_properties
from .materials import (
                          MU0,
                          absolute_to_relative_permeability,
                          relative_to_absolute_permeability,
)
from .nonlinear import NonlinearCoreMaterial, material_from_materialdatabase
from .nonlinear_solver import solve_nonlinear_frequency_domain
from .physics import (
                          complex_conductivity,
                          complex_relative_permeability,
                          core_loss_density,
                          eddy_current_density,
                          electric_field_from_a_phi,
                          electric_loss_density,
                          litz_transport_loss_density,
                          magnetic_loss_density,
                          total_litz_loss_density,
)
from .plot import PhysicalField, plot_mesh, plot_physical_field
from .postprocess import (
                          WindingLosses,
                          core_loss,
                          core_losses,
                          inductance,
                          magnetic_energy,
                          magnetic_field_intensity,
                          magnetic_flux_density,
                          magnetic_flux_density_nodal,
                          winding_copper_losses,
                          winding_flux_linkage,
                          winding_flux_linkages,
                          winding_litz_losses,
                          winding_voltages,
)
from .preprocess import AxisymmetricMesh, build_mesh, load_geometry, mesh_from_json
from .report import SimulationResultReport, build_result_report
from .simulation import FrequencyDomainResult, FrequencyDomainSimulation
from .solver import (
                          conductor_area,
                          conductor_current,
                          current_density_for_conductors,
                          solve_axisymmetric,
                          solve_json,
)
from .voltage_solver import solve_voltage_excitation

__all__ = [
                          "MU0",
                          "AxisymmetricMesh",
                          "FrequencyDomainResult",
                          "FrequencyDomainSimulation",
                          "LitzLoss",
                          "LitzProperties",
                          "NonlinearCoreMaterial",
                          "PhysicalField",
                          "SimulationResultReport",
                          "WindingLosses",
                          "absolute_to_relative_permeability",
                          "average_abs_current_density",
                          "build_mesh",
                          "build_result_report",
                          "compare_domainc_blocks",
                          "complex_conductivity",
                          "complex_relative_permeability",
                          "conductor_area",
                          "conductor_current",
                          "copper_loss",
                          "core_loss",
                          "core_loss_density",
                          "core_losses",
                          "current_density_for_conductors",
                          "eddy_current_density",
                          "electric_field_from_a_phi",
                          "electric_loss_density",
                          "inductance",
                          "integrated_current",
                          "litz_loss",
                          "litz_transport_loss_density",
                          "load_geometry",
                          "magnetic_energy",
                          "magnetic_field_intensity",
                          "magnetic_flux_density",
                          "magnetic_flux_density_nodal",
                          "magnetic_loss_density",
                          "material_from_materialdatabase",
                          "meeker_litz_properties",
                          "mesh_from_json",
                          "plot_mesh",
                          "plot_physical_field",
                          "relative_to_absolute_permeability",
                          "solve_axisymmetric",
                          "solve_combined",
                          "solve_frequency_domain",
                          "solve_json",
                          "solve_nonlinear_frequency_domain",
                          "solve_voltage_excitation",
                          "total_litz_loss_density",
                          "winding_copper_losses",
                          "winding_flux_linkage",
                          "winding_flux_linkages",
                          "winding_litz_losses",
                          "winding_voltages",
]
