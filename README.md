# pyMagFEM

A compact Python toolkit for 2-D axisymmetric electromagnetic finite-element
analysis.

## Overview

`pyMagFEM` models rotationally symmetric magnetic devices such as inductors,
transformers, and magnetic cores. It provides a consistent workflow from
parameterized geometry and materials to meshing, field solution, and physical
quantities such as flux density, energy, inductance, and winding losses.

Supported analyses include:

- magnetostatic and frequency-domain field problems;
- voltage-driven and current-driven windings;
- nonlinear and complex magnetic materials;
- copper, eddy-current, and Litz-wire loss calculations;
- mesh and field visualization.

The formulation uses the azimuthal vector potential in an axisymmetric weak
form. Geometry is meshed with Gmsh through `pygmsh`, while numerical assembly
uses NumPy and SciPy.

## Installation

Python 3.10 or newer is required.

```bash
python -m pip install -e .
```

Install the notebook, test, and development tools with:

```bash
python -m pip install -e ".[dev]"
```

## Quick start

Import the solver and model APIs through the public package namespace:

```python
import pymagfem
from pymagfem import fem, model
```

The existing top-level `fem` and `model` imports remain supported.
After updating an existing editable installation, rerun
`python -m pip install -e .` and restart any running notebook kernels.

### Equal-Current Parallel Windings

Use the high-level `model.RadialWindingSpec` and
`model.build_concentric_model` APIs to describe windings. For example, a
specification with `turns=2` and `parallel_paths=4` places eight physical
conductor bundles representing two series turns and four ideal parallel
paths. Excitation currents are terminal currents: each physical bundle carries
one quarter of the winding current. Flux linkage is weighted by the same
factor, so inductance matrices refer to the winding terminals. The default
`parallel_paths=1` retains series-winding behaviour.

This assumes perfect equal sharing, not an electrical network solution for
unequal branch currents or circulating currents. Physical bundle placements
must be divisible by the parallel-path count.

Litz wires use the existing `fem.litz` Meeker model. Specify `construction="litz"`,
`strands`, and the bare `strand_diameter` on a `RoundConductor`. The high-level
simulation derives each wire's fill factor from its strand and envelope areas.
`result.winding_losses()` returns winding-level transport/skin and proximity
losses, summing every physical bundle; `result.report()` uses the same losses.

### Inductance Matrix

For a constructed component, extract the complex inductance matrix in henries
using one open-circuit excitation per winding:

```python
matrix_h = fem.inductance_matrix(component)
```

Rows are linked windings and columns are driven windings, in model winding
order by default. The model's frequency and materials are preserved, and all
cases share a mesh. Windings must have distinct circuits. To reuse an existing
mesh and retain the solved fields:

```python
matrix_h, cases = fem.inductance_matrix(
	component,
	mesh=mesh,
	winding_order=["primary", "secondary"],
	test_current=1.0,
	return_results=True,
)
primary_driven_result = cases["primary"]
```

Field plots accept an explicit colour scale on both the result method and
the low-level plotting function:

```python
result.plot_physical_field(fem.PhysicalField.J, color_scale="linear")
result.plot_physical_field(fem.PhysicalField.J, color_scale="log")
```

The default is `"linear"`. Logarithmic scaling applies to field colours and
the colourbar, not the spatial axes. It masks non-positive values and requires
at least one positive value.

The repository includes executable examples in [`notebooks/`](notebooks/).
Start Jupyter from the repository root:

```bash
python -m jupyter lab
```

The minimal inductor example is a good starting point:
[`notebooks/example_inductor_minimal.ipynb`](notebooks/example_inductor_minimal.ipynb).
[`notebooks/example_inductor.ipynb`](notebooks/example_inductor.ipynb) demonstrates
the high-level inductor factory, compares current-fed and voltage-fed results,
and lets you select N49 or a custom linear core. The transformer workflow is shown in
[`notebooks/example_transformer.ipynb`](notebooks/example_transformer.ipynb).

## Model and solver structure

The consumer-facing workflow is the high-level model factory and simulation
facade described below. Internally, models are represented by validated,
versioned dataclasses in [`model/schema.py`](model/schema.py); these schema
objects and low-level constructors are implementation structures, not the
preferred application interface. The internal model separates geometry,
regions, materials, conductors, windings, circuits, excitation, mesh, solver,
and output settings.

The main packages are:

- [`model/`](model/): model schema, validation, normalization, and geometry constructors;
- [`fem/preprocess.py`](fem/preprocess.py): geometry conversion and Gmsh meshing;
- [`fem/solver.py`](fem/solver.py): axisymmetric FEM assembly and solution;
- [`fem/frequency_solver.py`](fem/frequency_solver.py): complex frequency-domain analysis;
- [`fem/nonlinear_solver.py`](fem/nonlinear_solver.py): field-dependent material analysis;
- [`fem/postprocess.py`](fem/postprocess.py): field, energy, flux-linkage, and loss quantities;
- [`fem/plot.py`](fem/plot.py): mesh and field visualization.

### Advanced Magnetostatic Solve

For static analyses, the lower-level solver accepts a mesh and current-density
map. This is an advanced path; frequency-domain application workflows should
prefer `fem.FrequencyDomainSimulation`.

```python
component = model.load(model_data)
mesh = fem.build_mesh(component, max_size=mesh_size_m)
a_phi = fem.solve_axisymmetric(
	mesh,
	relative_permeability={"core": core_relative_permeability},
	current_density={"winding": current_density_a_per_m2},
)
flux_density = fem.magnetic_flux_density(mesh, a_phi)
```

## Materials

`pymagfem.fem` exports reference copper properties: `COPPER_CONDUCTIVITY_AT_20C_S_PER_M`,
`COPPER_REFERENCE_TEMPERATURE_C`, and `COPPER_TEMPERATURE_COEFFICIENT_PER_C`.
`fem.copper_conductivity_at_temperature_s_per_m(temperature_c)` applies those
defaults using a linear temperature coefficient. Override the keyword-only
reference parameters when measured material data is available. Solvers do not
automatically apply the copper temperature correction; callers select and
assign the resulting conductivity to their material model.

```python
copper_conductivity_s_per_m = fem.copper_conductivity_at_temperature_s_per_m(140.0)
```

Material APIs use relative permeability. Internal calculations convert it to
absolute permeability in H/m using
$\mu=\mu_0\mu_r$. Nonlinear material adapters expose both relative and absolute
permeability consistently.

## Development

The following sections describe lower-level implementation and extension
details. Consumer applications should use the high-level model factory and
simulation APIs above; use schema and constructor objects directly only when
extending pyMagFEM or when a factory cannot express the required geometry.

### Winding Insulation Matrix

`model.InsulationSpec` accepts an optional symmetric `w2w_matrix` in metres,
ordered like the windings passed to the constructor or concentric factory:

```python
insulation = model.InsulationSpec(
	c2w_left=core_clearance_m,
	c2w_right=outer_clearance_m,
	c2w_top=core_clearance_m,
	c2w_bottom=core_clearance_m,
	w2w_matrix=[
		[ins_11, ins_12],
		[ins_12, ins_22],
	],
)
```

`ins_11` and `ins_22` set physical bundle spacing within each winding.
`ins_12` sets the primary-to-secondary envelope clearance. Diagonal values
control axial pitch/stack height; adjacent off-diagonal values control radial
spacing and window width. The constructor also validates pairwise clearances.
The matrix must be finite, non-negative, square, symmetric, and match the
winding count. Saved models record it and its winding order in core metadata.

Without a matrix, scalar `w2w` remains supported and applies to all pairs.
For `Constructor.place_winding` with a matrix, `winding_index` selects its
diagonal; `build_concentric_model` selects the entries automatically.

### Conductor Mesh Grading

Use `conductor_max_growth` to limit conductor-interior target sizes relative
to the conductor-boundary geometry size:

```python
mesh = fem.build_mesh(
	component,
	conductor_geometry_size=boundary_size_m,
	conductor_max_growth=1.5,
)
```

The cap is restricted to conductor surfaces and does not impose that size
on the core or surrounding air. A factor of 1 requests a nearly uniform
conductor mesh; 1.5 permits moderate interior growth. This controls target
sizes, not a strict upper bound on every triangle edge. The default `None`
preserves the previous grading. The same keyword is accepted by
`simulation.create_mesh()` and `mesh_from_json()`.

### Higher-Level Model and Analysis APIs

For application code, prefer this factory over manually assembling
`FEMModel`, `CoreSpec`, or `Constructor` objects:

`model.RadialWindingSpec` describes one single-column winding, including
electrical turns, parallel paths, and `bottom`, `center`, or `top` alignment.
Pass windings in inside-to-outside order to `model.build_concentric_model`:

```python
component = model.build_concentric_model(
	core_area=core_area_m2,
	windings=winding_specs,
	insulation=insulation,
	air_gap_length=air_gap_length_m,
)
```

The factory derives the window from the physical bundle stacks and builds
an equal-area centre/outer-leg core with equal minimum yoke cross section.
Dimensions, materials, and excitation frequency belong to the caller.

```python
equivalent = fem.transformer_equivalent(
	matrix_h, secondary_series_inductance_h=external_secondary_inductance_h,
)
equivalent.print()
parameters = equivalent.to_dict()
result = fem.solve_operating_point(
	component, mesh=mesh, frequency_hz=frequency_hz, currents=terminal_phasors,
)
losses_w = result.loss_summary()
```

For voltage-driven operation, use `simulation.solve_voltage(voltage_v,
winding="primary", nonlinear_materials=...)`. It iteratively adjusts the
terminal current to meet the requested voltage phasor, reuses the simulation
mesh, and supports the same nonlinear core materials as `solve()`.

The inductive T-model uses `sqrt(L11/L22)` and a primary-side magnetizing
branch. Its external-secondary series term is referred by the ratio squared.
Operating-point solves copy the model instead of overwriting identification
inputs. Current phasors use peak amplitudes for loss calculations; loss
summaries include winding transport/proximity and core contributions without
double-counting their totals.

Run the test suite with:

```bash
python -m pytest
```

Runtime dependencies are listed in [`requirements.txt`](requirements.txt),
development dependencies in [`requirements-dev.txt`](requirements-dev.txt), and
package metadata in [`pyproject.toml`](pyproject.toml).
