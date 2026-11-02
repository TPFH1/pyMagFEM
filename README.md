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

`Constructor.place_winding(..., turns=2, parallel_paths=4)` places eight
physical conductor bundles representing two series turns and four ideal
parallel paths. Excitation currents are terminal currents: each physical
bundle carries one quarter of the winding current. Flux linkage is weighted
by the same factor, so inductance matrices refer to the winding terminals.
The model, JSON loader, solver, and reports preserve this distinction.
The default `parallel_paths=1` retains series-winding behaviour.

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
Additional notebooks demonstrate static, frequency-domain, voltage-driven,
nonlinear-material, and transformer workflows.

## Model and solver structure

Models are represented by validated, versioned dataclasses in
[`model/schema.py`](model/schema.py). The model separates geometry, regions,
materials, conductors, windings, circuits, excitation, mesh, solver, and
output settings.

The main packages are:

- [`model/`](model/): model schema, validation, normalization, and geometry constructors;
- [`fem/preprocess.py`](fem/preprocess.py): geometry conversion and Gmsh meshing;
- [`fem/solver.py`](fem/solver.py): axisymmetric FEM assembly and solution;
- [`fem/frequency_solver.py`](fem/frequency_solver.py): complex frequency-domain analysis;
- [`fem/nonlinear_solver.py`](fem/nonlinear_solver.py): field-dependent material analysis;
- [`fem/postprocess.py`](fem/postprocess.py): field, energy, flux-linkage, and loss quantities;
- [`fem/plot.py`](fem/plot.py): mesh and field visualization.

Material APIs use relative permeability. Internal calculations convert it to
absolute permeability in H/m using
$\mu=\mu_0\mu_r$. Nonlinear material adapters expose both relative and absolute
permeability consistently.

## Development

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
