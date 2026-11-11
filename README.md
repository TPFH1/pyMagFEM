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

Use a high-level winding scheme to arrange sections, then let the wrapper
derive chamber and physical window dimensions from the winding envelope:

```python
insulation = model.InsulationSpec(
	c2w_left=core_clearance_m,
	c2w_right=outer_clearance_m,
	c2w_top=core_clearance_m,
	c2w_bottom=core_clearance_m,
)
primary = model.WindingDefinition("primary", primary_conductor, turns=8)
secondary = model.WindingDefinition("secondary", secondary_conductor, turns=2)
windings = (primary, secondary)

layout = model.RadialStack(
	sections=(
		model.WindingSection(
			"primary", conductor_count=8, parallel_paths=1, layers=1,
			h_loc=model.Align.Left, v_loc=model.Align.Center,
		),
		model.WindingSection(
			"secondary", conductor_count=8, parallel_paths=4, layers=1,
			h_loc=model.Align.Right, v_loc=model.Align.Center,
		),
	),
	split_ins=interwinding_insulation_m,
).layout(
	core_area=core_area_m2,
	windings=windings,
	insulation=insulation,
)
print(layout.window_outer_radius_m, layout.window_height_m)
component = layout.build_model()
fem.plot_winding_scheme(component)
```

For an axial scheme, supply sections in top-to-bottom order to
`model.AxialStack`; for example, primary then secondary places the primary
above the secondary. Either wrapper supports repeated winding sections, such
as primary / secondary / primary. `WindingSection.layers` is the radial
winding-layer count within that section, not the number of chambers or the
number of parallel paths. The wrappers return both usable chamber dimensions
and physical window dimensions; core-to-winding clearances are included when
the physical window is sized.

Electrical turn counts are defined once by `WindingDefinition` and apportioned
across its sections. Each section specifies an integer `conductor_count` (the
physical conductor bundles in that chamber), `parallel_paths`, and `layers`.
Section conductor counts for one winding must sum to its electrical turns times
its path count, and path counts must agree across sections. The compiler
distributes path membership across sections so every path has the declared
series-turn count.

Paths share current equally at the terminals. Two series turns with four paths
create eight physical bundles, each carrying one quarter of the terminal
current.

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
ordered like the winding definitions:

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

`ins_11` and `ins_22` can be used as same-winding conductor spacing when
building chamber assignments. Split insulation is specified explicitly on
`split_chamber`; it is not inferred from assignment order. The matrix must be
finite, non-negative, square, symmetric, and match the winding count.

Without a matrix, scalar `w2w` remains supported and applies to all pairs.
Chamber assignments use the matching matrix diagonal for same-winding bundle
spacing by default. `WindingDefinition.turn_insulation` can override that
spacing for a particular winding.

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

The scheme wrappers are the application-level geometry constructors.
`RadialStack` orders sections inner-to-outer; `AxialStack` orders sections
top-to-bottom. Each section selects its winding, integer `conductor_count`,
parallel paths, radial `layers`, `h_loc`, and `v_loc`. The conductor counts
across a winding's sections must total its electrical turns multiplied by the
parallel-path count. The compiler allocates conductors across paths in section
order so each path retains the declared series-turn count. The wrappers create
placement-only chambers, derive the physical window from the required winding
envelope and clearances, then compile to the regular FEM model. Direct
`split_chamber` and `build_chambered_model` remain available for arrangements
that do not fit these common scheme patterns.

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

`fem.transformer_equivalent()` accepts four `equivalent_type` values:
`"balanced_leakage"` uses `sqrt(L11/L22)` and remains the backward-compatible
default; `"physical_turns_ratio"` requires the positive `physical_turns_ratio`
argument; `"primary_concentrated_leakage"` selects the ratio that makes the
secondary leakage zero; and `"secondary_concentrated_leakage"` selects the
ratio that makes the primary leakage zero. Each mode reconstructs the same
real inductance matrix, but reports a different referral ratio and leakage
split. The physical mode is appropriate when referring terminal quantities by
the actual winding ratio. The balanced mode is convenient for equal referred
leakage; the concentrated modes show the limiting all-on-one-side models.
Negative leakage in a selected mode indicates that the chosen ratio does not
give a passive branch-by-branch T-model for that matrix. The optional external
secondary series term is referred by the selected ratio squared.

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
