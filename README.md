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

Run the test suite with:

```bash
python -m pytest
```

Runtime dependencies are listed in [`requirements.txt`](requirements.txt),
development dependencies in [`requirements-dev.txt`](requirements-dev.txt), and
package metadata in [`pyproject.toml`](pyproject.toml).
