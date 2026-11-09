# Repository Instructions

- This repository owns the reusable `pymagfem` model, mesh, solver, post-processing, loss, and plotting APIs. Put general electromagnetic functionality here; keep project- and design-specific dimensions and choices in consuming repositories.
- Preserve the public `pymagfem` namespace and backward compatibility for existing `fem` and `model` imports. When changing model schemas or serialized metadata, preserve loading of older models and add round-trip coverage.
- Treat `FEMModel`, schema dataclasses, and low-level object-oriented constructors as the internal representation. Keep consumer documentation and standard examples centered on high-level model factories and simulation facades; use lower-level constructors for implementation, tests, or geometry the factories cannot express.
- Keep electrical series turns distinct from physical conductor bundles. `parallel_paths` means ideal equal-current sharing at the terminals; current density, flux linkage, inductance, and losses must use consistent terminal/bundle conventions. Litz loss totals include every physical bundle.
- Keep winding order explicit. Insulation matrices are symmetric, ordered by winding order, and expressed in metres. Mesh controls are target sizes; document scope and limitations rather than promising a strict maximum element edge.
- Add focused regression tests for API and numerical behavior. Run `python -m pytest` from the repository root; the project targets Python 3.10+ and uses Ruff with a 100-character line length.
- Avoid unrelated changes to notebooks, model data, generated meshes, or simulation results. Use the existing tests and examples as the canonical API usage patterns.
