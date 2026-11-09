# Generalized Core-Window and Winding Construction Plan

## Scope

Implement the generalized geometry and winding-construction API entirely in `pyMagFEM`. Update the existing consumer notebooks in `hv-magnetics-data` to use the new API after it is implemented. Do not retain or adapt the old `RadialWindingSpec` / `build_concentric_model` API for backward compatibility; the project API is still under development.

Keep the current core model in this phase. The initial physical winding window comes from the core aperture and core-to-winding insulation. Chambers are rectangular placement regions inside that window. Air gaps remain in the center leg. Magnetic shunts are out of scope for this implementation.

## Existing Primitive Assessment

### Regions and Chambers

`model.schema.Region` can hold rectangular bounds as corners, but it represents a material region. It has no parent/child hierarchy, split metadata, usable bounds, insulation policy, or winding assignment. It is therefore not sufficient as the chamber abstraction by itself.

Add a dedicated placement-only `Chamber` (or `ChamberSpec`) type. It should represent an axis-aligned rectangle in the FEM meridian plane, identify its parent and children when split, and carry the construction data needed to reproduce the partition. Chambers are not FEM material regions: their boundaries do not need to be sent to Gmsh unless a later feature assigns physical material to them.

Use the existing core/aperture geometry and clearances to calculate the initial usable chamber. Keep the FEM winding-window region as the actual air aperture; do not shrink the meshed aperture to the usable conductor-placement bounds.

### Conductors

`ConductorSpec`, `RoundConductor`, and `RectangularConductor` provide the required round/rectangular conductor dimensions, physical area, material, and Litz strand data. Their dimension helpers can be reused by the placement engine, and their `to_model()` output can continue to feed the solver schema. They are sufficient for the currently requested bundle shapes; they do not model insulation coatings or arbitrary conductor cross-sections.

Before reusing rectangular conductors, make the shape name consistent end-to-end. `RectangularConductor.to_model()` emits `"rectangular"`, while schema validation accepts `"rectangle"`; preprocessing branches on `"rectangular"`. Choose one canonical value and use it in constructors, schema validation/normalization, meshing, and tests.

The conductor dimension classes should also validate finite positive dimensions. The placement engine must use the full round or rectangular footprint when checking chamber fit and clearances, not just conductor centers.

### Final FEM Representation

Reuse `TurnPlacement`, `Winding`, `Circuit`, `Conductor`, and `FEMModel` for the generated solver model. A chamber layout should compile into explicit physical conductor placements with winding/circuit identity, orientation/sign, and path information. `Circuit.turns` remains the electrical turn count; physical bundle count remains distinct.

`FEMModel.legacy_geometry()` currently produces the flat arrays consumed by FEM preprocessing, including conductor centers, winding/circuit names, signs, and current fractions. Despite its name, this is an active solver boundary, not just an old public API. Keep an equivalent conversion boundary or rename it to a solver-oriented name, and ensure it carries all per-conductor current-sharing information required by the new model.

## Proposed Model Objects

- `Chamber`: stable identifier and rectangular bounds, with optional parent identifier and split provenance.
- `ChamberSplit`: parent chamber, horizontal/vertical direction, split count or ordered relative positions, and `split_ins` in metres.
- `WindingDefinition`: winding identity, conductor, total electrical turns, and electrical/circuit metadata.
- `WindingSection`: one segment of a logical winding placed in a chamber; includes its electrical turn count, parallel paths, radial `layers` count, `h_loc`, `v_loc`, and vertical start direction. A winding may have multiple sections, including repeated sections in one scheme.
- `ChamberWindingAssignment`: compiled placement of one `WindingSection` in a leaf chamber.
- `RadialStack` and `AxialStack`: high-level wrappers that create chamber sections in inside-to-outside or top-to-bottom/bottom-to-top order, respectively.
- `h_loc`: `Align.Left`, `Align.Center`, `Align.Right`, or `Distribute.Horizontally`.
- `v_loc`: `Align.Top`, `Align.Center`, `Align.Bot`, or `Distribute.Vertically`.

A chamber may be split recursively. A returned child chamber can be assigned a winding independently of the split direction. Define enum types and these data structures in `pyMagFEM/model/`, export them through `model.__init__`, and keep all length values in metres.

The scheme's ordered **sections** describe how winding regions are arranged in the window. A section's **layers** describe how that section's physical conductor bundles are stacked radially within its own chamber. `layers` replaces the ambiguous former `radial_columns` name; it is a positive per-section value, not the number of chambers or the number of electrical parallel paths.

`RadialStack` arranges sections inside-to-outside; D2 is primary then secondary. `AxialStack` arranges sections in a selected vertical order; D3 is primary above secondary. Both support any section count and may repeat a logical winding, such as primary / secondary / primary. Repeated sections contribute turns to the same `WindingDefinition` and must pass the existing exact-total and path-consistency checks.

Each wrapper first derives the minimum required bounds of every section from its physical bundle count (`turns * parallel_paths`), conductor footprint, `layers`, and turn insulation. It then composes those section bounds with divider insulation to return a `ChamberLayout` containing the chamber tree, assignments, usable chamber envelope, and required physical winding-window dimensions. For a radial stack, section widths plus divider gaps determine radial extent and the maximum section height determines axial extent. For an axial stack, section heights plus divider gaps determine axial extent and the maximum section width determines radial extent. `h_loc`/`v_loc` position narrower sections within the composed envelope.

The wrapper-derived window dimensions should feed `core_for_window` so callers can omit explicit core-window dimensions. The result must distinguish usable chamber bounds from the physical aperture bounds after core-to-winding insulation is added. Window z-centering and top/bottom clearances must be handled consistently with the current symmetric `CoreSpec` window.

## Splitting Rules

1. Begin with one usable chamber derived from the core winding aperture and core-to-winding clearances.
2. A split takes an axis, either equal partitioning or explicit relative divider positions, and `split_ins`.
3. For equal splitting into `N` children, reserve `N - 1` insulation gaps and divide the remaining span equally.
4. For explicit positions, interpret each percentage as a divider coordinate measured from the low-coordinate edge of the parent chamber along the split axis. Center `split_ins` at each divider, assigning half the gap to each adjacent child.
5. Require positions to be finite, strictly increasing, within the parent, and to leave non-negative child dimensions after insulation. Reject overlapping or degenerate children.
6. Preserve split inputs and resulting child bounds in the model so serialized designs explain their layout.

## Winding Placement Rules

1. Every chamber assignment names one winding and declares its electrical turns and parallel paths. A winding can be assigned across multiple chambers.
2. The sum of electrical turns assigned to all chambers for each winding must equal its declared total exactly. Detect and report both under-allocation and over-allocation before creating the FEM model.
3. Define progression before location: start at the top or bottom, fill vertically in the chosen direction, then advance radially to the next winding layer. Use serpentine traversal by default, reversing horizontal direction on alternate layers.
4. Apply `h_loc` and `v_loc` independently inside each chamber. Alignment positions the packed bundle envelope against the selected side or center while honoring applicable insulation. Distribution spreads bundle centers over the available extent of the selected axis while respecting conductor footprints and required spacing.
5. Split orientation does not imply winding placement orientation. The same placement algorithm applies to any rectangular chamber.
6. Report a placement error with chamber and winding identifiers when a requested number of bundles cannot fit.

### Parallel-Path Semantics

The current model stores `parallel_paths` once per `Winding`; solver current fractions are then `1 / parallel_paths` for every physical conductor. The requested chamber-level setting cannot be implemented only in the constructor: per-conductor current allocation is consumed by frequency-domain current targeting, voltage solving, current-density calculations, flux linkage, DC resistance, winding loss, and reporting.

Represent path membership or current fraction explicitly on generated physical placements, and update all consumers to use that data rather than assuming one global winding value. Keep the existing ideal equal-current-sharing assumption; this feature is not an electrical network solver.

A logical winding split across chambers must have a well-defined electrical path interpretation. Until explicit branch connectivity is modeled, require chamber assignments belonging to one logical winding to have compatible parallel-path topology. Do not silently combine different path counts into one terminal winding. If truly different path counts between chambers are required, add path identifiers and connection topology before relaxing this validation; turns per parallel branch must remain electrically meaningful.

## Core and Air-Gap Construction

Reuse `CoreSpec` to retain the current core form and aperture definition. Compute the initial placement chamber from the aperture and core-to-winding clearances; then apply chamber splits and placements without changing the core sizing model in this phase.

`AirGapSpec` and `Geometry.air_gaps` already represent multiple positioned rectangular gaps, and the general constructor can accept multiple gaps. Retain them and constrain the new public construction workflow to center-leg gaps as required. Remove the old factory-only assumption that a gap is always centered; allow the new API to receive the desired center-leg gap specifications. Validate each gap against the center-leg bounds. Do not add shunt behavior in this phase; `Geometry.shunts` is present in the schema but is not currently meshed or solved.

## FEM and Schema Work

- Keep chambers as placement metadata, not meshed regions. The current Gmsh path fragments the outer core, winding window, air gaps, and final conductor shapes; it can remain chamber-agnostic if chamber placement compiles to valid conductor placements.
- Update the normalized model schema to retain chambers, splits, assignments, and path membership/current fractions. Bump the schema version as appropriate; backward loading of older schema/API versions is not a requirement for this in-development project.
- Ensure model serialization and normalization preserve the new geometry and winding data.
- Update frequency- and voltage-domain current mapping, Litz current-density assignment, flux linkage, winding/DC loss calculations, and reports to use per-placement path data consistently.
- Keep core material, excitation, mesh settings, and solver APIs independent of chamber construction.

## Remove or Replace

Remove the old `RadialWindingSpec` / `build_concentric_model` path and the automatic regular-row `Constructor.place_winding` helper (`radial_columns`, hard-coded concentric/vertical branches, and winding-wide alignment offsets). Keep `WindingLayer`, `WindingSpec`, and `Constructor.build()` only for low-level, explicitly authored geometries such as the standalone study; consumer construction uses chambers. There is no adapter for the removed high-level factory.

Retain and reuse `CoreSpec`, `InsulationSpec` data where appropriate, conductor definitions, `AirGapSpec`, final FEM schema objects, and the solver-facing geometry conversion. Replace only the high-level construction abstractions that impose the limitations described above.

## Notebook Migration

After the public API and serialization format are implemented, update the existing construction examples in:

- `hv-magnetics-data/notebooks/transformer_single.ipynb`
- `hv-magnetics-data/notebooks/transformer_sweep.ipynb`

Move no geometry-generation implementation into `hv-magnetics-data`. Keep design-specific dimensions, winding assignments, split positions, insulation, material values, and excitation values in the relevant notebook or design data. Demonstrate both a simple single-window arrangement and a recursively split/chamber-assigned arrangement. Preserve existing notebook cell order, metadata, and outputs unless an output must be regenerated as part of validation.

## Test and Acceptance Plan

1. Unit-test rectangular bounds, recursive splitting, equal partitions, explicit percentage dividers, gap insulation, and invalid/overlapping splits.
2. Test conductor footprint containment, clearances, all horizontal/vertical location modes, top/bottom starts, serpentine progression, and per-section radial layer counts for round and rectangular conductors.
3. Test chamber assignments across multiple chambers, exact per-winding turn totals, per-chamber parallel-path data, and rejection of incompatible path topology.
4. Assert generated FEM model conductor centers, conductor shapes, winding/circuit mapping, electrical circuit turns, signs, and current fractions/path identifiers.
5. Run frequency- and voltage-domain regression tests for current, flux linkage, inductance, Litz loss, DC resistance, and reporting under chamber-generated windings.
6. Test schema serialization/normalization round trips for the new model.
7. Test `RadialStack`, `AxialStack`, repeated logical windings, and wrapper-returned chamber/window bounds. Confirm wrapper dimensions include the correct divider and core clearances and can construct the core without manually specified window dimensions.
8. Run `python -m pytest` from the pyMagFEM repository root and Ruff with the repository's configured 100-character line length.
9. Update the D2/D3 notebook designs to use the high-level wrappers and validate the resulting layouts; do not leave notebook examples calling an API that has not been implemented.
