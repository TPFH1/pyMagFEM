---
description: "Use when extending WindingSection, physical conductor counts, parallel-path allocation, branch topology, induced or terminal voltage postprocessing, or imbalanced current sharing in pyMagFEM."
---

# Winding Sections and Parallel Paths

## Current Model

- `WindingDefinition.turns` is the integer electrical series-turn count per parallel branch.
- `WindingSection.conductor_count` is the integer number of physical conductor bundles placed in that section. For a winding with `P` parallel paths, conductor counts across its sections must sum to `WindingDefinition.turns * P`.
- Sections and chamber assignments require an explicit integer `conductor_count`; do not use electrical `turns` as a placement count.
- All sections of one logical winding use the same `parallel_paths`. Section order defines the physical chamber order for the selected radial or axial scheme.
- Under the current ideal-sharing model, conductor path IDs are assigned round-robin across the ordered sections. A path-local `turn_index` is recorded on each placement. This deterministic allocation is a convenience under equal sharing, not a complete electrical connection model.
- Every physical conductor location affects the electromagnetic solution. This applies to solid conductors and Litz bundles; Litz strands are homogenized rather than individually positioned.
- `TurnPlacement` stores `parallel_path` and `current_fraction`, but solver current distribution and most post-processing use the equal fraction `1 / parallel_paths`. Path IDs are not yet used to solve an electrical network.
- `winding_voltages()` calculates local per-conductor voltages. `winding_terminal_voltages()` combines them using current fractions and does not report per-path voltage sums or solve branch-voltage mismatch.

## Imbalanced-Sharing Work

- Keep electrical series turns, physical conductor bundles, and parallel branches as distinct concepts. Do not encode branch allocation by making series-turn counts fractional.
- Preserve the low-parameter high-level schemes for regular, equal-sharing layouts. Require explicit branch membership or connection data only when the requested topology cannot be represented unambiguously by the automatic allocation.
- Before adding branch-current behavior, define how physical placements map to ordered series paths, including current direction/polarity and any modeled lead or branch impedance. A path ID alone identifies membership but does not define the complete electrical network.
- For a branch-aware model, validate integer physical conductor counts and that each path has exactly the declared series-turn count. Preserve winding section order and deterministic path-local turn numbering.
- Branch currents must satisfy terminal-current conservation. Parallel branches share terminal voltage; unequal induced EMFs or impedances may produce unequal currents or circulating currents. Do not silently force equal sharing when the selected model requests imbalanced sharing.
- Decide explicitly whether field and circuit solutions are coupled or iterated. The conductor current distribution affects the magnetic field, while the field induces branch voltage; do not claim imbalanced-current support from a post-processing-only redistribution.
- Update all current consumers consistently: frequency-domain and voltage-driven solves, conductor current density, Litz and solid losses, flux linkage, inductance identification, DC resistance, induced conductor voltage, terminal voltage, complex power, and reports.
- Make voltage results distinguish per-conductor induced voltage, per-branch series voltage, and terminal voltage. Do not use the current equal-fraction weighted terminal-voltage aggregate as a branch-network solution.
- Preserve loading and behavior for existing models that omit branch topology; they should retain ideal equal current sharing unless a versioned schema explicitly selects another mode.

## Regression Coverage

- Test that sections and assignments require integer `conductor_count` and reject the removed `turns` placement alias.
- Test a winding split across sections with complementary path allocations, asserting every physical conductor's path ID and path-local turn index, equal branch turn totals, and unchanged circuit turns.
- Test invalid conductor totals, missing/duplicate path assignments, inconsistent path counts, and invalid path polarity/ordering when those concepts are added.
- For imbalanced sharing, test terminal-current conservation, branch-voltage compatibility, induced-voltage differences, current redistribution, power balance, and solid/Litz loss accounting.
- Add serialization round trips and backward-loading coverage whenever branch connectivity or current-sharing metadata changes the model schema.