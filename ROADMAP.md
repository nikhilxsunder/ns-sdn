# Roadmap

This document states what ns-sdn intends to do, and what it does not intend
to do, through the end of the fellowship research it supports. Releases are
gated on exit criteria, not dates. Issue-level tracking lives in the
[GitHub milestones](https://github.com/nikhilxsunder/ns-sdn/milestones);
this file is the summary those milestones roll up to.

Last updated: 2026-10-14

## Status

| Release | Theme                                                                      | State               |
| ------- | -------------------------------------------------------------------------- | ------------------- |
| 0.0.1   | Name reservation on PyPI                                                   | Released            |
| 1.0.0a1 | The IEEE-ORSS model through the cultivars model-fit-result pattern         | Released 2026-10-14 |
| 1.0.0   | Stable: API guarantee on the published model and its result                | Released 2026-10-14 |
| 1.0.x   | Validation evidence: reproduction fixture, business-day refit, conda-forge | In progress         |
| 1.1.0   | The journal paper's instruments: corrected spectral coordinates, ablations | Planned             |
| 1.2.0   | Applications: term structure, business cycles, cultivars integration       | Planned             |

## Principles

1. **The published run is the contract.** The defaults of `NSSDN` build the
   IEEE-ORSS configuration, the test suite pins its parameter count, and
   anything new goes behind an option that is off by default.
2. **The decomposition is the product.** Forecast accuracy is reported and
   benchmarked against the random walk, but the paths and the spectral state
   are what the package exists to emit, and every release is judged on
   whether they are easier to read and harder to misread.
3. **Permutation-invariant diagnostics.** Claims about regimes are made in
   coordinates that survive a refit, never in component indices.
4. **Two runtime dependencies.** numpy and torch. Dataframes, FRED access
   and plotting are extras.
5. **One pattern.** The specification, fit and result discipline of
   cultivars, so the model can be exposed there and so a user of either
   library can read the other.
6. **Criteria, not dates.** A release ships when its exit criteria are met.

## 1.0.0: stable

The alpha shipped the model; 1.0.0 freezes it. The public names, argument
names, result fields and summary wording of `NSSDN`, `NSSDNResult`,
`SpectralState`, `Stage` and `Penalties` are the 1.x contract under semantic
versioning, with `Development Status :: 5 - Production/Stable`, the
documentation version banner on, the signed release pipeline exercised on
the prerelease tag, Codecov reporting, the documentation site live, and the
release archived on Zenodo (concept DOI 10.5281/zenodo.23368625, one version
DOI per release, `CITATION.cff` carrying both).

From 1.0.0 onward, a change that alters the output of the default
configuration on the published data requires a major version.

## 1.0.x: validation evidence

Patch releases that add evidence without touching the API: the proof that
the package reproduces the papers and that the surrounding machinery works
end to end.

**Work**

- Reproduction fixture: the metrics table of `plot_03_reproduce_orss.py`
  pinned as a test fixture with a documented tolerance per platform, run on
  a schedule rather than on every push.
- Business-day refit: the same protocol on the trading-day series with the
  integer grid, to measure how much of the lowest published frequency
  (0.90 to 0.95 radians per day, against 2π/7 = 0.898) is the
  calendar-fill artifact. The result decides whether the documentation's
  default recommendation changes.
- Alignment study: the filtered and predicted alignments on the DGS10 run
  and on controlled synthetic series, with the variance split between the
  trend head and the components reported for each, so the nuances page can
  cite numbers rather than one example.
- Documentation: the site live at `nikhilxsunder.github.io/ns-sdn/` with
  the gallery executing in CI, every `testcode` block passing under the
  doctest builder, and `make strict` clean.
- Distribution: the conda-forge feedstock accepted.
- Coverage at or above 95%, with the spectral-state module fully covered.

**Exit criteria.** All of the above, each landing in the patch release that
carries it, with no change to a public surface.

## 1.1.0: the journal paper's instruments

The long-form paper presents NS-SDN as one self-consistent architecture
whose interpretive object is the time-local spectral measure. The package
work that supports it:

- **Corrected spectral coordinates.** A functional effective rank that
  accounts for the correlation between components, and an entanglement gap
  that makes the ORSS "spectral entanglement" observation a measurement
  rather than a description. Both live on `SpectralState` next to the
  existing coordinates, which stay as they are.
- **Initialization and optimizer.** The frequency head's initialization
  (softplus at zero places every component near 0.69 radians per step) and
  the staged schedule become named, documented choices with an initializer
  option, so the trend-absorption behaviour documented on the nuances page
  can be studied rather than worked around.
- **Controlled-simulation ablations.** A `simulate` helper that generates
  series with planted trends, chirps, amplitude steps and regime changes,
  and an ablation table in the documentation: each architectural choice
  (gate, mixing vector, ordered frequencies, each penalty) on and off.
- **Head-projection evaluation.** Tools to correlate the emitted paths with
  exogenous macro and financial series over rolling windows, the test of
  whether a component's movement is economically attributable.

## 1.2.0: applications and integration

- **Term structure.** Fits across the Treasury curve (DGS1 to DGS30), with
  the spectral state compared across maturities, and the yield-curve
  examples added to the gallery.
- **Business cycles.** Monthly activity series on a business-day or monthly
  grid, with recession overlays in the spectral-state figures.
- **cultivars.** The model exposed under cultivars' experimental namespace,
  with the identical result surface, and the two documentation sites
  cross-referenced through intersphinx.

## Under consideration

Not scheduled. Each needs a concrete case before it gets a milestone.

- A multivariate emission: one state, several observed series, shared
  frequencies and series-specific amplitudes.
- Exogenous inputs to the heads directly rather than through the state.
- A frequency head parameterized in period rather than angular frequency,
  for readability on monthly and quarterly data.
- `torch.compile` for the recursive mode, if a profile shows the gain.

## Not planned

- Foundation models, pretrained weights, or any architecture that needs a
  GPU to train at the published size.
- A general-purpose deep-learning forecasting toolkit. ns-sdn is one model,
  implemented carefully.
- Replacing torch with another framework, or supporting several.
- Accepting outside code contributions during the fellowship; see
  [CONTRIBUTING.md](CONTRIBUTING.md).

## How this roadmap changes

The file is revised at each release and whenever a milestone's scope
changes. Proposals go through a GitHub issue or Discussion; the papers in
`papers/` and the design notes in `.ns-sdn/` are the record of why the
items above are ordered as they are.
