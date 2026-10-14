# Changelog

All notable changes to ns-sdn are listed here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[PEP 440](https://peps.python.org/pep-0440/) versions under
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). From 1.0.0 the
public names, argument names, result fields and summary wording are frozen
for the 1.x line; a change that alters the output of the default
configuration on the published data is a major version, and every change to
a public surface is listed under _Changed_.

## [Unreleased]

## [1.0.0] - 2026-10-14

First stable release. The API of 1.0.0a1 is frozen as the 1.x contract: the
IEEE-ORSS model through the cultivars model-fit-result pattern, the spectral
state, the documentation site, the signed release pipeline and the Zenodo
archive.

### Added

- Documentation site (`docs/`): getting started, a user guide (model,
  decomposition, forecasting, spectral state, conventions, nuances), the API
  and engine references, and a version switcher deployed to
  `nikhilxsunder.github.io/ns-sdn/`.
- Example gallery (`examples/`): a synthetic series with a known
  decomposition, a shortened DGS10 run with its spectral state, and the
  IEEE-ORSS reproduction as `plot_03_reproduce_orss.py`.
- CI: analysis, tests with coverage, reports, CodeQL, dependency review,
  OpenSSF Scorecard, documentation deployment, and signed prerelease and
  release pipelines publishing to PyPI through trusted publishing.
- conda recipes (`conda-recipe/`, `conda-forge/`) for the Anaconda channel
  and the conda-forge feedstock.
- Community files: contributing policy, code of conduct, security policy,
  roadmap, citation metadata, issue and pull request templates, code owners.
- `[tool.uv.sources]` pinning Linux `torch` to the CPU index so continuous
  integration does not download the CUDA stack.
- Zenodo archiving of every GitHub release: concept DOI
  [10.5281/zenodo.23368625](https://doi.org/10.5281/zenodo.23368625) (resolves to the latest
  release) and version DOI [10.5281/zenodo.23368626](https://doi.org/10.5281/zenodo.23368626)
  for 1.0.0a1, with the badge in the README, the `doi` and `identifiers`
  fields in `CITATION.cff`, the software entries on the citing page and in
  `references/bibliography/master.bib`, and the JSON-LD identifier on the
  documentation site.
- Documentation: the JSON-LD template is now `layout.html`, the name Sphinx
  loads from `templates_path`; as `_layout.html` it was never rendered, so the
  structured data (software, website, article per page) now appears on every
  page, with the Zenodo DOI as the software identifier.
- `tests/test_spectral_decomposition.py` restored to the repository; the
  suite (15 tests, including the pin on the published 58,849-parameter
  count) had been dropped from the tree before the `1.0.0a1` commit.

### Changed

- `Development Status :: 5 - Production/Stable`; the documentation's
  version-warning banner is on, so a reader of `dev` or of an old release is
  pointed at `stable`.
- `examples/reproduce_orss.py` moved to `examples/plot_03_reproduce_orss.py`
  and gained the random-walk rows in log units and a spectral-state figure.
- Citation metadata: the IEEE-ORSS paper is published (_2026 IEEE
  Opportunity Research Scholars Symposium_, pp. 1-5,
  doi:10.1109/ORSS71174.2026.11684282) and is now the preferred citation in
  `CITATION.cff`, the README, the documentation and the module references.
- Docstrings across `src/ns_sdn`: every module, class, method and property
  now carries the model's equations in LaTeX, the row and scale conventions
  of each path, the formula behind each statistic and diagnostic, and
  doctested examples (71 doctests, up from 33). No signature changed.
- `references/`: a master bibliography (`bibliography/master.bib`, one
  canonical key per work) and a README mapping the stored papers to it;
  `docs/source/references.bib` is now a verbatim subset of the master.

### Fixed

- README quick-start output for the spectral state regenerated after the
  velocity fix below; the NSSDN summary block and the RMSE figures were
  unchanged.

- `SpectralState.velocity` and `velocity_frequency` were underestimated
  whenever the lowest frequency moved between rows: the 1-D Wasserstein
  helper indexed the cumulative mass with `searchsorted(...) - 1`, which
  wrapped to the total mass for grid points below a measure's smallest atom
  instead of zero. Two unit masses at 1.0 and 1.5 returned 0.0 rather than
  0.5. `velocity_mass` was unaffected. Velocity figures computed with
  1.0.0a1 should be regenerated.

## [1.0.0a1] - 2026-10-14

First functional release: the IEEE-ORSS model through the cultivars
model-fit-result pattern.

### Added

- `NSSDN`, the validated specification, with the published configuration as
  its defaults (sixteen components, a 128-dimensional GRU state, the filtered
  alignment, the two-stage Adam schedule with amplitude and frequency drift
  penalties; 58,849 trainable parameters without covariates).
- `NSSDNResult`, the frozen fitted record carrying every path of the
  decomposition on the original scale (`trend`, `amplitude`, `amplitude_raw`,
  `frequency`, `phase`, `phase_offset`, `gate`, `components`, `state`,
  `mix`), the one-step fit (`fittedvalues`, `resid`, `sigma2`, `llf`,
  `r_squared`), `decompose` on the fitted or a new sample with
  `scaled=True` for training units, one-step `predict`, recursive
  `forecast`, `spectral_state`, `summary`, `to_dict`, `to_pandas`,
  `information_criteria` and `compare`.
- `SpectralState`: the decomposition read as a time-local measure on the
  frequency axis, with centroid, dispersion, entropy, effective count,
  participation ratio, 1-Wasserstein velocity split into frequency and mass
  parts, the active set, the contribution correlation and the
  disentanglement score.
- `Stage` and `Penalties` schedule records; the `phase_curvature`, `trend`,
  `sparsity` and `spectrum` penalties, off by default.
- Options: `alignment` (`filtered`, `predicted`), `transition` (`gru`,
  `timeaware`), `positive_frequency`, `ordered_frequency`, `gating`,
  `objective` (`mse`, `huber`), `scaling` (`minmax`, `none`), `seed`,
  `device`, covariates through `exog`, and an irregular `time` grid.
- `SummaryTable` and `InformationCriteria` reporting records; summaries
  state the alignment, the concentrated likelihood and the permutation
  identifiability caveat.
- `DimensionError`, `SpecificationError` and `NumericalError` under
  `NSSDNError`, mirroring `cultivars.exceptions`.
- Pickling of results through the network's state dict.
- Test suite: behavioural tests including the published parameter count,
  the one-step holdout protocol, the predicted-alignment leak test, the
  path identities and the pickle round trip; doctests on every public
  object.

## [0.0.1] - 2026-08-19

Name reservation on PyPI. No functionality.

[Unreleased]: https://github.com/nikhilxsunder/ns-sdn/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/nikhilxsunder/ns-sdn/compare/v1.0.0a1...v1.0.0
[1.0.0a1]: https://github.com/nikhilxsunder/ns-sdn/releases/tag/v1.0.0a1
[0.0.1]: https://pypi.org/project/ns-sdn/0.0.1/
