# Contributing to ns-sdn

ns-sdn is the software artifact of research conducted under the ICSRI
Fellows Program at the Intelligent Computer Systems Research Institute,
Miami Herbert Business School, University of Miami. The code, the papers it
implements and the results it produces are evaluated as the work of the
fellow, so **the project does not accept code contributions from outside the
fellowship**. This page says what that means, what help is welcome, and how
the maintainer works so that the project stays reproducible for everyone who
uses it.

## Table of Contents

- [Contribution Policy](#contribution-policy)
  - [What is welcome](#what-is-welcome)
  - [What happens to an unsolicited pull request](#what-happens-to-an-unsolicited-pull-request)
  - [Forks and derived work](#forks-and-derived-work)
- [Reporting Issues](#reporting-issues)
- [Reporting a Numerical Discrepancy](#reporting-a-numerical-discrepancy)
- [How the Project Is Developed](#how-the-project-is-developed)
  - [Development setup](#development-setup)
  - [Architecture](#architecture)
  - [Coding standards](#coding-standards)
  - [Static analysis](#static-analysis)
  - [Testing](#testing)
  - [Documentation](#documentation)
  - [Release process](#release-process)
- [License](#license)
- [Security Vulnerability Reporting](#security-vulnerability-reporting)
- [Code of Conduct](#code-of-conduct)
- [Governance](#governance)
- [Contact](#contact)

## Contribution Policy

Authorship of the library is confined to the fellowship for the duration of
the research it supports. This is not a statement about the quality of
outside work; it is a condition of the program under which the work is
assessed and published. The policy will be revisited when the research
concludes, and this page will change before the first pull request from
outside is merged. Until then:

### What is welcome

- **Bug reports.** A wrong shape, an exception that should not have been
  raised, a summary that disagrees with the arrays behind it, a doctest that
  fails on your platform. Use the bug report template.
- **Reproduction reports.** Fits of the published configuration that do not
  reproduce the IEEE-ORSS metrics within the tolerances the documentation
  states, with the platform, torch version and seed. These are the most
  valuable reports the project can receive.
- **Feature requests and design discussion.** An option, a path, a
  spectral-state coordinate, a penalty. Use the feature request template or
  GitHub Discussions; proposals with a literature reference are easier to
  evaluate.
- **Documentation corrections** reported as issues: a wrong equation, a
  broken cross-reference, an example whose prose no longer matches its
  output.
- **Security reports**, through the private channel in
  [SECURITY.md](SECURITY.md).
- **Citations and usage reports.** If ns-sdn appears in your work, an issue
  or a Discussion post saying so helps the project make its case.

### What happens to an unsolicited pull request

It is closed with thanks and without review of the code. If it reports a
real defect or proposes a change the maintainer agrees with, the maintainer
implements it independently and credits the reporter in `CHANGELOG.md` by
GitHub handle. Closing a pull request is not a judgement on it; it is the
policy applied uniformly.

### Forks and derived work

The MIT license permits forking, modifying and redistributing the code, and
nothing here restricts that. A fork is a separate project; please rename it
and do not present it as ns-sdn. If you publish results from a modified
version, say which version and what was changed, since the published
metrics are tied to the exact configuration in this repository.

## Reporting Issues

Before opening a new issue:

- Search the tracker, including closed issues, for the same report.
- Use the issue templates; blank issues are disabled.
- Give a minimal reproduction on synthetic data where possible. If the
  problem needs a particular series, say how to fetch it (for FRED, the
  series id and the date window) rather than attaching data.
- Include the full traceback or the printed `summary()`.
- Include the environment: ns-sdn, Python, torch and numpy versions, the
  torch device, OS and architecture, and how ns-sdn was installed.

## Reporting a Numerical Discrepancy

The published run is reproduced by `examples/plot_03_reproduce_orss.py`.
Trained weights reproduce exactly on one machine with one seed and
approximately across BLAS backends and torch versions, so a report should
state:

1. The command and environment (`FRED_API_KEY` set, the script unmodified).
2. The `metrics.csv` the script wrote and the parameter count it printed
   (58,849 for the published configuration).
3. Which rows differ from the documented values and by how much.
4. Whether the difference persists across two seeds.

A difference in the fourth decimal of a holdout RMSE is expected across
platforms; a different parameter count, a different sign in `mix`, or a
holdout error worse than the random walk by a wide margin is a bug report.

## How the Project Is Developed

The rest of this page documents the maintainer's own process. It is public
so that the results are auditable and so that the conventions are stated
once, in one place.

### Development setup

ns-sdn uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management and [hatchling](https://hatch.pypa.io/) to build.

```bash
git clone https://github.com/nikhilxsunder/ns-sdn.git
cd ns-sdn
uv sync --group dev
uv run pre-commit install
```

The `dev` group composes `test`, `lint`, `docs`, `notebooks` and `tools`.
On Linux, `pyproject.toml` pins `torch` to the PyTorch CPU index so a sync
does not download the CUDA stack; macOS and Windows use PyPI's wheels.

The FRED API key is needed only by the two DGS10 examples and the
reproduction script; the library performs no network I/O. Keep the checkout
outside folders a sync service mirrors (iCloud Drive, Dropbox), which create
numbered copies of files that tools rewrite and that Python and Sphinx pick
up as real.

### Architecture

The package follows the conventions of
[cultivars](https://github.com/nikhilxsunder/cultivars), so that cultivars
can expose the model under its experimental namespace without adapters and
so that a workflow written against one library reads the other.

- **Three-object discipline.** `NSSDN` is the immutable, validated
  specification; `engine/_internals` holds the mechanism (the torch module,
  the objective, the staged trainer, the predictor); `NSSDNResult` is the
  frozen record assembled by `_from_fit`. The model object holds nothing but
  the specification.
- **Layers.** `engine/_core` holds pure values and imports only numpy: the
  option aliases, the defaults, the validators and the containers.
  `engine/_internals` holds everything that touches torch. The public
  modules (`spectral_decomposition`, `spectral_state`, `summary`, `typing`,
  `exceptions`) import from the engine and never the reverse.
- **Fidelity to the published run.** The defaults of `NSSDN` are the
  IEEE-ORSS configuration and the test suite pins the parameter count. New
  behaviour goes behind an option that is off by default.
- **Leaf-module imports.** The top-level package re-exports modules only;
  every public name is imported from the module that defines it.
- **Module header.** Every `.py` file begins with the MIT header
  (`Copyright (c) 2026 Nikhil Sunder`, the SPDX identifier), the module
  docstring and `from __future__ import annotations`.

### Coding standards

- [PEP 8](https://peps.python.org/pep-0008/) through `ruff format` and
  `ruff check` with the rule set in `pyproject.toml`.
- Complete type hints on every public parameter and return value, checked
  by `mypy`. Array annotations are written inline as
  `npt.NDArray[np.float64]`; there are no module-level array aliases.
  Categorical options are PEP 695 aliases over `Literal` in `ns_sdn.typing`
  and are validated at runtime against the same alias.
- Google-style docstrings with a doctest where one clarifies behaviour.
  Every public class carries the model in notation, the references, and a
  runnable example that fits in seconds.
- Defensive programming: explicit shape, length, finiteness and choice
  checks; every raised error is a `DimensionError`, `SpecificationError` or
  `NumericalError` with the offending value in the message; no silent
  failure, no bare `except`, no `assert` for input validation.
- Results are frozen dataclasses with `repr=False`; every array they expose
  is on the original scale unless `scaled=True` was asked for.
- Randomness is controlled by the `seed` argument through
  `torch.random.fork_rng`; nothing touches the global torch or numpy seed.

### Static analysis

```bash
uv run ruff check src/ns_sdn tests examples
uv run ruff format --check src/ns_sdn tests examples
uv run mypy
uv run bandit -r src/ns_sdn       # advisory
uv run pip-audit                  # advisory
```

The first three block in `analyze.yml`; the last two report.

### Testing

```bash
uv run pytest
```

`pyproject.toml` configures pytest to run the doctests under `src/ns_sdn`
and the behavioural tests under `tests/` together, with coverage. The
behavioural suite pins the published parameter count, the one-step holdout
protocol, the predicted-alignment leak test, the path identities, the
scaled paths, forecast shapes, pickling, rendering and frames, ordered
frequencies, the time-aware transition on an irregular grid, the covariate
contract at forecast time, the penalties, the specification errors, the
spectral state and the Wasserstein distance. New behaviour lands with a
test in the same commit. The coverage floor reported to Codecov is 90%.

### Documentation

The site is Sphinx with the pydata theme, built from `docs/`:

```bash
uv run make -C docs html                        # runs the gallery
DOCS_PLOT_GALLERY=0 uv run make -C docs html    # skips the gallery
uv run make -C docs strict                      # warnings are errors
uv run sphinx-build -b doctest docs/source docs/build/doctest
```

Every `testcode` block on the site runs under the doctest builder, and the
gallery examples execute at build time except the ORSS reproduction, which
runs only with `DOCS_RUN_ORSS=1`. The deploy workflow publishes `main` as
`stable`, `dev` as `dev`, and every tag under its own number.

### Release process

Versions follow Semantic Versioning with PEP 440 prerelease segments. The
version lives in `src/ns_sdn/__about__.py` and nowhere else.

1. Update `CHANGELOG.md`: move the *Unreleased* entries under the new
   version with its date.
2. Bump `__version__`, run the full suite and `uv lock`, commit.
3. Tag: `git tag -s vX.Y.Z` (or `vX.Y.ZaN` for a prerelease) and push the
   tag. `prerelease.yml` or `release.yml` then runs the tests against the
   tagged tree, builds, checks the sdist for notebooks, publishes to PyPI
   through trusted publishing (OIDC; no upload tokens exist), signs the
   artifacts with Sigstore and GPG, and creates the GitHub release.
4. After a final release, the conda-forge feedstock is updated by the
   autotick bot; `release.yml` has an opt-in manual fallback.

## License

ns-sdn is released under the [MIT License](LICENSE). Every source file
carries the MIT header. Any contribution that is ever accepted will be
licensed the same way and will carry a `Signed-off-by` line under the
[Developer Certificate of Origin](DCO.md).

## Security Vulnerability Reporting

Do not report security vulnerabilities through public issues. Use the
"Report a vulnerability" button under the repository's Security tab, or
email nsunder724@gmail.com. [SECURITY.md](SECURITY.md) has the full policy
and the threat model.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). It
applies to issues, Discussions and every other interaction in the project's
spaces. Report unacceptable behaviour to nsunder724@gmail.com.

## Governance

ns-sdn has a single maintainer, Nikhil Sunder, who has final authority on
the project's direction and on every change to it. Decisions about scope
are recorded in [ROADMAP.md](ROADMAP.md); decisions about behaviour are
recorded in `CHANGELOG.md` and in the documentation.

## Contact

Questions about using the model: GitHub Discussions. Defects: the issue
tracker. Anything else: nsunder724@gmail.com.
