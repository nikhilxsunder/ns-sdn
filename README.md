<div align="center">
    <img src="https://raw.githubusercontent.com/nikhilxsunder/ns-sdn/main/assets/exported/ns-sdn_banner.png"  alt="ns-sdn Logo">
</div>

## A neural state-space model that decomposes a non-stationary time series into a trend and a few sinusoids whose amplitude, frequency and participation move with a latent state, and reads the decomposition back as coordinates.

|                      |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **CI / Quality**     | [![Analyze](https://github.com/nikhilxsunder/ns-sdn/actions/workflows/analyze.yml/badge.svg)](https://github.com/nikhilxsunder/ns-sdn/actions/workflows/analyze.yml) [![Tests](https://github.com/nikhilxsunder/ns-sdn/actions/workflows/test.yml/badge.svg)](https://github.com/nikhilxsunder/ns-sdn/actions/workflows/test.yml) [![Docs](https://github.com/nikhilxsunder/ns-sdn/actions/workflows/docs.yml/badge.svg)](https://nikhilxsunder.github.io/ns-sdn/stable/) [![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) |
| **Security**         | [![CodeQL](https://github.com/nikhilxsunder/ns-sdn/actions/workflows/codeql.yml/badge.svg)](https://github.com/nikhilxsunder/ns-sdn/actions/workflows/codeql.yml) [![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/nikhilxsunder/ns-sdn/badge)](https://scorecard.dev/viewer/?uri=github.com/nikhilxsunder/ns-sdn) [![Socket](https://socket.dev/api/badge/pypi/package/ns-sdn/1.0.0?artifact_id=tar-gz)](https://socket.dev/pypi/package/ns-sdn/overview/1.0.0/tar-gz)                                                                                                                                             |
| **Coverage**         | [![Coverage](https://codecov.io/gh/nikhilxsunder/ns-sdn/graph/badge.svg)](https://codecov.io/gh/nikhilxsunder/ns-sdn)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| **Packaging**        | [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv) [![Hatch project](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pypa/hatch/master/docs/assets/badge/v0.json)](https://github.com/pypa/hatch)                                                                                                                                                                                                                                                                                                            |
| **Distribution**     | [![PyPI](https://img.shields.io/pypi/v/ns-sdn.svg)](https://pypi.org/project/ns-sdn/) [![Python Versions](https://img.shields.io/pypi/pyversions/ns-sdn.svg)](https://pypi.org/project/ns-sdn/)                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| **License**          | [![License](https://img.shields.io/pypi/l/ns-sdn.svg)](https://github.com/nikhilxsunder/ns-sdn/blob/main/LICENSE)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| **Usage**            | [![PyPI Downloads](https://static.pepy.tech/badge/ns-sdn)](https://pepy.tech/projects/ns-sdn)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Research / Index** | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23368625.svg)](https://doi.org/10.5281/zenodo.23368625) [![FLAIRS-39](https://img.shields.io/badge/FLAIRS--39-10.32473%2Fflairs.39.1.141588-blue)](https://doi.org/10.32473/flairs.39.1.141588) [![IEEE ORSS](https://img.shields.io/badge/IEEE%20ORSS-10.1109%2FORSS71174.2026.11684282-blue)](https://doi.org/10.1109/ORSS71174.2026.11684282)                                                                                                                                                                                                                                   |

> **Stable.** ns-sdn `1.0.0` is the first stable release. The public names, argument names, result fields and summary wording are frozen for the `1.x` line under semantic versioning, the defaults reproduce the IEEE-ORSS run to the parameter count, and a change that alters the output of the default configuration on the published data requires a major version. Pin the exact version in anything that must reproduce.

## Research context

> Note: Listing does not imply endorsement or affiliation.

ns-sdn is developed as fellowship research under the ICSRI Fellows Program at the Intelligent Computer Systems Research Institute, Miami Herbert Business School, University of Miami. For that reason the project does not accept outside code contributions; see [Contributing](#contributing).

<a href="https://herbert.miami.edu/" title="University of Miami Herbert Business School">
    <img src="https://mbaworldsummit.com/wp-content/uploads/2019/06/Miami-BU-Logo.png"
         alt="University of Miami Herbert Business School"
         height="150">
</a>

## What it is

At each step the network reads a latent state and emits a trend plus $K$ sinusoids,

$$
\hat y_t = B_t + \sum_{k=1}^{K} w_k\, g_{t,k}\, A_{t,k}\, \sin\!\bigl(\theta_{t,k} + \phi_{t,k}\bigr),
\qquad
\theta_{t,k} = \theta_{t-1,k} + \omega_{t,k}\,\Delta t,
\qquad
h_t = \mathrm{GRU}(h_{t-1}, y_t),
$$

where $B_t$ is the trend, $A_{t,k}$ the amplitude, $\omega_{t,k}$ the instantaneous frequency, $\phi_{t,k}$ the phase offset and $g_{t,k}$ the gate of component $k$, each a linear head on the state, and $w$ a learned mixing vector. The phase integrates the frequency, so each component is a chirp the state can steer. The network is trained as a one-step predictor with drift penalties on the amplitudes and frequencies, and read as a time-varying spectral decomposition.

The package implements the model of the IEEE-ORSS paper ([Sunder and Leon, 2026](https://doi.org/10.1109/ORSS71174.2026.11684282)) and the architecture introduced at FLAIRS-39 ([Sunder, 2026](https://doi.org/10.32473/flairs.39.1.141588)), in the model-fit-result pattern of [cultivars](https://github.com/nikhilxsunder/cultivars): a validated specification, one `fit()`, and a frozen result that carries every path of the decomposition on the original scale and prints a summary that states what its numbers assume.

| Module                          | Contents                                                                                                                                                                                                                                                    |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ns_sdn.spectral_decomposition` | `NSSDN`, the specification; `NSSDNResult`, the fitted record with the trend, amplitude, frequency, phase, offset, gate, component and state paths, one-step `predict`, recursive `forecast`, `decompose` on new data; `Stage` and `Penalties`, the schedule |
| `ns_sdn.spectral_state`         | `SpectralState`: the decomposition as a time-local measure on the frequency axis, with centroid, dispersion, entropy, effective count, participation ratio, Wasserstein velocity split into frequency and mass motion, active set, disentanglement score    |
| `ns_sdn.summary`                | `SummaryTable` and `InformationCriteria`, the records every result renders itself through                                                                                                                                                                   |
| `ns_sdn.typing`                 | The categorical option aliases: `Alignment`, `Transition`, `Objective`, `Scaling`                                                                                                                                                                           |
| `ns_sdn.exceptions`             | `DimensionError`, `SpecificationError`, `NumericalError`, one per kind of fix, under `NSSDNError`                                                                                                                                                           |

Dependencies are `numpy` and `torch`. `pandas` (dataframe export), `fedfred` (FRED access for the examples) and `matplotlib` (figures in the examples) are optional extras.

## Installation

Requires Python 3.12 or later.

```bash
pip install ns-sdn
```

With the extras:

```bash
pip install "ns-sdn[pandas,data]"     # dataframes and FRED access
pip install "ns-sdn[all]"             # plus matplotlib
```

With `uv`:

```bash
uv add ns-sdn
```

On Linux, PyPI's `torch` wheel depends on the CUDA libraries, several gigabytes that a CPU-only machine never loads. Install torch from the CPU index first and ns-sdn will use it:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ns-sdn
```

For a reproducible research environment, pin the exact version: `ns-sdn==1.0.0`. A conda-forge package follows this release.

## Quick start

### Fit a decomposition

A series with a drifting level, a chirp and a fast oscillation, and a three-component model on a short schedule so the example runs in seconds. The published configuration is the default (`NSSDN(y)`: sixteen components, a 128-dimensional state, the two-stage schedule).

```python
import numpy as np
from ns_sdn.spectral_decomposition import NSSDN, Stage

rng = np.random.default_rng(3)
t = np.arange(300.0)
y = 2.0 + 0.003 * t + np.sin(np.cumsum(0.10 + 0.0003 * t)) + 0.4 * np.sin(0.55 * t)
y += 0.05 * rng.standard_normal(300)
train, test = y[:270], y[270:]

res = NSSDN(train, k=3, state_dim=16, schedule=(Stage(1e-2, 80),), seed=0).fit()
print(res)
```

```text
                                       NSSDN(k=3) Results
================================================================================================
Model:          NSSDN(k=3, d=16, gru, filtered)  Log-likelihood:                          47.550
Objective:                                  mse  AIC:                                   2214.901
Scaling:                                 minmax  BIC:                                   6366.792
Observations:                               269  HQIC:                                  3882.309
Parameters:                                1155  One-step RMSE:                         0.202766
Gating:                                    True  Emission RMSE:                        0.0318907
------------------------------------------------------------------------------------------------
                mean amp   mean freq   period   mean gate      mix   power share
component 1       0.1031      0.4796     13.1       0.189   +0.241         0.945
component 2       0.2300      0.8186      7.7       0.365   +0.017         0.021
component 3       0.1594      0.7729      8.1       0.250   -0.030         0.033
================================================================================================
Filtered alignment: the emission at t conditions on y[t], so emission is a reconstruction of the
observed series and fittedvalues report the emission at t as the one-step forecast of y[t+1]
(the published protocol); compare against a random walk before reading them as skill.
llf is a concentrated Gaussian likelihood at a penalized fit and n_params the nominal weight
count; criteria compare k within one configuration only.
Components are identified up to permutation; indices are not stable across refits.
```

### Read the paths

Every path is an array on the result, on the original scale of the series, and the synthesis equation is an identity on them:

```python
res.trend.shape, res.amplitude.shape, res.frequency.shape, res.gate.shape   # (270,), (270, 3), (270, 3), (270, 3)
np.allclose(res.trend + res.components.sum(axis=1), res.emission)            # True
res.period.mean(axis=0)                                                      # 2*pi / frequency, in time units
res.decompose(scaled=True)                                                   # the same paths in training units, as the papers draw them
res.decompose(other_series)                                                  # the trained network over new data
res.to_pandas(index=dates)                                                   # one column per path, if pandas is installed
```

### Forecast two ways

The published evaluation is conditional one-step: run the trained network over training and holdout together under teacher forcing and read the emission at each step as the forecast of the next. The random walk is the benchmark to beat.

```python
one_step = res.predict(y)[-30:]            # one value per observation after the first; the last 30 are the holdout
naive = y[269:-1]                          # y[t-1] as the forecast of y[t]
rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
rmse(test, one_step), rmse(test, naive)    # 0.245, 0.223 on this short fit: persistence wins

path = res.forecast(10)                    # recursive: each emission fed back as the next input
```

### Read the spectral state

Sixteen amplitude and frequency paths are a lot to reason about. `spectral_state()` normalizes each row's contribution powers into a probability measure on the frequency axis and reports its coordinates:

```python
state = res.spectral_state()
print(state)
```

```text
                              Spectral state (k=3)
================================================================================
Rows:                              270  Disentanglement:                 0.988
Active threshold:                 0.01  Mean active:                      3.00
--------------------------------------------------------------------------------
coordinate                              min / mean / max
centroid                         0.4829 / 0.4981 / 0.578
dispersion                   0.04368 / 0.07437 / 0.08533
effective count                    1.207 / 1.301 / 1.402
participation ratio                 1.082 / 1.127 / 1.18
velocity                  8.892e-05 / 0.002766 / 0.04415
velocity (frequency)      5.351e-05 / 0.002124 / 0.04279
velocity (mass)          1.417e-05 / 0.0006716 / 0.00206
================================================================================
Diagnostics of a model trained on the prediction criterion alone; nothing here
enters a loss.
Identifiability degrades as frequencies converge: compression is also the regime
of weakest disentanglement, and per-component readings of inactive rows are not
meaningful.
```

Every coordinate is also an array on the state, one entry per row. A regime claim, such as the frequency convergence with amplitude concentration that the IEEE-ORSS paper reads around 2020, is a joint contraction of `dispersion` and `effective_count` across a window, in coordinates that survive a refit; the component indices do not.

### Reproduce the paper

```bash
FRED_API_KEY=... python examples/plot_03_reproduce_orss.py
```

fetches log DGS10 on the papers' daily grid, fits the default configuration with seed 1337, and writes the metrics table (with the random-walk rows the paper did not include), the six figures and the spectral state to `orss_reproduction/`.

## Design

Three commitments run through the package.

**The published run is the contract.** `NSSDN()` with no arguments is the IEEE-ORSS configuration, the test suite pins its 58,849 parameters, and every option that was not in the paper (`ordered_frequency`, the `predicted` alignment, the `timeaware` transition, the extra penalties) is off by default and named in the summary when on.

**The decomposition is the product, and it says what it is.** The result carries every path the papers' figures are drawn from and the identity between them. The summary states which alignment produced it and what that makes of its one-step numbers: under the default `filtered` alignment the emission at $t$ has seen $y_t$, so it is a reconstruction, and the one-step forecast is that emission carried forward, to be read against a random walk before being read as skill.

**Failures say what kind of fix is needed.** Every exception derives from `ns_sdn.exceptions.NSSDNError` and from one of `DimensionError` (reshape the data), `SpecificationError` (change the choice), `NumericalError` (look at the numbers). Each also derives from the matching builtin, and the hierarchy mirrors `cultivars.exceptions`, so one `except` clause handles both libraries.

Packages re-export modules rather than classes; import the model from the module that defines it, as above.

## Documentation

- Stable: <https://nikhilxsunder.github.io/ns-sdn/stable/>
- Development: <https://nikhilxsunder.github.io/ns-sdn/dev/>

The user guide derives the model and explains how to read each path, the two forecast protocols and the spectral-state coordinates; the nuances page lists what the model does that a reader of the papers may not expect (the trend head absorbing the series on short schedules, the calendar-fill frequency artifact, permutation identifiability). The API reference is generated from the docstrings, every one of which carries a runnable example; the examples double as the doctest suite.

## Development

```bash
git clone https://github.com/nikhilxsunder/ns-sdn.git
cd ns-sdn
uv sync --group dev
uv run pytest                                            # doctests + tests, with coverage
uv run ruff check src/ns_sdn tests examples
uv run ruff format --check src/ns_sdn tests examples
uv run mypy
uv run make -C docs html                                 # documentation, running the examples
```

## Contributing

ns-sdn is fellowship research and is assessed as the work of the fellow, so the project does not accept outside code contributions for the duration of the research; unsolicited pull requests are closed without review. Bug reports, reproduction reports, feature requests, documentation corrections and security reports are welcome and acted on, with credit in the changelog. [CONTRIBUTING.md](https://github.com/nikhilxsunder/ns-sdn/blob/main/CONTRIBUTING.md) has the policy and the maintainer's conventions, and [SECURITY.md](https://github.com/nikhilxsunder/ns-sdn/blob/main/SECURITY.md) the private channel for vulnerabilities.

## Citation

If ns-sdn contributes to published work, cite the IEEE-ORSS paper the package implements, the FLAIRS paper that introduced the architecture, and the software:

```bibtex
@inproceedings{sunder2026orss,
  title     = {Non-Stationary Spectral Decomposition Network: Adaptive Spectral Emission Heads and Frequency Modulation},
  author    = {Sunder, Nikhil and Leon, Maikel},
  booktitle = {2026 IEEE Opportunity Research Scholars Symposium (ORSS)},
  address   = {Atlanta, GA, USA},
  publisher = {IEEE},
  pages     = {1--5},
  year      = {2026},
  doi       = {10.1109/ORSS71174.2026.11684282}
}

@article{sunder2026flairs,
  title   = {Non-Stationary Spectral Decomposition Network for Econometric Time Series Forecasting},
  author  = {Sunder, Nikhil},
  journal = {The International FLAIRS Conference Proceedings},
  volume  = {39},
  number  = {1},
  year    = {2026},
  doi     = {10.32473/flairs.39.1.141588}
}

@software{sunder_nssdn,
  author    = {Sunder, Nikhil},
  title     = {ns-sdn: the Non-Stationary Spectral Decomposition Network},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.23368625},
  url       = {https://github.com/nikhilxsunder/ns-sdn}
}
```

The software DOI is the Zenodo concept DOI, which resolves to the latest release; every release is archived under its own DOI, listed on the [Zenodo record](https://doi.org/10.5281/zenodo.23368625). To pin a version in a citation, add `version = {1.0.0}` and cite that release's DOI. [CITATION.cff](https://github.com/nikhilxsunder/ns-sdn/blob/main/CITATION.cff) carries the same entries in a form GitHub and Zenodo read directly.

## License

MIT. See [LICENSE](https://github.com/nikhilxsunder/ns-sdn/blob/main/LICENSE).
