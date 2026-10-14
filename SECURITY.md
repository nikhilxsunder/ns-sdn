# Security Policy

## Table of Contents

- [Current Vulnerability Status](#current-vulnerability-status)
- [Reporting a Vulnerability](#reporting-a-vulnerability)
- [Response Process](#response-process)
- [Disclosure Policy](#disclosure-policy)
- [Supported Versions](#supported-versions)
- [Threat Model](#threat-model)
- [Vulnerability Classes and Mitigations](#vulnerability-classes-and-mitigations)
  - [Pickled results](#pickled-results)
  - [Resource exhaustion](#resource-exhaustion)
  - [Malformed input](#malformed-input)
  - [The torch dependency](#the-torch-dependency)
  - [The examples and the FRED key](#the-examples-and-the-fred-key)
  - [Build and release supply chain](#build-and-release-supply-chain)
- [Security Measures](#security-measures)
- [Secure Software Delivery](#secure-software-delivery)
- [Verifying a Release](#verifying-a-release)
- [Security Updates and Announcements](#security-updates-and-announcements)
- [Security Design Principles](#security-design-principles)

## Current Vulnerability Status

ns-sdn is a prerelease (1.0.0aN) research compute library. As of October
2026 there are no known unpatched vulnerabilities of medium or higher
severity in the ns-sdn code base.

Vulnerabilities are monitored through:

- Dependabot scanning of `uv.lock` and the GitHub Actions used by CI.
- `pip-audit` on every push, and `actions/dependency-review-action` on
  every pull request.
- CodeQL static analysis on every push and weekly.
- OpenSSF Scorecard, weekly.
- Reports through the private channel below.

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

Report them through one of:

- **GitHub Security Advisories**: the "Report a vulnerability" button under
  the repository's *Security* tab, which opens a private advisory
  (preferred).
- **Email**: nsunder724@gmail.com.

Include what you can of:

- The type of vulnerability and the file(s) and lines involved.
- The commit, tag or release affected.
- Steps or a script to reproduce, and any special configuration needed.
- A proof of concept, if one exists.
- The impact and how it could be exploited.

## Response Process

1. **Acknowledgment** within 48 hours.
2. **Verification** of the report and its impact.
3. **Remediation**: a fix, tested, in a new prerelease or release.
4. **Disclosure**, coordinated with the reporter.

## Disclosure Policy

- A fix for a verified vulnerability of medium or higher severity is
  targeted within 60 days of verification.
- Reporters are credited in the GitHub release notes and `CHANGELOG.md`
  unless they ask not to be.
- Public disclosure happens with the fix, through a GitHub Security
  Advisory.

## Supported Versions

ns-sdn is pre-1.0. Only the latest prerelease receives security updates;
there are no maintenance branches before 1.0.0.

| Version             | Supported |
| ------------------- | --------- |
| Latest 1.0.0 prerelease | Yes   |
| 0.0.x               | No (name reservation, no functionality) |

Once 1.0.0 ships, this table will track supported major lines.

## Threat Model

ns-sdn is a numerical **compute library**, not a service and not a client.
It reads in-memory arrays, trains a small recurrent network with PyTorch,
and returns typed result records. The consequences:

- **The library performs no network I/O, opens no files it was not given,
  and holds no credentials.** Network access exists only in the example
  scripts, which download public FRED series through `fedfred` with a key
  read from the environment.
- **The attack surface is data, not traffic.** The two risks with real
  consequences are deserializing an untrusted pickled result, which, like
  every `pickle` load, can execute arbitrary code, and resource exhaustion
  from adversarial specifications.
- **Inputs are validated at the specification boundary.** An invalid
  `NSSDN` cannot be constructed: shape, length, finiteness, grid
  monotonicity and every categorical option are checked before any
  computation, and each failure raises a specific exception with the
  offending value.

## Vulnerability Classes and Mitigations

### Pickled results

`NSSDNResult` pickles through the network's state dict stored as numpy
arrays plus the configuration that rebuilds the module. Loading a pickle
from an untrusted source can execute arbitrary code during unpickling; this
is a property of `pickle`, not of ns-sdn. **Treat a pickled result as
executable code and load it only from sources you trust.** ns-sdn never
deserializes anything on its own: there is no load-from-path API, no cache,
and no network input. A result reloaded from a pickle rebuilds the module
from its configuration with `load_state_dict`, which accepts tensors only.

### Resource exhaustion

Cost scales with the sample length times the schedule length, and memory
with `state_dim` and `k`. Pathological values (a million-step schedule, a
state dimension in the tens of thousands) exhaust CPU or memory rather than
raise, because there is no principled bound to validate against. Validation
rejects non-positive and non-integer sizes and non-finite penalty weights;
callers running untrusted specifications should bound `k`, `state_dim` and
the total steps themselves. A fit that diverges to a non-finite loss raises
`NumericalError` immediately instead of running the schedule out.

### Malformed input

Non-finite values, wrong ranks, misaligned covariates and non-increasing
grids are rejected at construction with `DimensionError`,
`SpecificationError` or `NumericalError`. Exceptions describe shapes,
lengths and the offending option value; they never echo data payloads.

### The torch dependency

PyTorch is the package's one heavy dependency and the one with native code.
ns-sdn uses a small, fixed subset of it: `nn.GRU`, `nn.Linear`, the
elementwise activations, `cumsum`, `rfft`, Adam, and `state_dict` /
`load_state_dict`. It does not call `torch.load`, does not compile or JIT
anything, and does not download models or weights. Torch advisories are
tracked through Dependabot and `pip-audit`; the lower bound in
`pyproject.toml` is raised when an advisory warrants it.

### The examples and the FRED key

The DGS10 examples and the reproduction script read `FRED_API_KEY` from the
environment (or from `fedfred.set_api_key`) and pass it to `fedfred`, which
enforces TLS. The key is never written to disk by ns-sdn, never logged, and
never appears in a traceback. Build artifacts are checked: the release
workflows refuse to publish an sdist that contains a notebook, because
notebooks are where credentials end up by accident.

### Build and release supply chain

Releases are published from pinned GitHub Actions workflows through PyPI
**trusted publishing (OIDC)**; no long-lived upload tokens exist. Every
artifact attached to a GitHub release carries a Sigstore signature bundle
(`.sigstore.json`) and a detached GPG signature (`.asc`). The release
workflows run the test suite against the tagged tree before building,
verify that the tag names the built version, and verify the GPG signatures
after producing them.

## Security Measures

- Strict typing checked by `mypy`; lint by `ruff` (`E`, `F`, `W`, `I`, `B`,
  `UP`, `SIM`, `RUF`, `ANN`, `D`) and Bandit; no bare `except`.
- Defensive validation with specific exception classes at every public
  boundary.
- A layered engine: `engine/_core` imports only numpy, torch is confined to
  `engine/_internals`, and the public modules depend on the engine and not
  the reverse.
- Explicit seeding through `torch.random.fork_rng`; nothing touches the
  global seeds.
- CodeQL, Dependabot, `pip-audit`, dependency review and OpenSSF Scorecard
  in CI.
- Trusted publishing, Sigstore and GPG signing of release artifacts, and
  signed tags.

## Secure Software Delivery

### PyPI

- All PyPI traffic uses HTTPS.
- ns-sdn is published only through trusted publishing from the
  `prerelease.yml` and `release.yml` workflows in this repository.
- Install with `pip install ns-sdn` or `uv add ns-sdn`; pin the exact
  version in anything that must reproduce.

### conda-forge

- Once the feedstock exists, packages are built by conda-forge CI from the
  public recipe and never uploaded from a personal machine; every artifact
  carries a SHA-256 checksum.
- Install with `conda install -c conda-forge ns-sdn`.

### GitHub Releases

- Tags are signed; GitHub shows a "Verified" badge on them.
- Every release attaches the sdist, the wheel, their `.asc` GPG signatures
  and their `.sigstore.json` bundles.

## Verifying a Release

**The signed tag.**

```sh
git verify-tag v1.0.0a1
```

**The GPG signature on an artifact.** The maintainer's public key is on the
GitHub profile (`https://github.com/nikhilxsunder.gpg`) and on
`keys.openpgp.org`.

```sh
curl -fsSL https://github.com/nikhilxsunder.gpg | gpg --import
gpg --verify ns_sdn-1.0.0a1.tar.gz.asc ns_sdn-1.0.0a1.tar.gz
```

**The Sigstore bundle**, which ties the artifact to the workflow that built
it:

```sh
pip install sigstore
sigstore verify github ns_sdn-1.0.0a1.tar.gz \
  --bundle ns_sdn-1.0.0a1.tar.gz.sigstore.json \
  --repository nikhilxsunder/ns-sdn
```

## Security Updates and Announcements

- GitHub Security Advisories.
- GitHub release notes and `CHANGELOG.md`.
- The documentation site.

## Security Design Principles

Adapted from Saltzer and Schroeder to a research compute library:

1. **Economy of mechanism.** One model, two runtime dependencies, a small
   fixed subset of torch, and a layered engine keep the trusted base small.
2. **Fail-safe defaults.** Strict validation, explicit seeding, a fit that
   raises on divergence rather than returning garbage.
3. **Complete mediation.** Every input is validated at the specification
   boundary before any computation.
4. **Open design.** MIT-licensed, fully open; correctness is established by
   reproducing the published run and the papers' tables, never by obscurity.
5. **Separation of privilege.** The library holds no credentials; the one
   example that needs a key reads it from the environment.
6. **Least privilege.** No network, no filesystem writes, no subprocesses
   in the library.
7. **Least common mechanism.** No global state; seeds are scoped with
   `fork_rng`.
8. **Psychological acceptability.** The cultivars model-fit-result pattern
   makes the safe path the default path.
9. **Limited attack surface.** Optional capabilities (dataframes, FRED,
   plotting) are extras.
10. **Input validation with allowlists.** Every categorical option is
    checked against its `Literal` alias; every size is checked for type and
    range.

---

Questions about this policy: nsunder724@gmail.com.
