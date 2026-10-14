---
name: Feature request
about: Propose an addition to the model, the result, the spectral state, or the tooling
title: "[FEATURE] "
labels: enhancement
assignees: ""
---

**The problem**
What you are trying to do with ns-sdn that it does not let you do, or makes awkward. A concrete analysis or workflow is more useful than an abstract capability.

**Proposed solution**
What you would like to exist, as the call you would make. For a new path, coordinate, penalty or option, say which object it belongs on (`NSSDN`, `NSSDNResult`, `SpectralState`, `Penalties`, `Stage`) and what it should return.

```python
res = NSSDN(y, ...).fit()
res.new_thing(...)
```

**Alternatives considered**
What you tried with the current API, and why it falls short. Note if `decompose(scaled=True)`, `spectral_state()`, `to_pandas()` or the training `history` already get most of the way.

**Reference**
If the idea comes from a paper or an existing implementation, cite it (title, authors, year, DOI or URL) so the design can be checked against it.

**Scope**
- [ ] Changes the published model (the IEEE-ORSS defaults must keep reproducing the 58,849-parameter run)
- [ ] Adds a new option that is off by default
- [ ] Only touches the result, the spectral state, or the reporting surface
- [ ] Documentation, examples or tooling only

**Additional context**
Anything else that helps, including whether you would be willing to open a pull request.
