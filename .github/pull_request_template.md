## Summary

<!-- What this PR does and why. Focus on the user-visible behaviour in ns-sdn. -->

## Related issues

<!-- e.g. Closes #12, Related to #34 -->

## Type of change

- [ ] Bug fix
- [ ] New feature (option, path, coordinate, penalty, or result method)
- [ ] Numerical change to the model or the trainer
- [ ] Performance improvement
- [ ] Documentation, examples or papers
- [ ] Refactor / internal change
- [ ] CI / tooling / packaging

## Public API impact

- [ ] No public API changes
- [ ] Adds public API (arguments, fields, methods, or return types)
- [ ] Changes existing public API
- [ ] Removes or deprecates public API

<!-- If there are API changes, state them, including anything breaking. -->

## Fidelity to the published model

- [ ] The default `NSSDN()` still builds the IEEE-ORSS configuration (the 58,849-parameter test passes)
- [ ] Any new behaviour is behind an option that is off by default
- [ ] The identity `trend + components.sum(axis=1) == emission` still holds
- [ ] N/A: this PR does not touch library code

## Conventions

- [ ] Three-object discipline kept: specification in `NSSDN`, mechanism in `engine/_internals`, record in `NSSDNResult`
- [ ] `engine/_core` imports only numpy; torch stays in `engine/_internals`
- [ ] Inline `npt.NDArray[np.float64]` annotations, no module-level array aliases
- [ ] Full MIT header with the `# filepath:` line on any new `.py` file
- [ ] Every raised error is a `DimensionError`, `SpecificationError` or `NumericalError` with the offending value in the message

## How to test

```bash
uv run pytest                                  # doctests + behavioural tests, with coverage
uv run ruff check src/ns_sdn tests examples
uv run ruff format --check src/ns_sdn tests examples
uv run mypy
DOCS_PLOT_GALLERY=0 uv run make -C docs html   # if the docs changed
```

## Checklist

- [ ] Tests added or updated and passing locally
- [ ] Docstrings (Google style, with a doctest where useful) updated for any changed behaviour
- [ ] `CHANGELOG.md` updated under _Unreleased_
- [ ] No generated files committed (`docs/build`, `auto_examples`, `htmlcov`, `dist`)

## Additional context

<!-- References implemented (with citation), known limitations, follow-up work. -->
