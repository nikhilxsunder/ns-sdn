# Test Coverage Report

Last updated: 2026-10-14

## Coverage Summary

Overall coverage: 95%

## Detailed Coverage

```
================================ tests coverage ================================
_______________ coverage: platform linux, python 3.13.16-final-0 _______________

Name                                          Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------
src/ns_sdn/__about__.py                           9      0   100%
src/ns_sdn/__init__.py                            3      0   100%
src/ns_sdn/engine/__init__.py                     3      0   100%
src/ns_sdn/engine/_core/__init__.py               3      0   100%
src/ns_sdn/engine/_core/_containers.py          133     19    86%   190, 253, 255, 458, 501-536, 577
src/ns_sdn/engine/_core/_defaults.py             27      0   100%
src/ns_sdn/engine/_core/_types.py                11      0   100%
src/ns_sdn/engine/_core/_validators.py           65      4    94%   151, 155, 193, 264
src/ns_sdn/engine/_internals/__init__.py          3      0   100%
src/ns_sdn/engine/_internals/_fits.py            18      0   100%
src/ns_sdn/engine/_internals/_mixins.py          64      8    88%   112, 136-137, 171, 199, 214, 219, 259
src/ns_sdn/engine/_internals/_models.py         133      8    94%   238, 241, 244, 247, 306, 321, 326, 367
src/ns_sdn/engine/_internals/_modules.py        102      2    98%   255, 364
src/ns_sdn/engine/_internals/_objectives.py      37      0   100%
src/ns_sdn/engine/_internals/_predictors.py      96      1    99%   195
src/ns_sdn/engine/_internals/_trainers.py        33      1    97%   149
src/ns_sdn/exceptions.py                          5      0   100%
src/ns_sdn/spectral_decomposition.py            141      3    98%   462, 786, 879
src/ns_sdn/spectral_state.py                     92      7    92%   445-453
src/ns_sdn/summary.py                             8      0   100%
src/ns_sdn/typing.py                              3      0   100%
---------------------------------------------------------------------------
TOTAL                                           989     53    95%
Coverage HTML written to dir htmlcov
71 passed in 52.47s
```

## Running Test Coverage Locally

The doctests and the behavioral suite run together with coverage:

```bash
uv sync --group test
uv run pytest
```

Then open `htmlcov/index.html` in your browser for the line-level report.

## Coverage Goals

- Maintain at least 80% overall coverage
- All public APIs should have 100% coverage
- Focus on testing edge cases and error conditions
