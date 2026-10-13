# Test Coverage Report

Last updated: 2026-10-13

## Coverage Summary

Overall coverage: 91%

## Detailed Coverage

```
============================= test session starts ==============================
platform linux -- Python 3.14.8, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/runner/work/ns-sdn/ns-sdn
configfile: pyproject.toml
testpaths: src/ns_sdn, tests
plugins: hypothesis-6.169.0, platformdirs-4.13.0, cov-7.1.0, anyio-4.15.1
collected 33 items

src/ns_sdn/__init__.py .                                                 [  3%]
src/ns_sdn/__about__.py .                                                [  6%]
src/ns_sdn/engine/_core/_containers.py .                                 [  9%]
src/ns_sdn/engine/_core/_validators.py ....                              [ 21%]
src/ns_sdn/exceptions.py .                                               [ 24%]
src/ns_sdn/spectral_decomposition.py .......                             [ 45%]
src/ns_sdn/spectral_state.py .                                           [ 48%]
src/ns_sdn/summary.py .                                                  [ 51%]
src/ns_sdn/typing.py .                                                   [ 54%]
tests/test_spectral_decomposition.py ...............                     [100%]

================================ tests coverage ================================
_______________ coverage: platform linux, python 3.14.8-final-0 ________________

Name                                          Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------
src/ns_sdn/__about__.py                           9      0   100%
src/ns_sdn/__init__.py                            3      0   100%
src/ns_sdn/engine/__init__.py                     3      0   100%
src/ns_sdn/engine/_core/__init__.py               3      0   100%
src/ns_sdn/engine/_core/_containers.py          133     30    77%   93, 124, 126, 163, 181, 211, 233, 264-299, 307-320, 324
src/ns_sdn/engine/_core/_defaults.py             26      0   100%
src/ns_sdn/engine/_core/_types.py                11      0   100%
src/ns_sdn/engine/_core/_validators.py           65     15    77%   54, 56, 81, 83, 85, 87, 115, 117, 119, 134, 147, 160-163
src/ns_sdn/engine/_internals/__init__.py          3      0   100%
src/ns_sdn/engine/_internals/_fits.py            18      0   100%
src/ns_sdn/engine/_internals/_mixins.py          64     12    81%   43, 52, 66-67, 75, 92, 111, 121, 126, 131, 149, 154
src/ns_sdn/engine/_internals/_models.py         133      9    93%   168, 171, 174, 177, 216, 236, 251, 256, 283
src/ns_sdn/engine/_internals/_modules.py        102      2    98%   116, 192
src/ns_sdn/engine/_internals/_objectives.py      37      0   100%
src/ns_sdn/engine/_internals/_predictors.py      96      2    98%   112, 122
src/ns_sdn/engine/_internals/_trainers.py        33      1    97%   89
src/ns_sdn/exceptions.py                          5      0   100%
src/ns_sdn/spectral_decomposition.py            141      4    97%   302, 471, 477, 533
src/ns_sdn/spectral_state.py                     92      9    90%   219-220, 238-246
src/ns_sdn/summary.py                             8      0   100%
src/ns_sdn/typing.py                              3      0   100%
---------------------------------------------------------------------------
TOTAL                                           988     84    91%
Coverage HTML written to dir htmlcov
Coverage XML written to file coverage.xml
============================= 33 passed in 29.09s ==============================
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
