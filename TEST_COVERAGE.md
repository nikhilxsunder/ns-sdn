# Test Coverage Report

Last updated: 2026-10-14

## Coverage Summary

Overall coverage: 79%

## Detailed Coverage

```
============================= test session starts ==============================
platform linux -- Python 3.14.8, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/runner/work/ns-sdn/ns-sdn
configfile: pyproject.toml
testpaths: src/ns_sdn, tests
plugins: platformdirs-4.13.0, hypothesis-6.169.1, cov-7.1.0, anyio-4.15.1
collected 18 items

src/ns_sdn/__init__.py .                                                 [  5%]
src/ns_sdn/__about__.py .                                                [ 11%]
src/ns_sdn/engine/_core/_containers.py .                                 [ 16%]
src/ns_sdn/engine/_core/_validators.py ....                              [ 38%]
src/ns_sdn/exceptions.py .                                               [ 44%]
src/ns_sdn/spectral_decomposition.py .......                             [ 83%]
src/ns_sdn/spectral_state.py .                                           [ 88%]
src/ns_sdn/summary.py .                                                  [ 94%]
src/ns_sdn/typing.py .                                                   [100%]

================================ tests coverage ================================
_______________ coverage: platform linux, python 3.14.8-final-0 ________________

Name                                          Stmts   Miss  Cover   Missing
---------------------------------------------------------------------------
src/ns_sdn/__about__.py                           9      0   100%
src/ns_sdn/__init__.py                            3      0   100%
src/ns_sdn/engine/__init__.py                     3      0   100%
src/ns_sdn/engine/_core/__init__.py               3      0   100%
src/ns_sdn/engine/_core/_containers.py          133     41    69%   108, 112, 142-147, 182, 200, 230, 244, 252, 266-279, 283-318, 326-339, 343
src/ns_sdn/engine/_core/_defaults.py             26      0   100%
src/ns_sdn/engine/_core/_types.py                11      0   100%
src/ns_sdn/engine/_core/_validators.py           65     28    57%   71, 73, 75, 98-107, 132-139, 153, 166, 169, 179-182
src/ns_sdn/engine/_internals/__init__.py          3      0   100%
src/ns_sdn/engine/_internals/_fits.py            18      0   100%
src/ns_sdn/engine/_internals/_mixins.py          64     28    56%   62, 68, 71, 85-86, 94, 105-113, 130, 135, 140, 145, 150, 167-187
src/ns_sdn/engine/_internals/_models.py         133      9    93%   187, 190, 193, 196, 235, 255, 270, 275, 302
src/ns_sdn/engine/_internals/_modules.py        102     15    85%   91-92, 100, 135, 151, 174, 189-194, 204-205, 211
src/ns_sdn/engine/_internals/_objectives.py      37     12    68%   47, 53-55, 91-92, 94-95, 97-98, 100-101
src/ns_sdn/engine/_internals/_predictors.py      96     17    82%   131, 141, 173, 184-186, 263, 272, 289, 303-310
src/ns_sdn/engine/_internals/_trainers.py        33      1    97%   108
src/ns_sdn/exceptions.py                          5      0   100%
src/ns_sdn/spectral_decomposition.py            141     39    72%   293, 319, 324, 329-330, 335, 340-342, 401-402, 415-429, 458-462, 493, 499, 527, 531-567
src/ns_sdn/spectral_state.py                     92     15    84%   229, 234, 238-239, 257-265, 268-283
src/ns_sdn/summary.py                             8      0   100%
src/ns_sdn/typing.py                              3      0   100%
---------------------------------------------------------------------------
TOTAL                                           988    205    79%
Coverage HTML written to dir htmlcov
Coverage XML written to file coverage.xml
============================= 18 passed in 16.34s ==============================
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
