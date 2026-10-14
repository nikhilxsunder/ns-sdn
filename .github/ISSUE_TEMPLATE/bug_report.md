---
name: Bug report
about: Something raises, returns the wrong shape, or disagrees with the documentation
title: "[BUG] "
labels: bug
assignees: ""
---

**What happened**
A clear description of the behaviour, and of what you expected instead.

**Minimal reproduction**
A short script that triggers it. Synthetic data is best; if the problem needs a particular series, say how to fetch it (for FRED, the series id and the date window) rather than attaching the data.

```python
import numpy as np
from ns_sdn.spectral_decomposition import NSSDN, Stage

y = ...
res = NSSDN(y, k=..., state_dim=..., schedule=(Stage(..., ...),), seed=0).fit()
...
```

**Traceback or output**
The full traceback, or the printed output that is wrong. For a fit that completed, paste `print(res.summary())`.

```text

```

**Specification**
The arguments that differ from the defaults (alignment, transition, ordered_frequency, gating, objective, penalties, schedule, scaling, device), and whether the series had covariates or an irregular time grid.

**Environment**

- ns-sdn version (`python -c "import ns_sdn.__about__ as a; print(a.__version__)"`):
- Python version:
- torch version (`python -c "import torch; print(torch.__version__)"`) and device (CPU, CUDA, MPS):
- numpy version:
- OS and architecture:
- Installed from (PyPI wheel, conda-forge, source):

**Additional context**
Anything else: whether it reproduces with another seed, a shorter schedule, or the `predicted` alignment; whether it is new in this version.
