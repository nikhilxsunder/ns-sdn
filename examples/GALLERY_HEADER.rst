Examples
========

Worked examples for ns-sdn. Each script is executed at documentation-build
time, so the figures and printed output are produced by the code shown.
Scripts are plain Python with reST in the module docstring and in ``# %%``
cell comments; run any of them directly with ``python examples/plot_<name>.py``.
The first needs no network; the second and third download the 10-year
Treasury yield from FRED and need a ``FRED_API_KEY``. The third is the
paper's full protocol, which takes longer than a documentation build should,
so the build renders it without running it unless ``DOCS_RUN_ORSS=1`` is set;
run it directly to reproduce the published metrics and figures.
