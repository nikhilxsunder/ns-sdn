"""Sphinx configuration file for the ns-sdn project."""
# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import importlib
import inspect
import os
import sys
import tomllib
from pathlib import Path
from types import FunctionType, MethodType, ModuleType
from typing import TypeAliasType

from sphinx.application import Sphinx
from sphinx.pycode import ModuleAnalyzer

# -- Path setup --------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))

# -- Project information -----------------------------------------------------
with open(_ROOT / "pyproject.toml", "rb") as _f:
    _META = tomllib.load(_f)["project"]

from ns_sdn.__about__ import __version__  # noqa: E402

project: str = "ns-sdn"
copyright: str = "2026, Nikhil Sunder"
author: str = "Nikhil Sunder"
release: str = __version__
version: str = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------
extensions: list[str] = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.extlinks",
    "sphinx.ext.doctest",
    "sphinx_autodoc_typehints",
    "sphinx_design",
    "sphinx_sitemap",
    "sphinxext.opengraph",
    "myst_parser",
    "sphinx_codeautolink",
    "sphinx_copybutton",
    "sphinx_remove_toctrees",
    "sphinx_last_updated_by_git",
    "sphinxcontrib.bibtex",
    "matplotlib.sphinxext.plot_directive",
    "sphinx_gallery.gen_gallery",
    "sphinx_tippy",
]

templates_path: list[str] = ["_templates"]
exclude_patterns: list[str] = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    # sphinx-gallery writes notebooks and backreference stubs next to the
    # rendered pages; they are outputs, not sources.
    "auto_examples/**/*.ipynb",
    "auto_examples/**/*.py",
    "gen_modules/backreferences/*.examples",
]

source_suffix: dict[str, str] = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Warnings never fail a build here; ``make strict`` adds ``-W``. Private
# cross-refs (``_SpectralPredictor``) are intentionally undocumented on the
# public pages, so nitpicky stays off.
nitpicky: bool = False
suppress_warnings: list[str] = [
    # Dataclass fields whose annotation names a private type imported in the
    # defining module; the rendered type falls back to the literal string.
    "sphinx_autodoc_typehints.forward_reference",
    # Relative Markdown links inside the included root CHANGELOG.
    "myst.xref_missing",
    # Insurance against an unpicklable config value disabling the env cache.
    "config.cache",
]

# sphinx -b doctest runs only explicit .. testcode:: / .. doctest:: blocks.
# Bare >>> blocks in docstrings are pytest --doctest-modules' job, where each
# docstring gets its own globals.
doctest_test_doctest_blocks: str = ""

# -- MyST --------------------------------------------------------------------
myst_enable_extensions: list[str] = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "linkify",  # requires linkify-it-py in the docs dependency group
    "replacements",
    "smartquotes",
    "substitution",
    "tasklist",
]
myst_heading_anchors: int = 4

# -- autodoc / autosummary ---------------------------------------------------
autosummary_generate: bool = True
# The top-level ``__init__`` re-exports modules only, so honouring ``__all__``
# documents each name under its leaf module (ns_sdn.spectral_decomposition.NSSDN).
autosummary_ignore_module_all: bool = False
autosummary_imported_members: bool = False

# Member selection lives in the autosummary templates (``_templates/autosummary``)
# rather than here: the module pages must not re-document what the class
# pages already own.
autodoc_default_options: dict[str, str] = {
    "member-order": "groupwise",
    "exclude-members": "__init__, __new__, __weakref__",
}
autodoc_typehints: str = "description"
autodoc_typehints_format: str = "short"
autodoc_typehints_description_target: str = "documented_params"
autodoc_class_signature: str = "mixed"
autodoc_member_order: str = "groupwise"
autodoc_preserve_defaults: bool = True

# sphinx_autodoc_typehints
typehints_fully_qualified: bool = False
always_document_param_types: bool = False
typehints_document_rtype: bool = True
typehints_use_signature: bool = False
typehints_use_signature_return: bool = False
simplify_optional_unions: bool = True

# -- napoleon (Google style, matches ruff pydocstyle convention) --------------
napoleon_google_docstring: bool = True
napoleon_numpy_docstring: bool = False
napoleon_include_init_with_doc: bool = False
napoleon_include_private_with_doc: bool = False
napoleon_use_admonition_for_examples: bool = True
napoleon_use_admonition_for_notes: bool = True
napoleon_use_admonition_for_references: bool = False
napoleon_use_ivar: bool = True  # dataclass fields: avoid duplicate attribute entries
napoleon_use_param: bool = True
napoleon_use_rtype: bool = False
napoleon_preprocess_types: bool = True
napoleon_attr_annotations: bool = True
napoleon_custom_sections: list[tuple[str, str]] = [
    ("Shapes", "params_style"),
]

# -- intersphinx -------------------------------------------------------------
intersphinx_mapping: dict[str, tuple[str, str | None]] = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "fedfred": ("https://nikhilxsunder.github.io/fedfred/", None),
    "cultivars": ("https://nikhilxsunder.github.io/cultivars/stable/", None),
}
intersphinx_timeout: int = 10

# -- extlinks ----------------------------------------------------------------
extlinks: dict[str, tuple[str, str]] = {
    "github": ("https://github.com/nikhilxsunder/ns-sdn/%s", "GitHub: %s"),
    "issue": ("https://github.com/nikhilxsunder/ns-sdn/issues/%s", "issue %s"),
    "doi": ("https://doi.org/%s", "doi:%s"),
    "fred": ("https://fred.stlouisfed.org/series/%s", "FRED: %s"),
    "numpy-doc": ("https://numpy.org/doc/stable/reference/%s", "NumPy Docs: %s"),
    "torch-doc": ("https://docs.pytorch.org/docs/stable/%s", "PyTorch Docs: %s"),
}

# -- switcher ---------------------------------------------------------------

_DOCS_VERSION: str = os.environ.get("DOCS_VERSION", "dev")  # dev | stable | <release>
_DOCS_GIT_REF: str = os.environ.get("DOCS_GIT_REF", "main")  # branch or tag, for Edit on GitHub
# Origin and path prefix the site is served from, with the live site as the
# default. The deploy workflow passes its DOCS_BASE_URL; ``make preview``
# passes http://localhost:<port>/ and mirrors the live layout (switcher.json
# at the root, one directory per version), so the same switcher works in
# both. Canonical-URL metadata (html_baseurl, opengraph, sitemap) stays on
# the live stable URL regardless.
_DOCS_SITE_URL: str = os.environ.get("DOCS_SITE_URL", "https://nikhilxsunder.github.io/ns-sdn/")
if not _DOCS_SITE_URL.endswith("/"):
    _DOCS_SITE_URL += "/"

# Every build declares stable as canonical: dev and archives consolidate to it.
html_baseurl: str = "https://nikhilxsunder.github.io/ns-sdn/stable/"

# Only stable publishes a sitemap; nothing else is meant to be indexed.
if _DOCS_VERSION != "stable":
    extensions.remove("sphinx_sitemap")

# -- HTML output -------------------------------------------------------------
html_last_updated_fmt: str = "%Y-%m-%d"

html_theme: str = "pydata_sphinx_theme"
html_title: str = "ns-sdn"
html_logo: str = "_static/ns-sdn-logo.png"
html_favicon: str = "_static/ns-sdn-favicon.ico"
html_static_path: list[str] = ["_static"]
# robots.txt and the IndexNow key live at the site root and are written by
# the deploy workflow's ``root`` job from docs/root/ and a secret.
html_extra_path: list[str] = []
html_css_files: list[str] = ["custom.css"]
html_show_sourcelink: bool = False

html_theme_options: dict[str, object] = {
    "analytics": {"google_analytics_id": "G-T9HLR97XCS"},
    "logo": {
        "image_light": "_static/ns-sdn-logo.png",
        "image_dark": "_static/ns-sdn-logo-dark.png",
    },
    "header_links_before_dropdown": 6,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": _DOCS_SITE_URL + "switcher.json",
        "version_match": _DOCS_VERSION,
    },
    "check_switcher": False,
    "show_version_warning_banner": False,
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navbar_align": "left",
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/nikhilxsunder/ns-sdn",
            "icon": "fab fa-github",
        },
        {"name": "PyPI", "url": "https://pypi.org/project/ns-sdn/", "icon": "fab fa-python"},
        {
            "name": "Codecov",
            "url": "https://app.codecov.io/gh/nikhilxsunder/ns-sdn",
            "icon": "fas fa-umbrella",
        },
    ],
    "use_edit_page_button": True,
    "show_toc_level": 2,
    "show_prev_next": True,
    "footer_start": ["ns-sdn-footer"],
    "footer_center": [],
    "footer_end": [],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "pygments_light_style": "friendly",
    "pygments_dark_style": "monokai",
}

html_context: dict[str, str] = {
    "github_user": "nikhilxsunder",
    "github_repo": "ns-sdn",
    "doc_path": "docs/source",
    "github_version": _DOCS_GIT_REF,
}

html_meta: dict[str, str] = {
    "description": _META["description"],
    "keywords": ", ".join(_META["keywords"]),
}

# -- sitemap -----------------------------------------------------------------
sitemap_filename: str = "sitemap.xml"
sitemap_url_scheme: str = "{link}"

# -- opengraph ---------------------------------------------------------------
ogp_site_url: str = html_baseurl
ogp_image: str = html_baseurl + "_static/ns-sdn-mark.png"
ogp_description_length: int = 300
ogp_type: str = "website"
ogp_enable_meta_description: bool = True
ogp_custom_meta_tags: list[str] = [
    '<meta property="og:locale" content="en_US" />',
    '<meta property="og:site_name" content="ns-sdn Documentation" />',
    '<meta property="og:image:alt" content="ns-sdn Logo" />',
]

# -- bibtex --
bibtex_bibfiles: list[str] = ["references.bib"]
bibtex_default_style: str = "unsrt"
bibtex_reference_style: str = "author_year"

# -- plot_directive --
plot_include_source: bool = True
plot_html_show_source_link: bool = False
plot_html_show_formats: bool = False
plot_formats: list[tuple[str, int]] = [("png", 150)]
plot_pre_code: str = "import numpy as np\nimport matplotlib.pyplot as plt\nimport ns_sdn"

# -- sphinx-gallery ----------------------------------------------------------
# Every value is picklable (sort keys by name, not by class) so Sphinx can
# cache the environment between builds.
#
# plot_03_reproduce_orss.py runs the published 3,000-step fit, hours on a CI
# runner, so it is rendered without execution unless DOCS_RUN_ORSS=1. Files
# that miss filename_pattern are still listed in the gallery, as code.
_RUN_ORSS: bool = os.environ.get("DOCS_RUN_ORSS", "0") == "1"
sphinx_gallery_conf: dict[str, object] = {
    "examples_dirs": [str(_ROOT / "examples")],  # needs examples/GALLERY_HEADER.rst
    "gallery_dirs": ["auto_examples"],  # relative to docs/source
    # plot_NN_name.py only; rejects "name 2.py" copies a sync service makes.
    "filename_pattern": r"/plot_\d{2}_\w+\.py$" if _RUN_ORSS else r"/plot_(?!03_)\d{2}_\w+\.py$",
    # Helpers (``_fred.py``) stay out of the gallery; the pattern is searched
    # in the bare file name.
    "ignore_pattern": r"^_",
    "within_subsection_order": "ExampleTitleSortKey",
    "doc_module": ("ns_sdn",),
    "reference_url": {"ns_sdn": None},  # None = link into this build
    "backreferences_dir": "gen_modules/backreferences",
    "image_scrapers": ("matplotlib",),
    "reset_modules": ("matplotlib",),
    "capture_repr": ("_repr_html_", "__repr__"),
    "matplotlib_animations": False,
    "remove_config_comments": True,
    "download_all_examples": False,
    "show_memory": False,
    "show_signature": False,
    "min_reported_time": 10,
    # A failing example is rendered with its traceback and reported as a
    # Sphinx warning; it fails the build only under -W (make strict).
    "abort_on_example_error": False,
    "only_warn_on_example_error": True,
    "plot_gallery": os.environ.get("DOCS_PLOT_GALLERY", "1") == "1",
    "default_thumb_file": str(_ROOT / "docs" / "source" / "_static" / "ns-sdn-mark.png"),
}

# Keep generated backreference stubs out of the sidebar toctree.
remove_from_toctrees: list[str] = ["gen_modules/backreferences/*"]

# -- sphinx-tippy ------------------------------------------------------------
tippy_props: dict[str, object] = {
    "placement": "auto-start",
    "maxWidth": 500,
    "theme": "material",
    "interactive": True,
    "delay": [200, 0],
}
tippy_anchor_parent_selector: str = "article.bd-article"  # pydata body only; skips navbar/sidebars
tippy_skip_anchor_classes: list[str] = ["headerlink", "sd-stretched-link"]
tippy_enable_mathjax: bool = True  # tips for equation references
tippy_enable_wikitips: bool = False  # build-time network fetch; low value here
tippy_enable_doitips: bool = False
tippy_add_class: str = "has-tippy"

# -- doctest -----------------------------------------------------------------
doctest_global_setup: str = "import numpy as np\nimport ns_sdn"


# -- linkcode ----------------------------------------------------------------
def linkcode_resolve(domain: str, info: dict[str, str]) -> str | None:
    """Link each documented object to its source lines on GitHub."""
    if domain != "py" or not info["module"]:
        return None
    obj: object = importlib.import_module(info["module"])
    for part in info["fullname"].split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    if isinstance(obj, property):
        obj = obj.fget
    if isinstance(obj, (FunctionType, MethodType)):
        target: ModuleType | type[object] | FunctionType | MethodType = inspect.unwrap(obj)
    elif isinstance(obj, (ModuleType, type)):
        target = obj
    else:
        return None
    try:
        source_file = inspect.getsourcefile(target)
        lines, first = inspect.getsourcelines(target)
    except (OSError, TypeError):
        return None
    if source_file is None:
        return None
    try:
        relative = Path(source_file).resolve().relative_to(_ROOT)
    except ValueError:
        return None
    last = first + len(lines) - 1
    return (
        f"https://github.com/nikhilxsunder/ns-sdn/blob/{_DOCS_GIT_REF}"
        f"/{relative.as_posix()}#L{first}-L{last}"
    )


# -- autodoc hooks -----------------------------------------------------------
def _document_type_alias(
    app: Sphinx,
    what: str,
    name: str,
    obj: object,
    options: dict[str, object],
    lines: list[str],
) -> None:
    """Document PEP 695 ``type`` aliases re-exported through ``ns_sdn.typing``.

    A ``TypeAliasType`` carries no ``__doc__`` of its own, so autodoc falls back
    to the generic ``typing.TypeAliasType`` docstring. Replace it with the
    docstring-comment that follows the alias in its defining module, and lead
    with the aliased value so the permitted literals are visible.
    """
    if what != "data" or not isinstance(obj, TypeAliasType):
        return
    value = repr(obj.__value__).replace("typing.", "")
    doc: list[str] = []
    module = obj.__module__
    if module is not None:
        try:
            attr_docs = ModuleAnalyzer.for_module(module).find_attr_docs()
            doc = list(attr_docs.get(("", obj.__name__), []))
        except Exception:  # pragma: no cover - analyzer failure is non-fatal
            doc = []
    lines[:] = [f"Alias of ``{value}``.", "", *doc]


def setup(app: Sphinx) -> None:
    """Setup hook for Sphinx."""
    app.connect("autodoc-process-docstring", _document_type_alias)
