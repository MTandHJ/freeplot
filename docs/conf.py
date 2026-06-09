"""Sphinx configuration for FreePlot documentation."""

import os
import sys
from importlib.util import find_spec

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, os.path.abspath(".."))

project = "FreePlot"
copyright = "2026, MTandHJ"
author = "MTandHJ"
release = "0.5.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

if find_spec("sphinxcontrib.mermaid") is not None:
    extensions.append("sphinxcontrib.mermaid")

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "zh_CN"
locale_dirs = ["locales/"]
gettext_compact = False

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

autosummary_generate = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
}

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]
html_theme_options = {
    "logo": {
        "text": "FreePlot",
    },
    "header_links_before_dropdown": 6,
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "footer_start": ["copyright"],
    "footer_end": ["theme-version"],
    "pygments_light_style": "default",
    "pygments_dark_style": "monokai",
    "navbar_end": ["theme-switcher", "navbar-icon-links", "components/lang-switcher"],
}
