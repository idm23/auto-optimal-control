"""Sphinx configuration for the auto-optimal-control documentation."""

from importlib.metadata import version as _version

project = "auto-optimal-control"
author = "Ian Mackey"
copyright = "2026, Ian Mackey"
release = _version("auto-optimal-control")
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
]

exclude_patterns = ["_build", "visuals/*.npz"]

# Notebooks are committed without outputs and executed at build time. The
# cache re-runs a notebook only when its code changes.
nb_execution_mode = "cache"
nb_execution_timeout = 1800
nb_execution_raise_on_error = True
nb_execution_show_tb = True
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence"]
myst_heading_anchors = 3

autodoc_member_order = "bysource"
autodoc_default_options = {"members": True, "show-inheritance": False}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sympy": ("https://docs.sympy.org/latest", None),
}

html_theme = "pydata_sphinx_theme"
html_title = "auto-optimal-control"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/idm23/auto-optimal-control",
    "navbar_align": "left",
    "show_toc_level": 2,
    "icon_links": [{"name": "PyPI", "url": "https://pypi.org/project/auto-optimal-control/",
                    "icon": "fa-brands fa-python"}],
}
html_context = {"default_mode": "light"}
