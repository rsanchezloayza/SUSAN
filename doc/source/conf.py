# Configuration file for the Sphinx documentation builder.
#
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import stat
import sys

# ---------------------------------------------------------------------------
# Make the susan package importable from the project root.
# susan.__init__ calls _add_susan_bin_to_path() at import time and raises
# ImportError if no binary is found.  During a doc build the CUDA binaries
# are not compiled, so we create a dummy executable so the check passes.
# ---------------------------------------------------------------------------
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, _root)

_dummy_bin = os.path.join(_root, 'bin', 'susan_aligner')
if not os.path.exists(_dummy_bin):
    os.makedirs(os.path.dirname(_dummy_bin), exist_ok=True)
    open(_dummy_bin, 'w').close()
    os.chmod(_dummy_bin, os.stat(_dummy_bin).st_mode | stat.S_IEXEC)

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project   = 'SUSAN'
copyright = '2019-2026, Ricardo M. Sánchez L.'
author    = 'Ricardo M. Sánchez L.'
release   = '0.9'
version   = release

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',        # pull docstrings from Python source
    'sphinx.ext.napoleon',       # NumPy / Google docstring styles
    'sphinx.ext.viewcode',       # [source] links next to each object
    'sphinx.ext.mathjax',        # math rendering
    'sphinx_copybutton',         # copy button on code blocks
    'sphinx_design',             # tab-set, cards, and grid directives
]

templates_path   = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
language         = 'en'
source_suffix    = '.rst'

# ---------------------------------------------------------------------------
# autodoc
# ---------------------------------------------------------------------------
# The compiled Cython extensions are not available during a doc build.
# Sphinx will replace them with MagicMock objects so import succeeds.
autodoc_mock_imports = [
    'susan.data._particles_core',
    'susan.data._ptclsgeom_core',
    'susan.utils._functions_core',
    'torch',
]

autodoc_member_order   = 'bysource'   # preserve source order, not alphabetical
autodoc_default_options = {
    'members':         True,
    'undoc-members':   False,
    'show-inheritance': True,
}

# ---------------------------------------------------------------------------
# napoleon (docstring style)
# ---------------------------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring  = True
napoleon_use_param        = True
napoleon_use_rtype        = True
napoleon_use_ivar         = False  # render Attributes sections as .. attribute:: directives;
                                   # this lets .. rubric:: headings between multiple Attributes
                                   # sections render correctly.  All class stubs use :no-members:
                                   # so there is no risk of autodoc duplicating the entries.

# ---------------------------------------------------------------------------
# HTML output — furo theme with light/dark logos
# ---------------------------------------------------------------------------
html_theme       = 'furo'
html_static_path = ['_static']

html_favicon = '_static/icon.ico'
html_extra_head = [
    '<link rel="icon" type="image/svg+xml" href="_static/logo.svg">'
]

html_theme_options = {
    "light_logo": "logo_light.svg",
    "dark_logo":  "logo_dark.svg",
}

html_css_files = ['custom.css']
