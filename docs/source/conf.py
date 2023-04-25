# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from pathlib import Path, PurePosixPath

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent))
# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DeltaTopic'
copyright = '2023, Yichen Zhang'
author = 'Yichen Zhang'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc", 
              "sphinx.ext.viewcode", 
              "sphinx.ext.autosummary",
              "sphinx.ext.napoleon",
              "sphinx_toolbox.decorators",
]

# Generate the API documentation when building
autosummary_generate = True
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# manually add the path for python packages
autodoc_mock_imports = ["torch", "anndata", "pytorch_lightning", "h5py", "scanpy", "numpy", "scipy", "pandas"]