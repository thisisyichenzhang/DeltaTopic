Installation
------------

DeltaTopic requires Python 3.8 or later. We recommend to use Miniconda_. 

PyPI
^^^^

Install DeltaTopic from PyPI_ using::

    pip install DeltaTopic

Development Version
^^^^^^^^^^^^^^^^^^^

To work with the latest development version, install from GitHub_ using::

    git clone git clone https://github.com/causalpathlab/deltaTopic.git && deltaTopic
    python setup.py build
    python setup.py install

Dependencies
^^^^^^^^^^^^

- `torch <https://pytorch.org/>`_, `pytroch-lightning <https://pytorch-lightning.readthedocs.io/en/latest/>`_ - deep learning framework.
- `anndata <https://anndata.readthedocs.io/>`_ - annotated data object.
- `scanpy <https://scanpy.readthedocs.io/>`_ - toolkit for single-cell analysis.
- `numpy <https://docs.scipy.org/>`_, `scipy <https://docs.scipy.org/>`_ 

Analysis pipeline
^^^^^^^^^^^^^^^^^

The analyis in `Zhang et al., 2023 <https://www.biorxiv.org/content/10.1101/2023.03.11.532182v1.abstract>`_ requires the following R packages:

- data.table, dplyr - data wrangling
- goseq, fgsea - gene set enrichment analysis
- ggplot2, ComlexHeatmap, circlize - visualization

To reproduce the analyis in the paper, follow Rmd files in the `project repository <https://github.com/causalpathlab/DeltaTopic/tree/main/R_figures/>`_.

If you run into issues, do not hesitate to approach us or raise a `GitHub issue`_.

.. _Miniconda: http://conda.pydata.org/miniconda.html
.. _PyPI: https://pypi.org/project/DeltaTopic
.. _Github: https://github.com/causalpathlab/deltaTopic
.. _`Github issue`: https://github.com/causalpathlab/deltaTopic/issues/new/choose
