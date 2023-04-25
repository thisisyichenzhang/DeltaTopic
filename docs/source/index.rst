|PyPI| |Docs|

Documentaiton for BALSAM and DeltaTopic
======================================

**BALSAM**, short for Bayesian Latent topic analysis with Sparse Association Matrix, is a Bayesian topic modelling approach to summarize static transcriptome patterns from raw gene expression count data. **BALSAM** views cells as an admixture of gene topics, and relies on Variational AutoEncoder (VAE) and sparse-inducing priors to learn cell topics and infer the gell-topic relationship. 

**DeltaTopic** is a extension of **BALSAM** model to ascertain common cellular topic space and topic-specific relationships between the unspliced and spliced data. The detailed description of both methods in preprint `Zhang et al <https://www.biorxiv.org/content/10.1101/2023.03.11.532182v1.abstract>`_.

.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   about
   installation
   api
   references

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :hidden:

   getting_started

.. |PyPI| image:: https://badge.fury.io/py/DeltaTopic.svg
   :target: https://pypi.org/project/DeltaTopic/

.. |Docs| image:: https://readthedocs.org/projects/deltatopic/badge/?version=latest
   :target: https://deltatopic.readthedocs.io