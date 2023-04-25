Getting Started
---------------

Here, you will be briefly guided through the basics of how to use **BALSAM** and **DeltaTopic**. 

The input data for BALSAM is one count matrice of RNA abundances, which can be obtained from standard sequencing protocols, using  `kallisto`_ counting pipeline.

BALSAM
^^^^^^^^^^^^^^^^^^^^^^^^^^^
Import BALSAM and DeltaTopic as::

    from DeltaTopic.nn.modelhub import BALSAM, DeltaTopic.
    
Read your data
''''''''''''''
Read your data file, for example, a h5ad file, using `scanpy`::
    
    import scanpy as sc
    adata = sc.read(filename)

which stores the data matrix (``adata.X``).

By defualt **BALSAM** uses ``adata.layers['counts']`` for training, so you will need to register your data via `setup_anndata`::
    
    from DeltaTopic.nn.util import setup_anndata
    setup_anndata(adata, layer="counts")
    
If you already have an existing preprocessed adata object you can simply merge the spliced/unspliced counts via::

    ldata = scv.read(filename.loom, cache=True)
    adata = scv.utils.merge(adata, ldata)

If you do not have a datasets yet, you can still play around using one of the in-built datasets, e.g.::

    adata = scv.datasets.pancreas()


BALSAM
''''''''''''''
The core of the software is the efficient and robust estimation of velocities, obtained with::

    scv.tl.velocity(adata, mode='stochastic', **params)

The velocities are vectors in gene expression space obtained by solving a stochastic model of transcriptional dynamics.
The solution to the deterministic model is obtained by setting ``mode='deterministic'``.

The solution to the dynamical model is obtained by setting ``mode='dynamical'``, which requires to run
``scv.tl.recover_dynamics(adata, **params)`` beforehand.

The velocities are stored in ``adata.layers`` just like the count matrices.

The velocities are projected into a lower-dimensional embedding by translating them into likely cell transitions.
That is, for each velocity vector we find the likely cell transitions that are in accordance with that direction.
The probabilities of one cell transitioning into another cell are computed using cosine correlation
(between the potential cell transition and the velocity vector) and are stored in a matrix denoted as velocity graph::

    scv.tl.velocity_graph(adata, **params)

Visualization
'''''''''''''

Finally, the velocities can be projected and visualized in any embedding (e.g. UMAP) on single cell level, as gridlines, or as streamlines::

    scv.pl.velocity_embedding(adata, basis='umap', **params)
    scv.pl.velocity_embedding_grid(adata, basis='umap', **params)
    scv.pl.velocity_embedding_stream(adata, basis='umap', **params)

For every tool module there is a plotting counterpart, which allows you to examine your results in detail, e.g.::

    scv.pl.velocity(adata, var_names=['gene_A', 'gene_B'], **params)
    scv.pl.velocity_graph(adata, **params)
