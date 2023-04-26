Getting Started
---------------

Here, you will be briefly guided through the basics of how to use **BALSAM** and **DeltaTopic**. 

   
Data Preparation
''''''''''''''''
Read your data file, for example, a h5ad file, using `scanpy`::
    
    import scanpy as sc
    adata = sc.read(filename_spliced)
    adata_unspliced = sc.read(filename_unspliced)

Register spliced and unspliced counts::    
    
    adata.layers["counts"] = adata.X.copy()
    adata.obsm["unspliced_expression"] = adata_unspliced.X.copy()

Setup anndata::    
    
    from DeltaTopic.nn.util import setup_anndata
    setup_anndata(adata, layer="counts", unspliced_obsm_key = "unspliced_expression")

Note: if you are training BALSAM only, you can skip the additional step to register unspliced counts.

Training
''''''''

Import the model and train::

    from DeltaTopic.nn.modelhub import BALSAM, DeltaTopic
    model = DeltaTopic(adata, n_latent = 32)
    model.train(epochs = 400)

Save model states and output the latent space::

    import pandas as pd
    model.save(SavePATH)
    model.get_parameters(save_dir = SavePath) # spike and slab parameters
    topics_np = model.get_latent_representation() # latent topic proportions
    pd.DataFrame(topics_np).to_csv(SaveFILENAME)
    
Analysis
''''''''

Finally, perform your favorite analyis on the latent space and topic loading. For an example of analyis used in the paper, please refer to the Rmd files in the `project repository <https://github.com/causalpathlab/DeltaTopic/tree/main/R_figures/>`_.