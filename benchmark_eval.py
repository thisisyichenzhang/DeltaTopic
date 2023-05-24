#!/usr/bin/env python
import pyliger
from DeltaTopic.nn.util_benchmark import *
import argparse

# Create the parser
parser = argparse.ArgumentParser(description="Benchmark evaluation script")

# Add an argument
parser.add_argument('--model_id', type=str, required=True, 
                    help='The ID of the model to evaluate')

# Parse the arguments
args = parser.parse_args()
model_id = args.model_id
# In[79]:

#model_id = 'BALSAM_ep2000_nlv32_simseed11_seed11_N10000_G5000_T16_topk100_pip0.1_v4'
adata = sc.read(os.path.join('models', model_id, 'adata.h5ad'))
N, G = adata.shape

theta = np.genfromtxt(os.path.join('models', model_id, 'topics.csv'), delimiter=',', skip_header=1)
theta = theta[:,1:]


# In[4]:


adata_unspliced = adata.copy()
adata_unspliced.X = adata.obsm["unspliced_expression"].copy()
adata_concat = ad.concat([adata, adata_unspliced])

T = 32
topic_label = create_topic_label(N, T)
topic_label_1d = np.sum(topic_label, axis=1)


# In[5]:


nmi = compute_nmi(topic_label, theta)
nmi

print(nmi)
# In[6]:


# Compute the PCA
adata.X = adata.layers['counts'].copy()
sc.tl.pca(adata, svd_solver='arpack')
# Compute the neighborhood graph
sc.pp.neighbors(adata, n_pcs=T, use_rep='X_pca')
# Compute the Louvain clusters
sc.tl.louvain(adata)


# In[7]:


nmi_pca = normalized_mutual_info_score(topic_label_1d, adata.obs['louvain'])

print(nmi_pca)
# In[8]:


# Compute the PCA
adata_concat.X = adata_concat.layers['counts'].copy()
sc.tl.pca(adata_concat, svd_solver='arpack')
# Compute the neighborhood graph
sc.pp.neighbors(adata_concat, n_pcs=T, use_rep='X_pca')
# Compute the Louvain clusters
sc.tl.louvain(adata_concat)


# In[9]:


topic_label_1d_concat = np.concatenate((topic_label_1d, topic_label_1d))


# In[10]:


nmi_pca_concat = normalized_mutual_info_score(topic_label_1d_concat, adata_concat.obs['louvain'])
nmi_pca_concat_S = normalized_mutual_info_score(topic_label_1d, adata_concat.obs['louvain'][:N])
nmi_pca_concat_U = normalized_mutual_info_score(topic_label_1d, adata_concat.obs['louvain'][N:])

print(nmi_pca_concat)
print(nmi_pca_concat_S)
print(nmi_pca_concat_U)


# In[11]:


adata.obs.index.name = 'cell_id'
adata.var.index.name = 'gene_id_spliced'
adata.uns['sample_name'] = "spliced"

adata_unspliced.obs.index.name = 'cell_id'
adata_unspliced.var.index.name = 'gene_id_unspliced'
adata_unspliced.uns['sample_name'] = "unspliced"

liger_object = pyliger.create_liger([adata,adata_unspliced])


# In[12]:


pyliger.normalize(liger_object)
pyliger.select_genes(liger_object)
pyliger.scale_not_center(liger_object)


# In[13]:


pyliger.optimize_ALS(liger_object, k = T)


# In[14]:


pyliger.quantile_norm(liger_object)


# In[15]:


pyliger.louvain_cluster(liger_object)


# In[16]:


W_spliced = liger_object.adata_list[0].obs['cluster']
nmi_liger_S = normalized_mutual_info_score(topic_label_1d, W_spliced)
W_unspliced = liger_object.adata_list[1].obs['cluster']
nmi_liger_U = normalized_mutual_info_score(topic_label_1d, W_unspliced)
nmi_liger = normalized_mutual_info_score(topic_label_1d_concat, np.concatenate((W_spliced, W_unspliced)))

print(f"NMI_liger_S: {nmi_liger_S}")
print(f"NMI_liger_U: {nmi_liger_U}")
print(f"NMI_liger: {nmi_liger}")


# In[20]:


liger_object.adata_list[0]


# In[17]:


# Perform NMF on spliced data
from sklearn.decomposition import NMF
model_NMF = NMF(n_components=T, init='random', random_state=0)
W_nmf = model_NMF.fit_transform(adata.X)
H_nmf = model_NMF.components_
adata.obsm['X_nmf'] = W_nmf
adata_NMF = adata.copy()


# In[18]:


# Compute the PCA
sc.tl.pca(adata_NMF.obsm['X_nmf'], svd_solver='arpack')
# Compute the neighborhood graph
sc.pp.neighbors(adata_NMF, n_pcs=T, use_rep='X_nmf')
# Compute the Louvain clusters
sc.tl.louvain(adata_NMF)
nmi_nmf = normalized_mutual_info_score(topic_label_1d, adata_NMF.obs['louvain'])


# In[19]:


print("Normalized mutual information:", nmi)
print("Normalized mutual information (NMF, spliced):", nmi_nmf)
print("Normalized mutual information (PCA, concatenated):", nmi_pca_concat)
print("Normalized mutual information (PCA, concat-spliced):", nmi_pca_concat_S)
print("Normalized mutual information (PCA, concat-unspliced):", nmi_pca_concat_U)
print("Normalized mutual information (PCA, spliced):", nmi_pca)
print("Normalized mutual information (LIGER, concatenated):", nmi_liger)
print("Normalized mutual information (LIGER, concat-spliced):", nmi_liger_S)
print("Normalized mutual information (LIGER, concat-unspliced):", nmi_liger_U)


# In[68]:


def process_liger_object(liger_object, adata):
    # Extract V_spliced and V_unspliced and create DataFrames
    V_spliced = liger_object.adata_list[0].varm['V']
    V_df_spliced = pd.DataFrame(V_spliced, index=liger_object.adata_list[0].var.index)

    V_unspliced = liger_object.adata_list[1].varm['V']
    V_df_unspliced = pd.DataFrame(V_unspliced, index=liger_object.adata_list[1].var.index)

    # Create full_df and join with V_df_spliced and V_df_unspliced
    full_df = pd.DataFrame(index = adata.var.index)
    full_df['gene_order'] = range(adata.var.index.shape[0])

    df_spliced = full_df.join(V_df_spliced, how='outer')
    df_spliced = df_spliced.fillna(0)
    
    df_unspliced = full_df.join(V_df_unspliced, how='outer')
    df_unspliced = df_unspliced.fillna(0)

    # Sort by gene order
    sorted_df_spliced = df_spliced.sort_values(by='gene_order')
    sorted_df_unspliced = df_unspliced.sort_values(by='gene_order')
    
    return sorted_df_spliced, sorted_df_unspliced

df_W_liger_spliced, df_W_liger_unspliced = process_liger_object(liger_object, adata)
weight_liger_S = df_W_liger_spliced.iloc[:, 1:].T
weight_liger_U = df_W_liger_unspliced.iloc[:, 1:].T


# In[60]:


# read in the rho and delta matrices
rho = np.genfromtxt('data/rho_weight.csv', delimiter=',', skip_header=1)
rho = rho[:,1:]
delta = np.genfromtxt('data/delta_weight.csv', delimiter=',', skip_header=1)
delta = delta[:,1:]


# In[78]:


weights = {
    #"rho": weight_rho,
    #"delta": weight_delta,
    #"pca_concat": weight_pca_concat,
    #"pca_spliced": weight_pca,
    "liger_concat_S": weight_liger_S,
    "liger_concat_U": weight_liger_U,
    "nmf_spliced":H_nmf,
}
betas = {
    "unspliced": rho,
    "spliced": delta,
}
df_out = pd.DataFrame()
df_col_out = pd.DataFrame()

for k in [10,20,30,40,50,100,300,500,1000,5000]:
    
    results = calculate_common_entries(weights, betas, k)

    df, df_col = create_dataframe_from_results(results)
    df["k"] = k
    df_col["k"] = k
    
    df_out = pd.concat([df_out, df], ignore_index=True)
    df_col_out = pd.concat([df_col_out, df_col], ignore_index=True)


# In[80]:


# Create the output directory if it doesn't exist
output_dir = os.path.join("models", model_id, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# Save the NMI values to a text file
np.savetxt(os.path.join(output_dir, "nmi_values_new.txt"), [nmi, nmi_pca, nmi_pca_concat, nmi_pca_concat_S, nmi_pca_concat_U, nmi_liger, nmi_liger_S, nmi_liger_U, nmi_nmf])

output_file = os.path.join(output_dir, "common_rowmax_liger.csv")
df_out.to_csv(output_file, index=False)
output_file = os.path.join(output_dir, "common_colmax_liger.csv")
df_col_out.to_csv(output_file, index=False)

