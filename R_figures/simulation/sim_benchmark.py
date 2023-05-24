import numpy as np
import pandas as pd
import scanpy as sc
import os
import argparse
import anndata as ad
from scipy.special import softmax
from scipy.sparse import csr_matrix
from DeltaTopic.nn.util import setup_anndata
from DeltaTopic.nn.modelhub import BALSAM, DeltaTopic
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
logger = WandbLogger(project="DeltaTopic_benchmark")
import wandb
from DeltaTopic.nn.util_benchmark import *
wandb.init(project="DeltaTopic_benchmark")
#%%
parser = argparse.ArgumentParser(description='Parameters for benchmark study')
parser.add_argument('--model', choices=['DeltaTopic', 'BALSAM'])
parser.add_argument('--EPOCHS', type=int, help='EPOCHS', default=2000) # 1000
parser.add_argument('--nLV', type=int, help='User specified nLV', default=32) 
parser.add_argument('--N', type=int, help='number of cells to select randomly', default=10000)
parser.add_argument('--G', type=int, help='number of genes to select randomly', default=5000)
parser.add_argument('--T', type=int, help='number of rho and delta topics, respectively', default=16)
parser.add_argument('--my_seed', type=int, help='simulation_seed', default=66)
parser.add_argument('--seed', type=int, help='seed', default=66)
parser.add_argument('--use_gpu', type=int, help='which GPU to use', default=0)
parser.add_argument('--topk', type=int, help='number of top and bottom genes to check', default=100)
parser.add_argument('--pip', type=float, help='proportion of top and genes', default=0.1)
args = parser.parse_args()
print(args)

#%%

model_id = f"{args.model}_ep{args.EPOCHS}_nlv{args.nLV}_simseed{args.my_seed}_seed{args.seed}_N{args.N}_G{args.G}_T{args.T}_topk{args.topk}_pip{args.pip}_v4"
print(model_id)

my_seed = args.my_seed
pip = args.pip

# read in the rho and delta matrices
rho = np.genfromtxt('data/rho_weight.csv', delimiter=',', skip_header=1)
rho = rho[:,1:]
delta = np.genfromtxt('data/delta_weight.csv', delimiter=',', skip_header=1)
delta = delta[:,1:]
n_topic, n_gene = rho.shape  # get the total number of columns in rho

#%%
DeltaTopic_theta = np.genfromtxt('data/models/DeltaTopic/topics.csv', delimiter=',', skip_header=1)
DeltaTopic_theta = DeltaTopic_theta[:,1:]

#%%
beta_spliced = rho + delta
beta_unspliced = rho
T,G = beta_spliced.shape
#%%
N_per_topic = 1000 # number of cells per topic
N = T * N_per_topic
prob_max = 0.9 # probability of the most likely topic
alpha = np.ones([T, T]) * (1 - prob_max) / (T - 1) # N x 2
np.fill_diagonal(alpha, prob_max)
alpha = np.repeat(alpha, N_per_topic, axis=0)
alpha.shape # (N, T)

# sample theta from Dirichlet distribution
# Set the random seed

np.random.seed(my_seed)
theta = []
for i in range(alpha.shape[0]):
    sample = np.random.dirichlet(alpha[i])
    theta.append(sample)
    
theta = np.array(theta)



# sample X from multinomial distribution
## assuming all cells have the same sequencing depth
np.random.seed(my_seed)
seq_depth = G
X_spliced = []; X_unspliced = []
loading_spliced = np.dot(theta, beta_spliced)
loading_unspliced = np.dot(theta, beta_unspliced)

for i in range(N):
    sample = np.random.multinomial(n= seq_depth, pvals=softmax(loading_spliced[i]), size=1)
    X_spliced.append(sample)
    
X_spliced = np.squeeze(np.array(X_spliced))

for i in range(N):
    sample = np.random.multinomial(n= seq_depth, pvals=softmax(loading_unspliced[i]), size=1)
    X_unspliced.append(sample)
        
X_unspliced = np.squeeze(np.array(X_unspliced))

#%%
adata = ad.AnnData(csr_matrix(X_spliced))
adata.layers["counts"] = adata.X.copy()
adata.obsm["unspliced_expression"] = csr_matrix(X_unspliced)

adata.obs_names = [f"Cell_{i:d}" for i in range(adata.n_obs)]
adata.var_names = [f"Gene_{i:d}" for i in range(adata.n_vars)]
adata.obs["topic_true"] = np.repeat([f"topic_{i:d}" for i in range(T)], adata.n_obs/T)

setup_anndata(adata, layer="counts", unspliced_obsm_key = "unspliced_expression")
model_kwargs = {'use_gpu':args.use_gpu}

if args.model == "DeltaTopic":
    model = DeltaTopic(adata, n_latent = args.nLV)
elif args.model == "BALSAM":
    model = BALSAM(adata, n_latent = args.nLV)

#%%        
seed_everything(args.seed, workers=True)
#set deterministic=True for reproducibility
model.train(
    args.EPOCHS, 
    logger = logger, 
    deterministic=True, 
    **model_kwargs,
    )

model.save(os.path.join("models", model_id), overwrite=False,save_anndata=True)
print(f"Model saved at:", os.path.join("models", model_id))
#%% save output
# spike, slab, standard deviation
print("---Saving global parameters: spike, slab, standard deviation---\n")
model.get_parameters(save_dir = os.path.join("models", model_id), overwrite = False)
topics_np = model.get_latent_representation(deterministic=True, output_softmax_z=True)
# topic proportions (after softmax)
print("---Saving topic proportions (after softmax)---\n")
topics_df = pd.DataFrame(topics_np, index= model.adata.obs.index, columns = ['topic_' + str(j) for j in range(topics_np.shape[1])])
topics_df.to_csv(os.path.join("models", model_id,"topics.csv"))
# Save the array as a text file
np.savetxt(os.path.join("models", model_id,"theta.txt"), theta)

#%%
# Create the output directory if it doesn't exist
output_dir = os.path.join("models", model_id, "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)   
#%%    
spike_logit_rho = np.loadtxt(os.path.join("models", model_id, "model_parameters", "spike_logit_rho.txt"))
slab_mean_rho = np.loadtxt(os.path.join("models", model_id, "model_parameters", "spike_logit_rho.txt"))
pip_rho = 1/np.exp(-spike_logit_rho)    
weight_rho = slab_mean_rho * pip_rho

if args.model == "DeltaTopic":
    spike_logit_delta = np.loadtxt(os.path.join("models", model_id, "model_parameters", "spike_logit_delta.txt"))
    slab_mean_delta = np.loadtxt(os.path.join("models", model_id, "model_parameters", "spike_logit_delta.txt"))
    pip_delta = 1/np.exp(-spike_logit_delta)
    weight_delta = slab_mean_delta * pip_delta
elif args.model == "BALSAM":
    weight_delta = np.array([])
    
#%% 
T = topics_np.shape[1]  # number of topics in the theta matrix
    
# generate topic labels
topic_label = create_topic_label(N, T)
# compute pca
weight_pca, pc_matrix, weight_pca_concat, pc_matrix_concat = compute_pca(adata, T)

# compute nmi
nmi = compute_nmi(topic_label, theta)
nmi_pca = compute_nmi(topic_label, pc_matrix)
nmi_pca_concat = compute_nmi(topic_label, pc_matrix_concat[:N,:])

print("Normalized mutual information:", nmi)
print("Normalized mutual information (PCA, concatenated):", nmi_pca_concat)
print("Normalized mutual information (PCA):", nmi_pca)

# Save the NMI values to a text file
np.savetxt(os.path.join(output_dir, "nmi_values.txt"), [nmi, nmi_pca, nmi_pca_concat])
#%%
weights = {
    "rho": weight_rho,
    "delta": weight_delta,
    "pca_concat": weight_pca_concat,
    "pca_spliced": weight_pca,
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

 
output_file = os.path.join(output_dir, "common_rowmax.csv")
df_out.to_csv(output_file, index=False)
output_file = os.path.join(output_dir, "common_colmax.csv")
df_col_out.to_csv(output_file, index=False)
    
np.savetxt(os.path.join(output_dir, "delta_subset.txt"), delta)
np.savetxt(os.path.join(output_dir, "rho_subset.txt"), rho)