import time
import pandas as pd
import cupy as cp

# RAPIDS libraries for preprocessing and TSNE
from cuml.preprocessing import StandardScaler
from cuml.decomposition import PCA
from cuml.manifold import TSNE

# ----------------------
# 1. PARAMETERS
# ----------------------

input_file = 'full_complete.CN'  # <-- Update if needed
output_file = 'tsne_output.dat'
perplexity = 50           # TSNE perplexity (very large for typical use)
learning_rate = 50.0      # TSNE learning rate
early_exaggeration = 10.0 # TSNE early exaggeration
angle = 0.8               # Barnes-Hut angle
n_iter = 2000             # TSNE iterations
use_pca = True            # Whether to apply PCA before TSNE
pca_components = 10       # Number of PCA components (adjust as needed)

# ----------------------
# 2. LOAD DATA
# ----------------------
# Read CSV; limit columns if desired
dados = pd.read_csv(input_file, delim_whitespace=True, header=0, usecols=range(13))

print("Data Shape:", dados.shape)
print("First row:", dados.iloc[0].to_dict())

# ----------------------
# 3. MOVE DATA TO GPU
# ----------------------
X_gpu = cp.array(dados.values, dtype=cp.float32)

# ----------------------
# 4. PREPROCESSING
# ----------------------

# (A) Standard Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_gpu)

# (B) Optional PCA
#     Reduces dimensionality before TSNE.
#     This can speed up TSNE and sometimes improve cluster separation.
if use_pca:
    # If the dataset has fewer columns than pca_components, automatically adjust
    max_components = min(X_scaled.shape[1], pca_components)
    pca = PCA(n_components=max_components, random_state=42)
    X_preprocessed = pca.fit_transform(X_scaled)
    print(f"Applying PCA: reduced from {X_scaled.shape[1]}D to {max_components}D")
else:
    X_preprocessed = X_scaled

# ----------------------
# 5. CONFIGURE TSNE
# ----------------------
tsne = TSNE(
    n_components=2,
    perplexity=perplexity,
    learning_rate=learning_rate,
    early_exaggeration=early_exaggeration,
    angle=angle,
    n_iter=n_iter,
    init='random',
    metric='cosine', # 'euclidean' is typical, but you could use 'cosine', 'manhattan'
    method='barnes_hut',
    random_state=42,
    verbose=2
)

# ----------------------
# 6. FIT & TRANSFORM
# ----------------------
start_time = time.time()
X_tsne_gpu = tsne.fit_transform(X_preprocessed)
end_time = time.time()

# ----------------------
# 7. PRINT METRICS
# ----------------------
print(f"\n--- t-SNE Convergence Metrics ---")
print(f"Total time (seconds): {end_time - start_time:.2f}")
print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")

# ----------------------
# 8. SAVE RESULTS
# ----------------------
X_tsne_cpu = X_tsne_gpu.get()  # cupy -> numpy
df_tsne = pd.DataFrame(X_tsne_cpu, columns=['#Componente_1', 'Componente_2'])
df_tsne.to_csv(output_file, sep='\t', index=False, float_format='%.10f', header=False)

print(f"t-SNE embedding saved to: {output_file}")
