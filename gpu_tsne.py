import time
import pandas as pd
import cupy as cp
from cuml.manifold import TSNE

perp = 2000  # Adjust this value to set the number of neighbors
output_file = f"tsne_perp{perp}.dat"

# 1. Load Data
dados = pd.read_csv(
    'full_complete.CN',
    delim_whitespace=True,
    header=0,
    usecols=range(13)
)

print("Data Shape:", dados.shape)
print("First row:", dados.iloc[0].to_dict())

# 2. Convert to GPU (cupy) array
X = cp.array(dados.values, dtype=cp.float32)

# 3. Configure t-SNE
tsne = TSNE(
    n_components=2,
    perplexity=perp,
    learning_rate=10.0,
    early_exaggeration=10.0,
    angle=0.8,
    n_iter=1000,
    init='random',
    metric='euclidean',
    method='barnes_hut',
    random_state=42,
    verbose=2  # More detailed logging
)

# 4. Fit & Transform with timing
start_time = time.time()
X_tsne_gpu = tsne.fit_transform(X)
end_time = time.time()

# 5. Print metrics for convergence analysis
print(f"\n--- t-SNE Convergence Metrics ---")
print(f"Total time (seconds): {end_time - start_time:.2f}")

# cuML TSNE provides the final Kullback-Leibler divergence as an attribute:
# (In scikit-learn, you'd use tsne.kl_divergence_ as well.)
print(f"Final KL divergence: {tsne.kl_divergence_:.4f}")

# 6. Convert result back to CPU & Save
X_tsne_cpu = X_tsne_gpu.get()  # cupy array -> numpy array
df_tsne = pd.DataFrame(X_tsne_cpu, columns=['#Componente_1', 'Componente_2'])
df_tsne.to_csv(
    output_file,
    sep='\t',
    index=False,
    float_format='%.10f',
    header=False
)

print(f"t-SNE embedding saved to: {output_file}")

