# dimensionality-reduction
Dimensionality-Reduction for Gold clusters

## Create mamba env
```
mamba create -n redct_dim \
  -c rapidsai -c nvidia -c conda-forge \
  python=3.9 \
  rapids=22.12 \
  cudatoolkit=11.5 \
  pandas \
  scikit-learn \
  -y
```
