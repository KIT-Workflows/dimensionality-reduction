# dimensionality-reduction
Dimensionality-Reduction for Gold clusters

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hf-0tiF0aGdc8LFTXC5KiTX1jjYj3Ind)


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
