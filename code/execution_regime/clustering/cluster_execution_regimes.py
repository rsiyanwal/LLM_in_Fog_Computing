# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 14:08:32 2026

@author: Rahul
"""

"""
Research-grade clustering of execution regimes.

Features:
- Header normalization (case-insensitive, hyphen-safe)
- Log + robust scaling for heavy-tailed data
- Density-based clustering (HDBSCAN)
- Explicit noise labeling
- Deterministic, reproducible, publication-ready
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import hdbscan

# Data
df = pd.read_csv("C:/Users/Rahul/Robodrive/OneDrive/LLM fog/parsed/tinyllama/tinyllama_derived_features.csv")

# Normalize
df.columns = (
    df.columns.str.strip().str.lower().str.replace("-", "_").str.replace(" ", "_")
)

# Feature selection
FEATURES = [
    "cycles_per_token",
    "instructions_per_token",
    "cache_misses_per_token",
    "branch_misses_per_token",
    "ipc",
    "cache_miss_ratio",
]
missing = [c for c in FEATURES if c not in df.columns]
if missing:
    raise ValueError(f"Missing required feature columns: {missing}")
X = df[FEATURES].copy()

# Sanity checks
if X.isnull().any().any():
    raise ValueError("NaNs detected in clustering features")

# Only log-transform strictly positive columns
LOG_FEATURES = [
    "cycles_per_token",
    "instructions_per_token",
    "cache_misses_per_token",
    "branch_misses_per_token",
]

if (X[LOG_FEATURES] <= 0).any().any():
    raise ValueError("Non-positive values found in log-transformed features")

# Log transform (scale stabilization)
for col in LOG_FEATURES:
    X[col] = np.log10(X[col])
    
# Robust scaling (outlier resistant)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Clustering (HDBSCAN)
clusterer = hdbscan.HDBSCAN(
    min_cluster_size = 4,
    min_samples = 3,
    metric = "euclidean",
    cluster_selection_method = "eom",
)
labels = clusterer.fit_predict(X_scaled)

# Attach clustering results
df["cluster"] = labels
df["cluster_confidence"] = clusterer.probabilities_

# PCA for diagnostics / visualization ONLY
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

df["pca_1"] = X_pca[:, 0]
df["pca_2"] = X_pca[:, 1]

# Report
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = int(np.sum(labels == -1))

print("Clustering complete")
print(f"Clusters found : {n_clusters}")
print(f"Noise points   : {n_noise}")











