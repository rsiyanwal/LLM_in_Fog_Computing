#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Derive scale-free execution regime features from raw perf stat CSV.

Input  : CSV with raw counters + token counts
Output : CSV with derived per-token & efficiency features
"""

import argparse
import pandas as pd
import numpy as np

# ----------------------------
# CLI
# ----------------------------

parser = argparse.ArgumentParser(
    description="Derive per-token execution features for regime clustering"
)
parser.add_argument(
    "--input",
    required=True,
    help="Input CSV with raw perf counters",
)
parser.add_argument(
    "--output",
    required=True,
    help="Output CSV with derived features",
)
args = parser.parse_args()

# ----------------------------
# Load data
# ----------------------------

df = pd.read_csv(args.input)

# Normalize column names (lowercase + underscores)
df.columns = (
    df.columns
      .str.strip()
      .str.lower()
      .str.replace("-", "_")
      .str.replace(" ", "_")
)

# ----------------------------
# Required columns check
# ----------------------------

REQUIRED = [
    "cycles",
    "instructions",
    "cache_misses",
    "cache_references",
    "branch_misses",
    "total_tokens",
]

missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Avoid divide-by-zero
df = df.copy()
df["total_tokens"] = df["total_tokens"].replace(0, np.nan)
df["cycles"] = df["cycles"].replace(0, np.nan)
df["cache_references"] = df["cache_references"].replace(0, np.nan)

# ----------------------------
# Feature derivation
# ----------------------------

df["cycles_per_token"] = df["cycles"] / df["total_tokens"]
df["instructions_per_token"] = df["instructions"] / df["total_tokens"]
df["cache_misses_per_token"] = df["cache_misses"] / df["total_tokens"]
df["branch_misses_per_token"] = df["branch_misses"] / df["total_tokens"]

df["ipc"] = df["instructions"] / df["cycles"]
df["cache_miss_ratio"] = df["cache_misses"] / df["cache_references"]

# ----------------------------
# Optional: clean infinities
# ----------------------------

df.replace([np.inf, -np.inf], np.nan, inplace=True)

# ----------------------------
# Column ordering (nice to have)
# ----------------------------

FEATURE_COLS = [
    "cycles_per_token",
    "instructions_per_token",
    "cache_misses_per_token",
    "branch_misses_per_token",
    "ipc",
    "cache_miss_ratio",
]

META_COLS = [c for c in df.columns if c not in FEATURE_COLS]

df = df[META_COLS + FEATURE_COLS]

# ----------------------------
# Save
# ----------------------------

df.to_csv(args.output, index=False)

print(f"Derived features written to: {args.output}")
print(f"Rows: {len(df)}")
print(f"Feature columns: {FEATURE_COLS}")
