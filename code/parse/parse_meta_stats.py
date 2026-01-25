#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import csv
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = Path("/home/pi/edge-llm-bench/runs/meta/")          
OUTPUT_CSV = "tinyllama_parsed_meta_stats.csv"

# -----------------------------
# MAIN PARSER
# -----------------------------
rows = []

for json_file in sorted(INPUT_DIR.glob("stat_*.json")):
    name = json_file.stem  # stat_tinyllama_T01

    parts = name.split("_", 2)
    if len(parts) != 3:
        continue

    _, model, task = parts

    try:
        data = json.loads(json_file.read_text())
    except Exception as e:
        print(f"[!] Failed to parse {json_file}: {e}")
        continue

    row = {
        "Model name": model,
        "Task": task,
        "wall_time_sec": data.get("wall_time_sec", ""),
        "input_tokens": data.get("input_tokens", ""),
        "output_tokens": data.get("output_tokens", ""),
        "model_path": data.get("model_path", ""),
        "prompt_path": data.get("prompt_path", ""),
        "prompt_source": data.get("prompt_source", ""),
        "token_control": data.get("token_control", ""),
    }

    rows.append(row)

# -----------------------------
# WRITE CSV
# -----------------------------
headers = [
    "Model name",
    "Task",
    "wall_time_sec",
    "input_tokens",
    "output_tokens",
    "model_path",
    "prompt_path",
    "prompt_source",
    "token_control",
]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

print(f"Parsed {len(rows)} JSON files -> {OUTPUT_CSV}")
