#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import csv
from pathlib import Path

# -----------------------------
# CONFIG
# -----------------------------
INPUT_DIR = Path("/home/pi/edge-llm-bench/runs/stat/")
OUTPUT_CSV = "tinyllama_parsed_perf_stats.csv"

# -----------------------------
# REGEX PATTERNS
# -----------------------------
PATTERNS = {
    "cycles": re.compile(r"([\d,]+)\s+cycles\b"),
    "instructions": re.compile(r"([\d,]+)\s+instructions\b"),
    "branch-misses": re.compile(r"([\d,]+)\s+branch-misses\b"),
    "cache-references": re.compile(r"([\d,]+)\s+cache-references\b"),
    "cache-misses": re.compile(r"([\d,]+)\s+cache-misses\b"),
    "time_elapsed": re.compile(r"([\d.]+)\s+seconds time elapsed"),
    "time_user": re.compile(r"([\d.]+)\s+seconds user"),
    "time_sys": re.compile(r"([\d.]+)\s+seconds sys"),
}

# -----------------------------
# HELPERS
# -----------------------------
def clean_int(value):
  """ Removes commas and converts to int."""
  return int(value.replace(",", ""))
  
def extract(pattern, text, is_int = True):
  match = pattern.search(text)
  if not match:
    return ""
  return clean_int(match.group(1)) if is_int else float(match.group(1))
  
# -----------------------------
# MAIN PARSER
# -----------------------------
rows = []

for txt_file in sorted(INPUT_DIR.glob("*.txt")):
    name = txt_file.stem  # e.g. tinyllama_T02
    if "_" not in name:
        continue

    model, task = name.split("_", 1)

    content = txt_file.read_text(errors="ignore")

    row = {
        "Model name": model,
        "Task": task,
        "Cycles": extract(PATTERNS["cycles"], content),
        "Instructions": extract(PATTERNS["instructions"], content),
        "branch-misses": extract(PATTERNS["branch-misses"], content),
        "cache-references": extract(PATTERNS["cache-references"], content),
        "cache-misses": extract(PATTERNS["cache-misses"], content),
        "seconds time elapsed": extract(PATTERNS["time_elapsed"], content, is_int=False),
        "seconds user": extract(PATTERNS["time_user"], content, is_int=False),
        "seconds sys": extract(PATTERNS["time_sys"], content, is_int=False),
    }

    rows.append(row)

# -----------------------------
# WRITE CSV
# -----------------------------
headers = [
    "Model name",
    "Task",
    "Cycles",
    "Instructions",
    "branch-misses",
    "cache-references",
    "cache-misses",
    "seconds time elapsed",
    "seconds user",
    "seconds sys",
]

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)

print(f"Parsed {len(rows)} files -> {OUTPUT_CSV}")