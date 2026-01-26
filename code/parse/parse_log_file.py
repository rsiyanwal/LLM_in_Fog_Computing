#!/usr/bin/env python3
import re
import csv
import sys
import json
import os

LOG_FILE = sys.argv[1]
OUT_CSV  = sys.argv[2]

# ---------------- regex patterns ----------------
re_ctx_train = re.compile(r"print_info:\s+n_ctx_train\s+=\s+(\d+)")
re_kv = re.compile(r"CPU KV buffer size\s+=\s+([\d.]+)\s+MiB")
re_load = re.compile(r"load time =\s+([\d.]+)\s+ms")

re_prompt = re.compile(
    r"prompt eval time =\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+tokens\s+\(\s+([\d.]+)\s+ms per token,\s+([\d.]+)\s+tokens per second\)"
)

re_eval = re.compile(
    r"eval time =\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+runs\s+\(\s+([\d.]+)\s+ms per token,\s+([^\s]+)\s+tokens per second\)"
)

re_total = re.compile(r"total time =\s+([\d.]+)\s+ms")
re_graphs = re.compile(r"graphs reused =\s+(\d+)")
re_json = re.compile(r"^\{.*\"model_path\".*\}$")

# ---------------- defaults ----------------
def fresh_run():
    return {
        "n_ctx_train": 0,
        "kv_buffer_mib": 0.0,
        "load_time_ms": 0.0,
        "prompt_ms": 0.0,
        "prompt_tokens": 0,
        "prompt_ms_per_token": 0.0,
        "prompt_tokens_per_sec": 0.0,
        "eval_ms": 0.0,
        "eval_runs": 0,
        "eval_ms_per_token": 0.0,
        "eval_tokens_per_sec": 0.0,
        "total_time_ms": 0.0,
        "wall_time_sec": 0.0,
        "graphs_reused": 0,
    }

runs = []
current = fresh_run()

with open(LOG_FILE, "r", errors="ignore") as f:
    for line in f:
        line = line.strip()

        m = re_ctx_train.search(line)
        if m:
            current["n_ctx_train"] = int(m.group(1))

        m = re_kv.search(line)
        if m:
            current["kv_buffer_mib"] = float(m.group(1))

        m = re_load.search(line)
        if m:
            current["load_time_ms"] = float(m.group(1))

        m = re_prompt.search(line)
        if m:
            current.update({
                "prompt_ms": float(m.group(1)),
                "prompt_tokens": int(m.group(2)),
                "prompt_ms_per_token": float(m.group(3)),
                "prompt_tokens_per_sec": float(m.group(4)),
            })

        m = re_eval.search(line)
        if m:
            tps = m.group(4)
            current.update({
                "eval_ms": float(m.group(1)),
                "eval_runs": int(m.group(2)),
                "eval_ms_per_token": float(m.group(3)),
                "eval_tokens_per_sec": float("inf") if tps == "inf" else float(tps),
            })

        m = re_total.search(line)
        if m:
            current["total_time_ms"] = float(m.group(1))

        m = re_graphs.search(line)
        if m:
            current["graphs_reused"] = int(m.group(1))

        # -------- JSON = end of run --------
        if re_json.match(line):
            meta = json.loads(line)

            current["model_name"] = os.path.basename(meta["model_path"])
            current["task"] = os.path.splitext(os.path.basename(meta["prompt_path"]))[0]
            current["wall_time_sec"] = float(meta["wall_time_sec"])

            runs.append(current)
            current = fresh_run()

# ---------------- CSV output ----------------
fields = [
    "model_name",
    "task",
    "n_ctx_train",
    "kv_buffer_mib",
    "load_time_ms",
    "prompt_ms",
    "prompt_tokens",
    "prompt_ms_per_token",
    "prompt_tokens_per_sec",
    "eval_ms",
    "eval_runs",
    "eval_ms_per_token",
    "eval_tokens_per_sec",
    "total_time_ms",
    "wall_time_sec",
    "graphs_reused",
]

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in runs:
        writer.writerow(r)

print(f"Parsed {len(runs)} runs ? {OUT_CSV}")
