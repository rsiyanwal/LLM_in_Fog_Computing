
#!/usr/bin/env python3
import os, sys, time, json, subprocess, re, argparse, math, shutil
from pathlib import Path
import yaml
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

# ---------------------- metrics ---------------------- #
from rouge_score import rouge_scorer
import sacrebleu

# ---------------------- per-task generation caps (speed) ---------------------- #
N_PREDICT_BY_TASK = {
    "summarization": 64,   # one sentence
    "qa":            24,   # short spans
    "hellaswag":     12,   # just a letter
    "gsm8k":        160,   # a bit of room for steps + "####"
    "translation":   64,   # sentence pairs
}

# ---------------------- prompt helpers ---------------------- #
def prompt_summarize(doc):
    return f"Summarize the following article in one sentence.\n\nArticle:\n{doc}\n\nSummary:"

def prompt_squad(context, question):
    return ("Answer the question using the context. "
            "If unanswerable, output exactly: unanswerable.\n\n"
            f"Context: {context}\n\nQuestion: {question}\n\nAnswer:")

def prompt_hellaswag(row):
    ctx_a = row["ctx_a"]; ctx_b = row["ctx_b"]
    choices = row["endings"]
    letters = ["A","B","C","D"]
    opts = "\n".join(f"{letters[i]}. {choices[i]}" for i in range(4))
    return (f"{ctx_a}{ctx_b}\n\nWhich ending is most plausible?\n{opts}\n\n"
            "Return only the letter of the best option (A/B/C/D). Answer:")

def prompt_gsm8k(q):
    return (f"{q}\n\nSolve step by step. Give final numeric answer as:\n"
            "#### <number>\nAnswer:")

def prompt_translate(src_text, src_lang, tgt_lang):
    return (f"Translate from {src_lang} to {tgt_lang}.\n\n"
            f"Source: {src_text}\n\nTranslation:")

# ---------------------- scoring helpers ---------------------- #
def em(pred, golds):
    pred = pred.strip()
    golds = [g.strip() for g in golds]
    return 1.0 if pred in golds else 0.0

def f1_token(pred, golds):
    def normalize(s): return re.findall(r'\w+', s.lower())
    pred_toks = normalize(pred)
    best = 0.0
    for g in golds:
        gold_toks = normalize(g)
        common = set(pred_toks) & set(gold_toks)
        if not pred_toks and not gold_toks: return 1.0
        if not pred_toks or not gold_toks:  continue
        p = len(common)/len(pred_toks)
        r = len(common)/len(gold_toks)
        f1 = 2*p*r/(p+r) if (p+r)>0 else 0.0
        best = max(best, f1)
    return best

def rouge_l(pred, ref):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    return scorer.score(ref, pred)['rougeL'].fmeasure

def bleu_score(preds, refs):
    return sacrebleu.corpus_bleu(preds, [refs]).score if preds and refs else 0.0

def gsm8k_extract(ans):
    m = re.search(r'####\s*([-+]?\d+(?:\.\d+)?)', ans)
    return m.group(1) if m else None

def hellaswag_letter(pred):
    m = re.search(r'\b([ABCD])\b', pred.strip(), re.I)
    return m.group(1).upper() if m else None

# ---------------------- helpers ---------------------- #
def detect_gnu_time():
    """
    Return absolute path to GNU time supporting -v, or None.
    """
    cand = shutil.which("time")
    if not cand:
        return None
    # cheap feature test: run `time -v true`
    try:
        r = subprocess.run([cand, "-v", "true"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=2)
        if r.returncode == 0 and re.search(r'Maximum resident set size', r.stdout, re.I):
            return cand
    except Exception:
        pass
    return None

# ---------------------- llama.cpp runner ---------------------- #
def run_llama(main_bin, model_path, prompt, n_predict, ctx, threads, batch, decoding, kvbits, gnu_time_bin=None):
    topk, topp, temp = 0, 1.0, 0.0
    if decoding != "greedy":
        m = re.search(r'topk(\d+)_p(\d+)_t(\d+)', decoding)
        if m:
            topk = int(m.group(1)); topp = float(f"0.{m.group(2)}"); temp = float(f"0.{m.group(3)}")

    base_cmd = [
        main_bin, "-m", model_path, "-n", str(n_predict),
        "--ctx-size", str(ctx), "-t", str(threads), "-b", str(batch),
        "--top-k", str(topk), "--top-p", str(topp), "--temp", str(temp),
        "-p", prompt, "--ignore-eos", "-no-cnv", "--no-warmup"
    ]

    # no flash_attn in your build -> keep KV in f16 if requested
    if str(kvbits).lower() in {"16", "f16"}:
        base_cmd += ["-ctk", "f16", "-ctv", "f16"]

    # Wrap with GNU time to capture peak RSS
    cmd = ([gnu_time_bin, "-v"] + base_cmd) if gnu_time_bin else base_cmd

    t0 = time.time()
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wall_ms = (time.time() - t0) * 1000.0
    out = p.stdout

    if p.returncode != 0 or re.search(r"\berror:|invalid argument:|usage:", out, re.I):
        raise RuntimeError(f"llama.cpp invocation failed:\n{' '.join(base_cmd)}\n---\n{out}")

    # -------------------- parse timing metrics -------------------- #
    # 1) TTFT: prefer explicit "time to first token", else use "prompt eval time"
    ttft = None
    m_ttft = re.search(r'(?:time to first token|first token).*?=\s*([0-9.]+)\s*ms', out, re.I)
    if m_ttft:
        ttft = float(m_ttft.group(1))
    else:
        m_prompt = re.search(r'prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*\d+\s*tokens', out, re.I)
        if m_prompt:
            ttft = float(m_prompt.group(1))

    # 2) Generation time: use the *non-prompt* eval line with "/ N runs"
    #    Use negative lookbehind to avoid matching "prompt eval time"
    gen_time_ms = None
    m_eval = re.findall(r'(?<!prompt )eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*runs', out, re.I)
    if m_eval:
        # take the last eval block (final gen)
        gen_time_ms = float(m_eval[-1][0])

    # 3) Tokens/sec: prefer the eval block's "(ms per token, tokens per second)"
    tokps = None
    m_eval_tps = re.findall(
        r'(?<!prompt )eval time\s*=\s*[0-9.]+\s*ms\s*/\s*\d+\s*runs\s*\(\s*([0-9.]+)\s*ms per token,\s*([0-9.]+)\s*tokens per second\)',
        out, re.I
    )
    if m_eval_tps:
        tokps = float(m_eval_tps[-1][1])
    else:
        # fallback: any "... ms per token" (last occurrence)
        m_msp = re.findall(r'([0-9.]+)\s*ms per token', out, re.I)
        if m_msp:
            try:
                tokps = 1000.0 / float(m_msp[-1])
            except ZeroDivisionError:
                tokps = None

    # -------------------- parse memory metrics -------------------- #
    rss_mb = peak_mb = None

    # prefer llama's MEM line if present
    m_mem = re.search(r'MEM\s+(\d+)\s+(\d+)\s+MB', out)
    if m_mem:
        rss_mb = int(m_mem.group(1))
        peak_mb = int(m_mem.group(2))
    else:
        # try GNU time -v output (kbytes)
        if gnu_time_bin:
            m_maxrss = re.search(r'Maximum resident set size.*?:\s*([0-9]+)', out, re.I)
            m_avgrss = re.search(r'Average resident set size.*?:\s*([0-9]+)', out, re.I)
            if m_maxrss:
                try:
                    peak_mb = int(m_maxrss.group(1)) // 1024
                except Exception:
                    peak_mb = None
            if m_avgrss:
                try:
                    rss_mb = int(m_avgrss.group(1)) // 1024
                except Exception:
                    rss_mb = None
            # some Linux builds report 0 for average; fall back to peak
            if rss_mb in (None, 0) and peak_mb is not None:
                rss_mb = peak_mb

    # -------------------- extract generated text (trim logs) -------------------- #
    tail = out[-8000:]
    tags = [t for t in ("Answer:", "Summary:", "Translation:") if t in prompt]
    if tags:
        last_pos = -1; last_tag = None
        for t in tags:
            pos = tail.rfind(t)
            if pos > last_pos:
                last_pos, last_tag = pos, t
        text_after = tail[last_pos + len(last_tag):] if last_pos != -1 else tail
    else:
        text_after = tail
    text_after = re.split(
        r'\n(?:llama_perf_|system_info:|sampler|main:|generate:|decode:|Command being timed:|User time \(seconds\):|Maximum resident set size)',
        text_after, maxsplit=1, flags=re.I
    )[0]
    pred = text_after.strip()

    return pred, {
        "tok_per_s": tokps,
        "ttft_ms": ttft,
        "gen_time_ms": gen_time_ms,
        "wall_ms": (time.time() - t0) * 1000.0,  # recompute plainly
        "rss_mb": rss_mb,
        "peak_mb": peak_mb,
        "raw": tail
    }

# ---------------------- utils ---------------------- #
def norm01(series, reverse=False):
    s = pd.Series(series, dtype=float)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return [1.0] * len(s)
    x = (s - mn) / (mx - mn)
    return list(1.0 - x if reverse else x)

# ---------------------- main ---------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="benchmarks.yaml")
    ap.add_argument("--models", default="models.yaml")
    ap.add_argument("--model_key", nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", default=["summarization_xsum","qa_squad2","commonsense_hellaswag","math_gsm8k","translation_flores200"])
    ap.add_argument("--build_dir", default=str(Path.home() / "edge-llm-bench/bin"))
    ap.add_argument("--outdir", default=str(Path.home() / "edge-llm-bench/results"))
    ap.add_argument("--n_predict", type=int, default=128)
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    bench = yaml.safe_load(open(args.bench))
    models = yaml.safe_load(open(args.models))

    gnu_time_bin = detect_gnu_time()

    rows = []

    for task_name in args.tasks:
        cfg = bench[task_name]

        # load dataset subset
        if cfg["task"] == "summarization":
            ds = load_dataset(cfg["dataset"], split=cfg["split"]).shuffle(seed=42).select(range(cfg["n_samples"]))
        elif cfg["task"] == "qa":
            ds = load_dataset(cfg["dataset"], split=cfg["split"]).shuffle(seed=42).select(range(cfg["n_samples"]))
        elif cfg["task"] == "hellaswag":
            ds = load_dataset(cfg["dataset"], split=cfg["split"]).shuffle(seed=42).select(range(cfg["n_samples"]))
        elif cfg["task"] == "gsm8k":
            ds = load_dataset(cfg["dataset"], cfg.get("subset", "main"), split=cfg["split"]).shuffle(seed=42).select(range(cfg["n_samples"]))
        elif cfg["task"] == "translation":
            ds = load_dataset(cfg["dataset"], cfg.get("subset", "all"), split=cfg["split"]).shuffle(seed=42).select(range(cfg["n_samples"]))
        else:
            raise ValueError("Unknown task")

        for model_key in args.model_key:
            m = models[model_key]
            main_bin = os.path.join(args.build_dir, f"main_{m.get('build', 'blas')}")

            preds, refs = [], []
            correct = 0; total = 0
            ems, f1s = [], []
            rougeLs = []
            per_item = []

            for row in tqdm(ds, desc=f"{task_name} · {model_key}"):
                # 1) Prompt
                if cfg["task"] == "summarization":
                    prompt = prompt_summarize(row[cfg["input_field"]]); gold = row[cfg["target_field"]]
                elif cfg["task"] == "qa":
                    prompt = prompt_squad(row[cfg["context_field"]], row[cfg["question_field"]]); gold_answers = row[cfg["answers_field"]]["text"]
                elif cfg["task"] == "hellaswag":
                    prompt = prompt_hellaswag(row); gold_letter = ["A","B","C","D"][int(row["label"])]
                elif cfg["task"] == "gsm8k":
                    prompt = prompt_gsm8k(row[cfg["question_field"]]); gold = re.findall(r'[-+]?\d+(?:\.\d+)?', row[cfg["answer_field"]].split("####")[-1])[0]
                elif cfg["task"] == "translation":
                    src = row[cfg["src_lang"]]; tgt = row[cfg["tgt_lang"]]; prompt = prompt_translate(src, cfg["src_lang"], cfg["tgt_lang"])
                else:
                    raise ValueError

                # 2) Length
                n_pred = N_PREDICT_BY_TASK.get(cfg["task"], args.n_predict)

                # 3) Run llama.cpp
                try:
                    pred, stats = run_llama(
                        main_bin=main_bin,
                        model_path=m["path"],
                        prompt=prompt,
                        n_predict=n_pred,
                        ctx=m["ctx"], threads=m["threads"], batch=m["batch"],
                        decoding=m["decoding"], kvbits=m.get("kvbits", "16"),
                        gnu_time_bin=gnu_time_bin,
                    )
                except Exception as e:
                    fail_log = Path(args.outdir) / "failed_invocations.log"
                    with open(fail_log, "a") as f:
                        f.write(f"\n[{task_name} · {model_key}] {e}\n")
                    stats = {"tok_per_s": None, "ttft_ms": None, "gen_time_ms": None, "wall_ms": None,
                             "rss_mb": None, "peak_mb": None, "raw": ""}
                    pred = ""

                # 4) Score
                if cfg["task"] == "summarization":
                    rougeLs.append(rouge_l(pred, gold))
                elif cfg["task"] == "qa":
                    ems.append(em(pred, gold_answers)); f1s.append(f1_token(pred, gold_answers))
                elif cfg["task"] == "hellaswag":
                    letter = hellaswag_letter(pred) or ""; correct += 1 if letter == gold_letter else 0; total += 1
                elif cfg["task"] == "gsm8k":
                    num = gsm8k_extract(pred); correct += 1 if (num is not None and str(num) == str(gold)) else 0; total += 1
                elif cfg["task"] == "translation":
                    preds.append(pred); refs.append(tgt)

                # 5) Log per-item metrics
                per_item.append({
                    "task": task_name, "model": model_key,
                    "tok_per_s": stats["tok_per_s"], "ttft_ms": stats["ttft_ms"],
                    "gen_time_ms": stats["gen_time_ms"], "wall_ms": stats["wall_ms"],
                    "rss_mb": stats["rss_mb"], "peak_mb": stats["peak_mb"]
                })

            # Aggregate
            if cfg["task"] == "summarization":
                qual = (sum(rougeLs) / len(rougeLs)) if rougeLs else 0.0; qual_name = "rougeL"
            elif cfg["task"] == "qa":
                qa_em = (sum(ems) / len(ems)) if ems else 0.0
                qa_f1 = (sum(f1s) / len(f1s)) if f1s else 0.0
                qual = 0.5 * qa_em + 0.5 * qa_f1; qual_name = "qa_avg(EM,F1)"
            elif cfg["task"] in {"hellaswag", "gsm8k"}:
                qual = (correct / total) if total else 0.0; qual_name = "accuracy"
            elif cfg["task"] == "translation":
                qual = bleu_score(preds, refs); qual_name = "bleu"
            else:
                qual = 0.0; qual_name = "unknown"

            expected_cols = ["task","model","tok_per_s","ttft_ms","gen_time_ms","wall_ms","rss_mb","peak_mb"]
            df_items = pd.DataFrame(per_item, columns=expected_cols)
            if df_items.empty:
                lat_ms = float("nan"); tokps = float("nan"); mempk = None
            else:
                lat_ms = pd.to_numeric(df_items["wall_ms"], errors="coerce").mean()
                tokps  = pd.to_numeric(df_items["tok_per_s"], errors="coerce").mean()
                mem_series = pd.to_numeric(df_items["peak_mb"], errors="coerce").dropna()
                mempk = mem_series.max() if not mem_series.empty else None

            rows.append({
                "task": task_name, "model": model_key,
                "quality_metric": qual_name, "quality": qual,
                "latency_ms_mean": lat_ms, "tok_per_s_mean": tokps,
                "peak_mem_mb": mempk
            })

            # Dump per-item CSV
            out_items = Path(args.outdir) / f"{model_key}_{task_name}_items.csv"
            df_items.to_csv(out_items, index=False)

    # ERS
    df = pd.DataFrame(rows)
    out_path = Path(args.outdir) / f"{model_key}_{task_name}_summary.csv"
    if df.empty:
        df.to_csv(out_path, index=False); print(f"Wrote {out_path} (no rows)"); return

    out_rows = []
    for task, g in df.groupby("task"):
        qn = norm01(g["quality"]); ln = norm01(g["latency_ms_mean"], reverse=True); mn = norm01(g["peak_mem_mb"], reverse=True)
        ers = [0.6*q + 0.3*l + 0.1*m for q, l, m in zip(qn, ln, mn)]
        gg = g.copy(); gg["ERS_no_energy"] = ers; out_rows.append(gg)

    out = pd.concat(out_rows, ignore_index=True)
    out.sort_values(["task", "ERS_no_energy"], ascending=[True, False]).to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(out.sort_values(["task","ERS_no_energy"], ascending=[True,False]).head(20))

if __name__ == "__main__":
    main()
