"""Step 3 - aggregate everything in results/ into tables, plots and the README.

    python -m src.compare

Reads  results/<system>.json, *_probes.json, *_train_log.json, tokenizer_stats.json
Writes results/results_summary.csv, results_table.md, temperature_table.md, flags_table.md,
       probe_comparison.md, quality_comparison.png, compute_comparison.png,
       temperature_analysis.png, training_curves.png
and replaces the <!-- AUTO:...:START --> ... <!-- AUTO:...:END --> blocks inside README.md.
"""
import re

from . import config as C
from .common import load_json

PRIMARY = ["transformer", "mt5", "llama_lora", "gemma_lora"]     # the four systems of the assignment
R = C.RESULTS_DIR


# --------------------------------------------------------------------------- #

def load_results():
    out = {}
    for p in sorted(R.glob("*.json")):
        try:
            d = load_json(p)
        except Exception:
            continue
        if isinstance(d, dict) and "bleu" in d and "name" in d:
            out[d["name"]] = d
    return out


def label(d):
    return f"{d['family']} - {d['adaptation']}"


def short(d):
    return d["name"].replace("_", " ")


def md_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(c).replace("|", "\\|").replace("\n", " ") for c in r) + " |")
    return "\n".join(lines)


def fmt_min(sec):
    return "–" if not sec else f"{sec / 60:.1f}"


def main_rows(results, names):
    rows = []
    for n in names:
        d = results[n]
        rows.append([
            label(d), d.get("model_id", ""), d["architecture"], f"{d['total_params'] / 1e6:.1f}",
            f"{d['trainable_params'] / 1e6:.2f}" if d["trainable_params"] else "0 (frozen)",
            d["n_train"] or "–", fmt_min(d["train_time_s"]), f"{d['inference_time_s']:.1f}", d["ms_per_sentence"],
            d["bleu"], d["chrf"],
        ])
    return rows


MAIN_HEADERS = ["System", "Model", "Architecture", "Total params (M)", "Trainable params (M)", "Train pairs",
                "Train time (min)", "Inference (s, full test)", "ms / sentence", "BLEU", "chrF"]


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #

def build_tables(results):
    primary = [n for n in PRIMARY if n in results]
    extra = [n for n in results if n not in primary]
    tables = {}
    tables["main"] = md_table(MAIN_HEADERS, main_rows(results, primary)) if primary else "_No results yet - run the training scripts._"
    tables["extended"] = md_table(MAIN_HEADERS, main_rows(results, primary + extra)) if results else ""

    # temperature
    temps = sorted({t for d in results.values() for t in d.get("temperature_analysis", {})}, key=float)
    rows = []
    for n in primary + extra:
        ta = results[n].get("temperature_analysis", {})
        if ta:
            rows.append([label(results[n])] + [f"{ta[t]['bleu']} / {ta[t]['chrf']}" if t in ta else "–" for t in temps])
    tables["temperature"] = md_table(["System"] + [f"T={t}" for t in temps], rows) if rows else "_No temperature results yet._"
    tables["temperature_note"] = (f"Cells show BLEU / chrF on the first {next(iter(results.values())).get('temperature_analysis', {}).get('0.0', {}).get('n', C.N_TEMP_TEST) if results else C.N_TEMP_TEST} "
                                  "test sentences. T=0.0 is greedy decoding; T>0 is pure temperature sampling (no top-k/top-p), one seed.")

    # heuristic flags
    kinds = ["empty", "copied_source", "length_outlier", "repetition", "number_mismatch"]
    rows = [[label(results[n])] + [f"{results[n]['flags']['rate_pct'][k]}%" for k in kinds] for n in primary + extra]
    tables["flags"] = md_table(["System"] + kinds, rows) if rows else ""

    # tokenizer statistics
    tk = R / "tokenizer_stats.json"
    if tk.exists():
        t = load_json(tk)
        rows = [[k, v["vocab_size"], v["tokens_per_hi_word"], v["tokens_per_en_word"]] for k, v in t.items()]
        tables["tokenizer"] = md_table(["Tokenizer", "Vocab size", "Tokens / Hindi word", "Tokens / English word"], rows)
    else:
        tables["tokenizer"] = "_Run `python -m src.tokenizer_stats` to fill this table._"

    # dataset statistics
    ds = R / "dataset_stats.json"
    if ds.exists():
        s = load_json(ds)
        rows = [[k, v["n_pairs"], v["src_words_mean"], v["tgt_words_mean"], v["src_words_max"], v["tgt_words_max"],
                 v["src_vocab_types"], v["tgt_vocab_types"]] for k, v in s["splits"].items()]
        tables["dataset"] = (md_table(["Split", "Sentence pairs", "EN words (mean)", "HI words (mean)", "EN words (max)",
                                       "HI words (max)", "EN word types", "HI word types"], rows)
                             + f"\n\nRaw official split sizes: {s['raw_split_sizes']}. "
                               f"Training sizes used per model: {s['model_train_sizes']}.")
    else:
        tables["dataset"] = "_Run `python -m src.data` and `python -m src.compare` to fill the dataset table._"

    # best model
    if primary:
        by_chrf = max(primary, key=lambda n: (results[n]["chrf"], results[n]["bleu"]))
        by_bleu = max(primary, key=lambda n: results[n]["bleu"])
        fastest = min(primary, key=lambda n: results[n]["ms_per_sentence"])
        b = results[by_chrf]
        txt = (f"**Best overall (highest chrF, ties broken by BLEU): {label(b)}** - BLEU {b['bleu']}, chrF {b['chrf']}. "
               f"Highest BLEU: {label(results[by_bleu])} ({results[by_bleu]['bleu']}). "
               f"Fastest inference: {label(results[fastest])} ({results[fastest]['ms_per_sentence']} ms/sentence).")
        tables["best"] = txt
    else:
        tables["best"] = "_Results not available yet._"
    return tables


def probe_table(results):
    names = [n for n in PRIMARY if n in results and (R / f"{n}_probes.json").exists()]
    if not names:
        return "_No probe outputs yet._"
    data = {n: load_json(R / f"{n}_probes.json") for n in names}
    rows = []
    for i, p in enumerate(data[names[0]]):
        rows.append([p["id"], p["category"], p["src"]] + [data[n][i]["hyp"] for n in names])
    return md_table(["#", "Category", "English source"] + [results[n]["family"] for n in names], rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def make_plots(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = [n for n in PRIMARY if n in results]
    if not primary:
        return
    names = [f"{results[n]['family']}\n{results[n]['adaptation'].split('(')[0].strip()}" for n in primary]
    colors = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"][: len(primary)]

    # quality
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, metric in zip(ax, ["bleu", "chrf"]):
        vals = [results[n][metric] for n in primary]
        bars = a.bar(names, vals, color=colors)
        a.bar_label(bars, fmt="%.1f")
        a.set_title(metric.upper() + " (higher is better)")
        a.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(R / "quality_comparison.png", dpi=150)
    plt.close(fig)

    # compute
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    spec = [("total_params", "Total parameters (M)", 1e6), ("train_time_s", "Training / fine-tuning time (min)", 60),
            ("ms_per_sentence", "Inference (ms / sentence)", 1)]
    for a, (key, title, div) in zip(ax, spec):
        vals = [results[n][key] / div for n in primary]
        bars = a.bar(names, vals, color=colors)
        a.bar_label(bars, fmt="%.1f")
        a.set_title(title + (" - log scale" if key == "total_params" else ""))
        if key == "total_params":
            a.set_yscale("log")                      # parameters span ~2 orders of magnitude
        a.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(R / "compute_comparison.png", dpi=150)
    plt.close(fig)

    # temperature
    with_t = [n for n in results if results[n].get("temperature_analysis")]
    if with_t:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        for n in with_t:
            ta = results[n]["temperature_analysis"]
            ts = sorted(ta, key=float)
            for a, metric in zip(ax, ["bleu", "chrf"]):
                a.plot([float(t) for t in ts], [ta[t][metric] for t in ts], marker="o", label=short(results[n]))
        for a, metric in zip(ax, ["BLEU", "chrF"]):
            a.set_xlabel("sampling temperature (0 = greedy)")
            a.set_ylabel(metric)
            a.set_title(f"{metric} vs temperature")
        ax[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(R / "temperature_analysis.png", dpi=150)
        plt.close(fig)

    # training curves
    panels = []
    for n, title in [("transformer", "Transformer (scratch)"), ("mt5", "mT5 fine-tuning")]:
        p = R / f"{n}_train_log.json"
        if p.exists() and load_json(p)["log"]:
            panels.append((title, load_json(p)["log"], "epoch"))
    for n in ["llama_lora", "gemma_lora"]:
        p = R / f"{n}_train_log.json"
        if p.exists() and load_json(p)["log"]:
            panels.append((f"{n} LoRA", load_json(p)["log"], "step"))
    if panels:
        fig, ax = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4), squeeze=False)
        for a, (title, log, xkey) in zip(ax[0], panels):
            xs = [r.get(xkey, k + 1) for k, r in enumerate(log)]
            a.plot(xs, [r["train_loss"] for r in log], marker="o", label="train loss")
            if "val_loss" in log[0]:
                a.plot(xs, [r["val_loss"] for r in log], marker="s", label="val loss")
            a.set_title(title, fontsize=10)
            a.set_xlabel(xkey)
            a.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(R / "training_curves.png", dpi=150)
        plt.close(fig)


# --------------------------------------------------------------------------- #

def update_readme(tables):
    readme = C.ROOT / "README.md"
    if not readme.exists():
        return
    text = readme.read_text(encoding="utf-8")
    for key, content in tables.items():
        pat = re.compile(rf"(<!-- AUTO:{key}:START -->)(.*?)(<!-- AUTO:{key}:END -->)", re.S)
        text = pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(3)}", text)
    readme.write_text(text, encoding="utf-8")
    print("README.md updated")


def main():
    results = load_results()
    if not results:
        print("No result files found in results/. Run the model scripts first.")
        return
    tables = build_tables(results)
    tables["probes"] = probe_table(results)

    import pandas as pd
    cols = ["name", "family", "architecture", "adaptation", "model_id", "tokenizer", "total_params", "trainable_params",
            "n_train", "train_time_s", "train_peak_gpu_mem_gb", "inference_time_s", "ms_per_sentence",
            "inference_peak_gpu_mem_gb", "bleu", "chrf", "device"]
    pd.DataFrame([{c: d.get(c) for c in cols} for d in results.values()]).to_csv(R / "results_summary.csv", index=False)

    (R / "results_table.md").write_text(tables["extended"] + "\n", encoding="utf-8")
    (R / "temperature_table.md").write_text(tables["temperature"] + "\n\n" + tables["temperature_note"] + "\n", encoding="utf-8")
    (R / "flags_table.md").write_text(tables["flags"] + "\n", encoding="utf-8")
    (R / "probe_comparison.md").write_text(tables["probes"] + "\n", encoding="utf-8")
    make_plots(results)
    update_readme(tables)
    print(tables["main"])
    print()
    print(tables["best"])


if __name__ == "__main__":
    main()
