"""Shared helpers used by every model script.

Kept importable without PyTorch (torch is imported lazily) so that the data
preparation and reporting scripts also run on a laptop.
"""
import json
import random
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from .probes import PROBES

# --------------------------------------------------------------------------- #
# Small IO helpers
# --------------------------------------------------------------------------- #

def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_split(name, n=None):
    """Load data/<name>.jsonl (rows are {"src":..., "tgt":...}); optionally keep the first n rows."""
    rows = read_jsonl(C.DATA_DIR / f"{name}.jsonl")
    return rows[:n] if n else rows


# --------------------------------------------------------------------------- #
# Reproducibility, devices, dtypes (torch imported lazily)
# --------------------------------------------------------------------------- #

def set_seed(seed=C.SEED):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def get_device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return "cpu"


def native_bf16():
    """True only on GPUs with real bf16 hardware (Ampere or newer). T4 is *not* native."""
    import torch
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8


def pick_dtype(allow_fp16=True):
    """bf16 on Ampere+, else fp16 (if allowed), else fp32. CPU always fp32."""
    import torch
    if not torch.cuda.is_available():
        return torch.float32
    if native_bf16():
        return torch.bfloat16
    return torch.float16 if allow_fp16 else torch.float32


def hf_dtype_kwargs(dtype):
    """`torch_dtype=` was renamed to `dtype=` in transformers 4.56+. Support both."""
    try:
        import transformers
        from packaging.version import Version
        if Version(transformers.__version__.split("+")[0]) >= Version("4.56.0"):
            return {"dtype": dtype}
    except Exception:
        pass
    return {"torch_dtype": dtype}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def reset_peak_mem():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def peak_mem_gb():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except ImportError:
        pass
    return 0.0


# --------------------------------------------------------------------------- #
# Batching helpers
# --------------------------------------------------------------------------- #

def bucket_batches(lengths, batch_size, seed, chunk_mult=50):
    """Shuffle, then sort inside large chunks so batches have similar lengths (less padding)."""
    rng = random.Random(seed)
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    chunk = batch_size * chunk_mult
    batches = []
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s:s + chunk], key=lambda i: lengths[i])
        batches += [part[j:j + batch_size] for j in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


def translate_in_batches(texts, translate_batch, batch_size):
    """Run `translate_batch(list[str]) -> list[str]` longest-first, then restore the original order."""
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    out = [None] * len(texts)
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        res = translate_batch([texts[i] for i in idx])
        for i, r in zip(idx, res):
            out[i] = r.strip()
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def corpus_scores(hyps, refs):
    """Corpus-level BLEU (sacreBLEU, default 13a tokenisation) and chrF (char 6-gram)."""
    from sacrebleu.metrics import BLEU, CHRF
    bleu = BLEU().corpus_score(hyps, [refs])
    chrf = CHRF().corpus_score(hyps, [refs])
    return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2)}


def metric_signatures():
    from sacrebleu.metrics import BLEU, CHRF
    try:
        bleu, chrf = BLEU(), CHRF()
        bleu.corpus_score(["a"], [["a"]])          # signatures are only defined after a scoring call
        chrf.corpus_score(["a"], [["a"]])
        return {"bleu": str(bleu.get_signature()), "chrf": str(chrf.get_signature())}
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Heuristic quality flags (support for the responsible-AI analysis)
# --------------------------------------------------------------------------- #
_LATIN = re.compile(r"[A-Za-z]")
_NUM = re.compile(r"\d+(?:[.,]\d+)*")
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")


def latin_fraction(text):
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if _LATIN.match(c)) / len(letters) if letters else 0.0


def quality_flags(srcs, hyps, refs, max_examples=15):
    """Cheap automatic red flags. They are *heuristics*: they point a human reviewer at suspicious outputs.

    empty            - nothing was generated
    copied_source    - output is (mostly) Latin script, i.e. left untranslated
    length_outlier   - output is <0.5x or >2x the reference length (omission / over-generation)
    repetition       - the same word 3-gram occurs 3+ times (degenerate looping)
    number_mismatch  - a number in the source does not appear in the output
    """
    kinds = ["empty", "copied_source", "length_outlier", "repetition", "number_mismatch"]
    counts = {k: 0 for k in kinds}
    examples = {k: [] for k in kinds}
    for i, (s, h, r) in enumerate(zip(srcs, hyps, refs)):
        found = []
        if not h.strip():
            found.append("empty")
        else:
            if h.strip() == s.strip() or latin_fraction(h) > 0.5:
                found.append("copied_source")
            ratio = len(h.split()) / max(len(r.split()), 1)
            if ratio < 0.5 or ratio > 2.0:
                found.append("length_outlier")
            toks = h.split()
            tri = Counter(zip(toks, toks[1:], toks[2:]))
            if tri and max(tri.values()) >= 3:
                found.append("repetition")
            h_norm = h.translate(_DEV_DIGITS).replace(",", "")
            if any(n.replace(",", "") not in h_norm for n in set(_NUM.findall(s))):
                found.append("number_mismatch")
        for k in found:
            counts[k] += 1
            if len(examples[k]) < max_examples:
                examples[k].append({"idx": i, "src": s, "hyp": h, "ref": r})
    n = max(len(hyps), 1)
    rates = {k: round(100 * counts[k] / n, 1) for k in kinds}
    return {"counts": counts, "rate_pct": rates, "examples": examples}


# --------------------------------------------------------------------------- #
# The evaluation harness shared by all four models
# --------------------------------------------------------------------------- #

def evaluate_model(name, translate_fn, meta, test=None, temperatures=None, n_temp=None, skip_temp=False):
    """Evaluate one system under identical conditions and write results/<name>.json.

    translate_fn(list[str], temperature: float) -> list[str]
        temperature == 0.0 must mean greedy decoding. Sampling uses pure
        temperature sampling (no top-k / top-p) so the effect of T is isolated.
    meta: dict with family, architecture, adaptation, total_params, trainable_params,
          train_time_s, n_train, train_peak_gpu_mem_gb, ... (merged into the result file)
    """
    test = test or load_split("test")
    temperatures = temperatures or C.TEMPERATURES
    n_temp = min(n_temp or C.N_TEMP_TEST, len(test))
    srcs = [r["src"] for r in test]
    refs = [r["tgt"] for r in test]
    out_dir = C.RESULTS_DIR

    # ---- main run: greedy decoding on the full test set ------------------------------------
    print(f"[{name}] translating {len(srcs)} test sentences (greedy)...")
    set_seed()
    reset_peak_mem()
    cuda_sync()
    t0 = time.perf_counter()
    hyps = translate_fn(srcs, 0.0)
    cuda_sync()
    infer_s = time.perf_counter() - t0
    infer_peak = peak_mem_gb()

    scores = corpus_scores(hyps, refs)
    flags = quality_flags(srcs, hyps, refs)
    write_jsonl(out_dir / f"{name}_translations.jsonl",
                [{"src": s, "ref": r, "hyp": h} for s, r, h in zip(srcs, refs, hyps)])
    print(f"[{name}] BLEU={scores['bleu']}  chrF={scores['chrf']}  inference={infer_s:.1f}s")

    # ---- temperature analysis -------------------------------------------------------------
    temp_results = {}
    if not skip_temp:
        temp_results["0.0"] = {**corpus_scores(hyps[:n_temp], refs[:n_temp]), "n": n_temp}
        for T in temperatures:
            if T <= 0:
                continue
            print(f"[{name}] sampling at temperature {T} on {n_temp} sentences...")
            set_seed()
            h = translate_fn(srcs[:n_temp], T)
            temp_results[str(T)] = {**corpus_scores(h, refs[:n_temp]), "n": n_temp,
                                    "flags_pct": quality_flags(srcs[:n_temp], h, refs[:n_temp])["rate_pct"]}
            write_jsonl(out_dir / f"{name}_temp{T}.jsonl",
                        [{"src": s, "ref": r, "hyp": x} for s, r, x in zip(srcs[:n_temp], refs[:n_temp], h)])

    # ---- responsible-AI probe sentences (greedy) -------------------------------------------
    probe_hyps = translate_fn([p["text"] for p in PROBES], 0.0)
    save_json(out_dir / f"{name}_probes.json",
              [{"id": p["id"], "category": p["category"], "src": p["text"], "hyp": h}
               for p, h in zip(PROBES, probe_hyps)])

    result = {
        "name": name,
        **meta,
        **scores,
        "n_test": len(srcs),
        "inference_time_s": round(infer_s, 2),
        "ms_per_sentence": round(1000 * infer_s / len(srcs), 1),
        "sentences_per_s": round(len(srcs) / infer_s, 2),
        "inference_peak_gpu_mem_gb": infer_peak,
        "flags": {"counts": flags["counts"], "rate_pct": flags["rate_pct"]},
        "temperature_analysis": temp_results,
        "metric_signatures": metric_signatures(),
        "device": device_info(),
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    save_json(out_dir / f"{name}.json", result)
    save_json(out_dir / f"{name}_flag_examples.json", flags["examples"])
    return result
