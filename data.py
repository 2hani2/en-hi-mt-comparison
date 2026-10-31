"""Step 1 - obtain, describe, clean and split the parallel corpus.

    python -m src.data

Source : IIT Bombay English-Hindi Parallel Corpus (Kunchukuttan et al., LREC 2018)
         https://www.cfilt.iitb.ac.in/iitb_parallel/
         https://huggingface.co/datasets/cfilt/iitb-english-hindi   (CC BY-NC 4.0)

The corpus ships with official train (1.66M pairs) / validation (520) / test (2,507) splits.
We keep those splits, clean them, and draw a manageable random pool from the training split:

    data/train.jsonl  - POOL_SIZE cleaned pairs (every model uses a nested prefix of this pool)
    data/val.jsonl    - official validation set (lightly cleaned)
    data/test.jsonl   - N_TEST sentences randomly sampled from the official test set
"""
import json
import random
import re
import unicodedata

from . import config as C
from .common import save_json, set_seed, write_jsonl

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
_LATIN = re.compile(r"[A-Za-z]")
_WS = re.compile(r"\s+")


def normalise(text):
    return _WS.sub(" ", unicodedata.normalize("NFC", text)).strip()


def script_fraction(text, pattern):
    letters = [c for c in text if c.isalpha()]
    return sum(1 for c in letters if pattern.match(c)) / len(letters) if letters else 0.0


def keep_pair(src, tgt, strict):
    """Return True if the pair passes the quality filters.

    strict=True  (training data): length limits, length-ratio, script checks, src != tgt
    strict=False (val/test)     : only drop empty or extremely long/short pairs
    """
    if not src or not tgt:
        return False
    ns, nt = len(src.split()), len(tgt.split())
    if strict:
        if not (C.MIN_WORDS <= ns <= C.MAX_WORDS and C.MIN_WORDS <= nt <= C.MAX_WORDS):
            return False
        ratio = nt / ns
        if not (0.4 <= ratio <= 2.5):
            return False
        if src.lower() == tgt.lower():
            return False
        if script_fraction(tgt, _DEVANAGARI) < 0.6:       # Hindi side must be mostly Devanagari
            return False
        if script_fraction(src, _LATIN) < 0.8:            # English side must be mostly Latin
            return False
    else:
        if not (2 <= ns <= 60 and 2 <= nt <= 80):
            return False
    return True


def load_raw():
    from datasets import load_dataset
    try:
        return load_dataset(C.DATASET_ID)
    except Exception as e:                                # fall back to the auto-converted parquet branch
        print("Default load failed (", e, ") - retrying with refs/convert/parquet")
        return load_dataset(C.DATASET_ID, revision="refs/convert/parquet")


def to_pairs(split):
    return [(normalise(r["translation"][C.SRC_LANG]), normalise(r["translation"][C.TGT_LANG])) for r in split]


def describe(pairs):
    import statistics as st
    ns = [len(s.split()) for s, _ in pairs]
    nt = [len(t.split()) for _, t in pairs]
    return {
        "n_pairs": len(pairs),
        "src_words_mean": round(st.mean(ns), 1), "src_words_median": st.median(ns), "src_words_max": max(ns),
        "tgt_words_mean": round(st.mean(nt), 1), "tgt_words_median": st.median(nt), "tgt_words_max": max(nt),
        "src_vocab_types": len({w.lower() for s, _ in pairs for w in s.split()}),
        "tgt_vocab_types": len({w for _, t in pairs for w in t.split()}),
    }


def main():
    set_seed()
    rng = random.Random(C.SEED)
    raw = load_raw()
    print({k: len(v) for k, v in raw.items()})
    raw_sizes = {k: len(v) for k, v in raw.items()}

    # ---- evaluation splits first (so we can remove leakage from the training pool) ----------
    val = [p for p in to_pairs(raw["validation"]) if keep_pair(*p, strict=False)]
    test_all = [p for p in to_pairs(raw["test"]) if keep_pair(*p, strict=False)]
    rng.shuffle(test_all)
    test = test_all[:C.N_TEST]
    heldout_src = {s.lower() for s, _ in val} | {s.lower() for s, _ in test_all}

    # ---- training pool: shuffle first, clean only as many rows as needed ----------------------
    train_split = raw["train"].shuffle(seed=C.SEED)
    take = min(len(train_split), C.POOL_SIZE * 4)          # filters drop ~30-60% of IITB rows
    seen, pool = set(), []
    for src, tgt in to_pairs(train_split.select(range(take))):
        key = (src.lower(), tgt)
        if key in seen or src.lower() in heldout_src:      # de-duplicate and prevent test leakage
            continue
        seen.add(key)
        if keep_pair(src, tgt, strict=True):
            pool.append((src, tgt))
            if len(pool) >= C.POOL_SIZE:
                break
    print(f"train pool: {len(pool)} pairs (from {take} sampled rows)")

    def dump(name, pairs):
        write_jsonl(C.DATA_DIR / f"{name}.jsonl", [{"src": s, "tgt": t} for s, t in pairs])

    dump("train", pool)
    dump("val", val)
    dump("test", test)

    stats = {
        "dataset": C.DATASET_ID, "language_pair": f"{C.SRC_LANG}->{C.TGT_LANG}",
        "raw_split_sizes": raw_sizes, "seed": C.SEED,
        "filters_train": {"words": [C.MIN_WORDS, C.MAX_WORDS], "length_ratio": [0.4, 2.5],
                          "min_devanagari_fraction_tgt": 0.6, "min_latin_fraction_src": 0.8,
                          "dedup": True, "remove_heldout_overlap": True},
        "splits": {"train_pool": describe(pool), "val": describe(val), "test": describe(test)},
        "model_train_sizes": {"transformer": C.N_TRAIN_SCRATCH, "mt5": C.N_TRAIN_MT5, "llm": C.N_TRAIN_LLM},
        "examples": [{"src": s, "tgt": t} for s, t in pool[:5]],
    }
    save_json(C.RESULTS_DIR / "dataset_stats.json", stats)
    print(json.dumps(stats["splits"], indent=2))

    # ---- length histogram -------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        ax[0].hist([len(s.split()) for s, _ in pool], bins=range(0, C.MAX_WORDS + 2, 2), color="#4c72b0")
        ax[0].set_title("English sentence length (train pool)")
        ax[1].hist([len(t.split()) for _, t in pool], bins=range(0, C.MAX_WORDS + 2, 2), color="#dd8452")
        ax[1].set_title("Hindi sentence length (train pool)")
        for a in ax:
            a.set_xlabel("words"); a.set_ylabel("sentence pairs")
        fig.tight_layout()
        fig.savefig(C.RESULTS_DIR / "length_hist.png", dpi=150)
    except Exception as e:
        print("Skipping histogram:", e)


if __name__ == "__main__":
    main()
