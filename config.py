"""Central configuration for the English -> Hindi MT comparison.

Every value can be overridden with an environment variable of the same name,
e.g.  N_TRAIN_SCRATCH=20000 python -m src.train_transformer
"""
import os
from pathlib import Path


def _env(name, default):
    """Read an env var and cast it to the type of `default`."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return type(default)(raw)


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CKPT_DIR = ROOT / "checkpoints"
for _d in (DATA_DIR, RESULTS_DIR, CKPT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Language pair + dataset
# --------------------------------------------------------------------------- #
SRC_LANG, TGT_LANG = "en", "hi"
SRC_NAME, TGT_NAME = "English", "Hindi"
DATASET_ID = "cfilt/iitb-english-hindi"          # IIT Bombay English-Hindi corpus
SEED = _env("SEED", 42)

# Size of the cleaned training pool that all models draw from (nested subsets).
POOL_SIZE = _env("POOL_SIZE", 100_000)
N_TEST = _env("N_TEST", 500)                      # sentences used for the main comparison
N_TEMP_TEST = _env("N_TEMP_TEST", 200)            # subset used for the temperature sweep
MAX_WORDS = _env("MAX_WORDS", 50)                 # max words per side (train)
MIN_WORDS = _env("MIN_WORDS", 3)

# --------------------------------------------------------------------------- #
# Decoding / evaluation (identical for every model)
# --------------------------------------------------------------------------- #
TEMPERATURES = [0.0, 0.3, 0.7, 1.0]               # 0.0 == greedy decoding
EVAL_BATCH_SIZE = _env("EVAL_BATCH_SIZE", 32)
MAX_NEW_TOKENS = _env("MAX_NEW_TOKENS", 160)

# --------------------------------------------------------------------------- #
# Model 1: Transformer trained from scratch
# --------------------------------------------------------------------------- #
N_TRAIN_SCRATCH = _env("N_TRAIN_SCRATCH", 100_000)
SCRATCH = dict(
    vocab_size=_env("SCRATCH_VOCAB", 16_000),
    d_model=256, n_heads=4, n_enc_layers=3, n_dec_layers=3, d_ff=1024, dropout=0.1,
    max_tokens=100,                               # drop training pairs longer than this (subword tokens)
    batch_size=64, epochs=_env("SCRATCH_EPOCHS", 12), lr=7e-4, warmup=1000,
    label_smoothing=0.1, patience=3,
)

# --------------------------------------------------------------------------- #
# Model 2: mT5 (fine-tuned)
# --------------------------------------------------------------------------- #
N_TRAIN_MT5 = _env("N_TRAIN_MT5", 50_000)
MT5 = dict(
    model_id=_env("MT5_ID", "google/mt5-small"),
    prefix="translate English to Hindi: ",
    max_src=64, max_tgt=96,
    # mT5 has a 250k-token vocabulary: logits for batch x length x 250k floats dominate GPU memory,
    # so use a small micro-batch and accumulate gradients (effective batch = 8 x 4 = 32).
    batch_size=8, grad_accum=4, epochs=_env("MT5_EPOCHS", 2), lr=5e-4, warmup=200,
)

# --------------------------------------------------------------------------- #
# Models 3 & 4: decoder-only LLMs (LLaMA, Gemma) -- LoRA fine-tuning / prompting
# --------------------------------------------------------------------------- #
N_TRAIN_LLM = _env("N_TRAIN_LLM", 10_000)
LLM_IDS = {
    "llama": _env("LLAMA_ID", "meta-llama/Llama-3.2-1B-Instruct"),   # gated: accept licence on Hugging Face
    "gemma": _env("GEMMA_ID", "google/gemma-3-1b-it"),               # gated: accept licence on Hugging Face
}
# (micro-batch, grad-accum) per family; effective batch = 32 either way. Gemma's 262k vocabulary and fp32 weights
# on T4-class GPUs need the smaller micro-batch.
LLM_BATCH = {"llama": (4, 8), "gemma": (2, 16)}
LLM = dict(
    max_len=320,                                   # prompt + target tokens (Hindi needs many LLM tokens)
    max_new_tokens=256,                            # generation cap for decoder-only models
    epochs=_env("LLM_EPOCHS", 1), lr=2e-4, warmup_ratio=0.05,
    lora_r=16, lora_alpha=32, lora_dropout=0.05,
    n_shots=3,                                     # few-shot examples in the prompt
)
PROMPT_HEADER = f"Translate {SRC_NAME} to {TGT_NAME}.\n\n"
