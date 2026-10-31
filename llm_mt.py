"""Models 3 & 4 - decoder-only LLMs (LLaMA and Gemma) for English -> Hindi translation.

    python -m src.llm_mt --family llama --mode lora        # LoRA fine-tuning (main result)
    python -m src.llm_mt --family llama --mode fewshot --skip_temp
    python -m src.llm_mt --family llama --mode zeroshot --skip_temp
    python -m src.llm_mt --family gemma --mode lora        # same three modes for Gemma

Modes
  zeroshot : instruction + "English: ... Hindi:" prompt, no weights updated
  fewshot  : same prompt preceded by k (default 3) example translations, no weights updated
  lora     : parameter-efficient fine-tuning (LoRA adapters on all attention/MLP projections),
             then zero-shot prompting with the fine-tuned model

Default checkpoints are the 1B releases meta-llama/Llama-3.2-1B-Instruct and google/gemma-3-1b-it. They are used
only as language-model initialisations: prompts are plain "English: ... Hindi:" text (no chat template). Switch to
the base checkpoints or to google/gemma-3-270m with the LLAMA_ID / GEMMA_ID environment variables. The checkpoints
are gated: accept the licences on Hugging Face and export HF_TOKEN (or `huggingface-cli login`).
"""
import argparse
import math
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import config as C
from .common import (bucket_batches, count_params, cuda_sync, evaluate_model, get_device, hf_dtype_kwargs,
                     load_json, load_split, native_bf16, peak_mem_gb, pick_dtype, reset_peak_mem, save_json,
                     set_seed, translate_in_batches)

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DISPLAY = {"llama": "LLaMA", "gemma": "Gemma"}


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def zero_shot_prompt(src):
    return f"{C.PROMPT_HEADER}{C.SRC_NAME}: {src}\n{C.TGT_NAME}:"


def make_few_shot_prompt(shots):
    prefix = C.PROMPT_HEADER + "".join(f"{C.SRC_NAME}: {s}\n{C.TGT_NAME}: {t}\n\n" for s, t in shots)
    return lambda src: f"{prefix}{C.SRC_NAME}: {src}\n{C.TGT_NAME}:"


def pick_shots(rows, k):
    """k fixed, medium-length training pairs (deterministic: first matches in the pool)."""
    shots = [(r["src"], r["tgt"]) for r in rows[:500] if 8 <= len(r["src"].split()) <= 20][:k]
    for r in rows:                                        # fallback if too few medium-length pairs were found
        if len(shots) >= k:
            break
        if (r["src"], r["tgt"]) not in shots:
            shots.append((r["src"], r["tgt"]))
    return shots


# --------------------------------------------------------------------------- #
# Environment check
# --------------------------------------------------------------------------- #

def check_peft_torchao():
    """Kaggle ships an old torchao (0.10) that recent peft versions refuse to work with when building LoRA layers.
    Detect it up front and tell the user the one-line fix instead of crashing after the model download."""
    try:
        from peft.import_utils import is_torchao_available
        is_torchao_available()
    except ImportError as e:
        if "torchao" in str(e):
            raise SystemExit("peft cannot be used with the installed torchao.\n"
                             "Fix: pip uninstall -y torchao   (torchao is not needed here), then run this command again.\n"
                             f"Original error: {e}")
        raise


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #

def load_model(model_id, family, device, load_4bit=False):
    dtype = pick_dtype(allow_fp16=(family != "gemma"))   # Gemma overflows in fp16 -> bf16 (Ampere+) or fp32
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kw = hf_dtype_kwargs(dtype)
    if family == "gemma":
        kw["attn_implementation"] = "eager"              # recommended by Google for training Gemma
    if load_4bit:                                        # experimental: lets a bigger model fit on a T4
        from transformers import BitsAndBytesConfig
        compute = torch.bfloat16 if native_bf16() else torch.float16
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=compute, bnb_4bit_use_double_quant=True)
        kw["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kw).to(device)
    print(f"loaded {model_id} | dtype={dtype} | 4bit={load_4bit}")
    return model, tok, dtype


# --------------------------------------------------------------------------- #
# LoRA fine-tuning
# --------------------------------------------------------------------------- #

def build_examples(rows, tok, max_len):
    """Return [(prompt_ids, target_ids)]. Loss is computed on the target (Hindi) tokens only."""
    ex = []
    for r in rows:
        p = tok(zero_shot_prompt(r["src"]), add_special_tokens=True)["input_ids"]
        t = tok(" " + r["tgt"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(p) + len(t) <= max_len:
            ex.append((p, t))
    return ex


def collate(examples, idx, pad_id, device):
    seqs = [examples[i][0] + examples[i][1] for i in idx]
    L = max(map(len, seqs))
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    att = torch.zeros((len(idx), L), dtype=torch.long)
    lab = torch.full((len(idx), L), -100, dtype=torch.long)
    for b, i in enumerate(idx):
        p, t = examples[i]
        n = len(p) + len(t)
        ids[b, :n] = torch.tensor(p + t)
        att[b, :n] = 1
        lab[b, len(p):n] = torch.tensor(t)
    return ids.to(device), att.to(device), lab.to(device)


def lm_loss(model, examples, tok, device, autocast_dtype, batch_size):
    model.eval()
    lengths = [len(p) + len(t) for p, t in examples]
    total, n_tok = 0.0, 0
    with torch.no_grad():
        for idx in bucket_batches(lengths, batch_size, seed=0):
            ids, att, lab = collate(examples, idx, tok.pad_token_id, device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            n = lab[:, 1:].ne(-100).sum().item()
            total += loss.item() * n
            n_tok += n
    return total / max(n_tok, 1)


def lora_finetune(model, tok, dtype, args, device, name):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    cfg = C.LLM
    if args.load_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                                             lora_dropout=cfg["lora_dropout"], target_modules=LORA_TARGETS,
                                             task_type="CAUSAL_LM"))
    model.print_trainable_parameters()
    total_params, trainable_params = count_params(model)

    train_ex = build_examples(load_split("train", args.n_train), tok, cfg["max_len"])
    val_ex = build_examples(load_split("val")[:300], tok, cfg["max_len"])
    print(f"fine-tuning on {len(train_ex)} examples (after length filter), validating on {len(val_ex)}")

    autocast_dtype = None if dtype == torch.float32 else dtype
    scaler = torch.amp.GradScaler("cuda", enabled=(dtype == torch.float16))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    micro_per_epoch = math.ceil(len(train_ex) / args.bs)
    total_steps = max(1, math.ceil(micro_per_epoch / args.grad_accum) * args.epochs)
    warm = max(1, int(cfg["warmup_ratio"] * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    lengths = [len(p) + len(t) for p, t in train_ex]

    model.config.use_cache = False
    log, opt_step, run_loss, run_n = [], 0, 0.0, 0
    reset_peak_mem()
    cuda_sync()
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = bucket_batches(lengths, args.bs, seed=C.SEED + epoch)
        for i, idx in enumerate(batches):
            ids, att, lab = collate(train_ex, idx, tok.pad_token_id, device)
            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            scaler.scale(loss / args.grad_accum).backward()
            run_loss += loss.item()
            run_n += 1
            if (i + 1) % args.grad_accum == 0 or i + 1 == len(batches):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
                opt_step += 1
                if opt_step % 10 == 0:
                    avg = run_loss / run_n
                    log.append({"step": opt_step, "train_loss": round(avg, 4)})
                    print(f"epoch {epoch} step {opt_step}/{total_steps} loss {avg:.4f}")
                    run_loss, run_n = 0.0, 0
    cuda_sync()
    train_time = time.perf_counter() - t_start
    train_peak = peak_mem_gb()

    val_loss = lm_loss(model, val_ex, tok, device, autocast_dtype, args.bs)
    print(f"validation loss (Hindi tokens): {val_loss:.4f}")
    model.config.use_cache = True
    ckpt = C.CKPT_DIR / name
    model.save_pretrained(ckpt)                           # saves only the LoRA adapter
    save_json(C.RESULTS_DIR / f"{name}_train_log.json", {
        "log": log, "train_time_s": round(train_time, 1), "val_loss": round(val_loss, 4),
        "n_train_used": len(train_ex), "train_peak_gpu_mem_gb": train_peak,
        "total_params": total_params, "trainable_params": trainable_params,
        "effective_batch": args.bs * args.grad_accum, "dtype": str(dtype)})
    return model


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

def make_translate_fn(model, tok, device, prompt_fn):
    model.eval()
    tok.padding_side = "left"                             # required for batched decoder-only generation

    @torch.no_grad()
    def gen_batch(srcs, temperature):
        enc = tok([prompt_fn(s) for s in srcs], return_tensors="pt", padding=True, add_special_tokens=True)
        enc = {k: enc[k].to(device) for k in ("input_ids", "attention_mask")}   # drop e.g. token_type_ids
        kw = dict(max_new_tokens=C.LLM["max_new_tokens"], pad_token_id=tok.pad_token_id,
                  eos_token_id=tok.eos_token_id, stop_strings=["\n"], tokenizer=tok)
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
        else:
            kw.update(do_sample=False, temperature=None, top_k=None, top_p=None)
        out = model.generate(**enc, **kw)
        new_tokens = out[:, enc["input_ids"].shape[1]:]
        texts = tok.batch_decode(new_tokens, skip_special_tokens=True)
        return [t.split("\n")[0].strip() for t in texts]

    def translate_fn(texts, temperature):
        return translate_in_batches(texts, lambda b: gen_batch(b, temperature), max(8, C.EVAL_BATCH_SIZE // 2))

    return translate_fn


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["llama", "gemma"], required=True)
    ap.add_argument("--mode", choices=["zeroshot", "fewshot", "lora"], default="lora")
    ap.add_argument("--model_id", default=None, help="override the Hugging Face model id from config.py")
    ap.add_argument("--n_train", type=int, default=C.N_TRAIN_LLM)
    ap.add_argument("--epochs", type=int, default=C.LLM["epochs"])
    ap.add_argument("--bs", type=int, default=None, help="micro-batch size (default: per family, see config.LLM_BATCH)")
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--load_4bit", action="store_true", help="(experimental) load the base model in 4-bit NF4")
    ap.add_argument("--skip_temp", action="store_true")
    ap.add_argument("--eval_only", action="store_true", help="lora mode: load the saved adapter instead of training")
    args = ap.parse_args()

    args.bs = args.bs or C.LLM_BATCH[args.family][0]
    args.grad_accum = args.grad_accum or C.LLM_BATCH[args.family][1]
    name = f"{args.family}_{args.mode}"
    if args.mode == "lora":
        check_peft_torchao()
    model_id = args.model_id or C.LLM_IDS[args.family]
    set_seed()
    device = get_device()
    print("device:", device, "| run:", name)

    model, tok, dtype = load_model(model_id, args.family, device, args.load_4bit)
    base_total, _ = count_params(model)

    train_time, n_train, train_peak, trainable = 0.0, 0, 0.0, 0
    total = base_total
    if args.mode == "zeroshot":
        prompt_fn, adaptation = zero_shot_prompt, "Zero-shot prompting"
    elif args.mode == "fewshot":
        shots = pick_shots(load_split("train"), C.LLM["n_shots"])
        prompt_fn, adaptation = make_few_shot_prompt(shots), f"{len(shots)}-shot prompting"
    else:
        prompt_fn, adaptation = zero_shot_prompt, f"LoRA fine-tuning (r={C.LLM['lora_r']})"
        if args.eval_only:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, C.CKPT_DIR / name)
        else:
            model = lora_finetune(model, tok, dtype, args, device, name)
        tlog = load_json(C.RESULTS_DIR / f"{name}_train_log.json")
        train_time, n_train = tlog["train_time_s"], tlog["n_train_used"]
        train_peak, total, trainable = tlog["train_peak_gpu_mem_gb"], tlog["total_params"], tlog["trainable_params"]
        if not args.load_4bit:
            model = model.merge_and_unload()               # fuse adapters -> same inference cost as the base model

    meta = {
        "family": DISPLAY[args.family], "architecture": "Decoder-only", "adaptation": adaptation, "pretrained": True,
        "model_id": model_id, "tokenizer": f"{type(tok).__name__} ({len(tok)} tokens)",
        "total_params": total, "trainable_params": trainable, "n_train": n_train,
        "train_time_s": train_time, "train_peak_gpu_mem_gb": train_peak, "dtype": str(dtype),
    }
    evaluate_model(name, make_translate_fn(model, tok, device, prompt_fn), meta, skip_temp=args.skip_temp)


if __name__ == "__main__":
    main()
