"""Model 2 - full fine-tuning of the pretrained mT5 encoder-decoder.

    python -m src.finetune_mt5                  # fine-tune + evaluate
    python -m src.finetune_mt5 --eval_only      # re-evaluate the saved checkpoint

A plain PyTorch loop is used (instead of Trainer) so the script behaves the same across
`transformers` versions. mT5 is trained in fp32 on T4-class GPUs (mT5 overflows in fp16) and in
bf16 on Ampere or newer GPUs.
"""
import argparse
import math
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.optimization import Adafactor

from . import config as C
from .common import (bucket_batches, count_params, cuda_sync, evaluate_model, get_device, load_json, load_split,
                     native_bf16, peak_mem_gb, reset_peak_mem, save_json, set_seed, translate_in_batches)

CKPT = C.CKPT_DIR / "mt5"
TRAIN_LOG = C.RESULTS_DIR / "mt5_train_log.json"


class Encoded:
    """Tokenised parallel data with a padding collate function."""

    def __init__(self, rows, tok, cfg):
        self.src = tok([cfg["prefix"] + r["src"] for r in rows], max_length=cfg["max_src"], truncation=True)["input_ids"]
        self.tgt = tok(text_target=[r["tgt"] for r in rows], max_length=cfg["max_tgt"], truncation=True)["input_ids"]
        self.lengths = [len(s) + len(t) for s, t in zip(self.src, self.tgt)]
        self.pad = tok.pad_token_id

    def batch(self, idx, device):
        src = [self.src[i] for i in idx]
        tgt = [self.tgt[i] for i in idx]
        S, T = max(map(len, src)), max(map(len, tgt))
        ids = torch.full((len(idx), S), self.pad, dtype=torch.long)
        att = torch.zeros((len(idx), S), dtype=torch.long)
        lab = torch.full((len(idx), T), -100, dtype=torch.long)
        for b, (s, t) in enumerate(zip(src, tgt)):
            ids[b, : len(s)] = torch.tensor(s)
            att[b, : len(s)] = 1
            lab[b, : len(t)] = torch.tensor(t)
        return ids.to(device), att.to(device), lab.to(device)


def validation_loss(model, data, device, amp, batch_size=16):
    model.eval()
    total, n_tok = 0.0, 0
    with torch.no_grad():
        for idx in bucket_batches(data.lengths, batch_size, seed=0):
            ids, att, lab = data.batch(idx, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            n = lab.ne(-100).sum().item()
            total += loss.item() * n
            n_tok += n
    return total / max(n_tok, 1)


def finetune(args, tok, device):
    cfg = C.MT5
    train_data = Encoded(load_split("train", args.n_train), tok, cfg)
    val_data = Encoded(load_split("val"), tok, cfg)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_id"]).to(device)     # fp32 master weights
    amp = native_bf16()
    print(f"training pairs: {len(train_data.src)} | bf16 autocast: {amp} | micro-batch {cfg['batch_size']} x accum {cfg['grad_accum']}")

    opt = Adafactor(model.parameters(), lr=cfg["lr"], scale_parameter=False, relative_step=False, warmup_init=False)
    accum = cfg["grad_accum"]
    steps_per_epoch = math.ceil(len(train_data.src) / (cfg["batch_size"] * accum))     # optimizer steps
    total_steps = max(1, steps_per_epoch * args.epochs)
    warm = min(cfg["warmup"], max(1, total_steps // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm, max(0.0, (total_steps - s) / max(1, total_steps - warm))))

    log, step = [], 0
    reset_peak_mem()
    cuda_sync()
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_n, t_ep = 0.0, 0, time.perf_counter()
        batches = bucket_batches(train_data.lengths, cfg["batch_size"], seed=C.SEED + epoch)
        for i, idx in enumerate(batches):
            ids, att, lab = train_data.batch(idx, device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            (loss / accum).backward()
            run_loss += loss.item()
            run_n += 1
            if (i + 1) % accum == 0 or i + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 50 == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} loss {run_loss / run_n:.4f}", flush=True)
        val = validation_loss(model, val_data, device, amp)
        entry = {"epoch": epoch, "train_loss": round(run_loss / max(run_n, 1), 4), "val_loss": round(val, 4),
                 "epoch_time_s": round(time.perf_counter() - t_ep, 1)}
        log.append(entry)
        print(entry)
    cuda_sync()
    train_time = time.perf_counter() - t_start

    model.save_pretrained(CKPT)
    tok.save_pretrained(CKPT)
    save_json(TRAIN_LOG, {"log": log, "train_time_s": round(train_time, 1), "n_train_used": len(train_data.src),
                          "train_peak_gpu_mem_gb": peak_mem_gb(), "bf16": amp})
    return model


def make_translate_fn(model, tok, device):
    model.eval()
    amp = native_bf16()
    prefix = C.MT5["prefix"]

    @torch.no_grad()
    def gen_batch(texts, temperature):
        enc = tok([prefix + t for t in texts], return_tensors="pt", padding=True, truncation=True,
                  max_length=128).to(device)
        kw = dict(max_new_tokens=C.MAX_NEW_TOKENS, num_beams=1)
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
        else:
            kw.update(do_sample=False)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = model.generate(**enc, **kw)
        return tok.batch_decode(out, skip_special_tokens=True)

    def translate_fn(texts, temperature):
        return translate_in_batches(texts, lambda b: gen_batch(b, temperature), C.EVAL_BATCH_SIZE)

    return translate_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=C.N_TRAIN_MT5)
    ap.add_argument("--epochs", type=int, default=C.MT5["epochs"])
    ap.add_argument("--skip_temp", action="store_true")
    ap.add_argument("--eval_only", action="store_true")
    ap.add_argument("--name", default="mt5")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    print("device:", device)
    if args.eval_only:
        tok = AutoTokenizer.from_pretrained(CKPT)
        model = AutoModelForSeq2SeqLM.from_pretrained(CKPT).to(device)
    else:
        tok = AutoTokenizer.from_pretrained(C.MT5["model_id"])
        model = finetune(args, tok, device)
    tlog = load_json(TRAIN_LOG)

    total, trainable = count_params(model)
    meta = {
        "family": "mT5", "architecture": "Encoder-decoder", "adaptation": "Full fine-tuning", "pretrained": True,
        "model_id": C.MT5["model_id"], "tokenizer": f"mT5 SentencePiece ({len(tok)} tokens)",
        "total_params": total, "trainable_params": trainable,
        "n_train": tlog["n_train_used"], "train_time_s": tlog["train_time_s"],
        "train_peak_gpu_mem_gb": tlog["train_peak_gpu_mem_gb"],
    }
    evaluate_model(args.name, make_translate_fn(model, tok, device), meta, skip_temp=args.skip_temp)


if __name__ == "__main__":
    main()
