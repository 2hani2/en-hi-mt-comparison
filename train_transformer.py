"""Model 1 - basic encoder-decoder Transformer, trained from scratch.

    python -m src.train_transformer                     # train + evaluate
    python -m src.train_transformer --eval_only         # re-evaluate the saved checkpoint

Design: joint English+Hindi SentencePiece-BPE vocabulary, shared (tied) embedding / output
matrix, sinusoidal positions, pre-LayerNorm encoder-decoder (torch.nn.Transformer),
label smoothing, inverse-sqrt warm-up schedule, mixed precision on GPU, early stopping
on validation loss. No pretraining of any kind.
"""
import argparse
import math
import re
import time
import warnings

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import config as C
from .common import (count_params, cuda_sync, evaluate_model, get_device, bucket_batches, load_json,
                     load_split, peak_mem_gb, reset_peak_mem, save_json, set_seed, translate_in_batches)

warnings.filterwarnings("ignore", message=".*enable_nested_tensor.*")

PAD, UNK, BOS, EOS = 0, 1, 2, 3
SPM_PREFIX = C.CKPT_DIR / "transformer_spm"
CKPT = C.CKPT_DIR / "transformer.pt"
TRAIN_LOG = C.RESULTS_DIR / "transformer_train_log.json"


# --------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------- #

def train_tokenizer(train_rows, vocab_size):
    corpus = C.CKPT_DIR / "spm_corpus.txt"
    with open(corpus, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(r["src"] + "\n" + r["tgt"] + "\n")
    while True:
        try:
            spm.SentencePieceTrainer.train(
                input=str(corpus), model_prefix=str(SPM_PREFIX), vocab_size=vocab_size,
                model_type="bpe", character_coverage=0.9995,
                pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS, minloglevel=2,
            )
            return
        except RuntimeError as e:                      # tiny corpora cannot support a big vocabulary
            m = re.search(r"<=\s*(\d+)", str(e))
            if "Vocabulary size too high" not in str(e) or not m or int(m.group(1)) >= vocab_size:
                raise
            vocab_size = int(m.group(1))
            print(f"corpus too small for the requested vocabulary - using vocab_size={vocab_size}")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab, d_model, n_heads, n_enc, n_dec, d_ff, dropout, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.drop = nn.Dropout(dropout)
        self.register_buffer("pe", self._sinusoid(max_len, d_model), persistent=False)
        self.core = nn.Transformer(d_model=d_model, nhead=n_heads, num_encoder_layers=n_enc,
                                   num_decoder_layers=n_dec, dim_feedforward=d_ff, dropout=dropout,
                                   batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, vocab, bias=False)
        self.out.weight = self.emb.weight                      # tie input/output embeddings
        for p in self.core.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        nn.init.normal_(self.emb.weight, mean=0.0, std=d_model ** -0.5)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()

    @staticmethod
    def _sinusoid(max_len, d):
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d))
        pe = torch.zeros(max_len, d)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)

    def embed(self, x):
        return self.drop(self.emb(x) * math.sqrt(self.d_model) + self.pe[:, : x.size(1)])

    def encode(self, src, src_pad):
        return self.core.encoder(self.embed(src), src_key_padding_mask=src_pad)

    def decode(self, tgt_in, memory, src_pad, tgt_pad=None):
        T = tgt_in.size(1)
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt_in.device), diagonal=1)
        h = self.core.decoder(self.embed(tgt_in), memory, tgt_mask=causal,
                              tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad)
        return self.out(h)

    def forward(self, src, tgt_in):
        src_pad = src.eq(PAD)
        memory = self.encode(src, src_pad)
        return self.decode(tgt_in, memory, src_pad, tgt_in.eq(PAD))


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

class PairData:
    def __init__(self, rows, sp, max_tokens):
        self.src, self.tgt = [], []
        for r in rows:
            s, t = sp.encode(r["src"]), sp.encode(r["tgt"])
            if 0 < len(s) <= max_tokens and 0 < len(t) <= max_tokens:
                self.src.append(s)
                self.tgt.append(t)
        self.lengths = [len(s) + len(t) for s, t in zip(self.src, self.tgt)]

    def batch(self, idx, device):
        src = [self.src[i] for i in idx]
        tgt = [self.tgt[i] for i in idx]
        B, S, T = len(idx), max(map(len, src)), max(map(len, tgt)) + 1
        src_t = torch.full((B, S), PAD, dtype=torch.long)
        tin = torch.full((B, T), PAD, dtype=torch.long)
        tout = torch.full((B, T), PAD, dtype=torch.long)
        for b, (s, t) in enumerate(zip(src, tgt)):
            src_t[b, : len(s)] = torch.tensor(s)
            tin[b, : len(t) + 1] = torch.tensor([BOS] + t)
            tout[b, : len(t) + 1] = torch.tensor(t + [EOS])
        return src_t.to(device), tin.to(device), tout.to(device)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

def build_model(vocab, cfg=None):
    cfg = cfg or C.SCRATCH
    return Seq2SeqTransformer(vocab, cfg["d_model"], cfg["n_heads"], cfg["n_enc_layers"],
                              cfg["n_dec_layers"], cfg["d_ff"], cfg["dropout"])


def evaluate_loss(model, data, device, batch_size=128):
    model.eval()
    total, n_tok = 0.0, 0
    batches = bucket_batches(data.lengths, batch_size, seed=0)
    with torch.no_grad():
        for idx in batches:
            src, tin, tout = data.batch(idx, device)
            logits = model(src, tin)
            total += F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tout.reshape(-1),
                                     ignore_index=PAD, reduction="sum").item()
            n_tok += tout.ne(PAD).sum().item()
    return total / max(n_tok, 1)


def train(args, sp, device):
    cfg = C.SCRATCH
    train_rows = load_split("train", args.n_train)
    val_rows = load_split("val")
    train_data = PairData(train_rows, sp, cfg["max_tokens"])
    val_data = PairData(val_rows, sp, cfg["max_tokens"])
    print(f"training pairs after length filter: {len(train_data.src)} | validation pairs: {len(val_data.src)}")

    model = build_model(sp.get_piece_size()).to(device)
    total, _ = count_params(model)
    print(f"model parameters: {total / 1e6:.2f}M")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=(0.9, 0.98), eps=1e-9, weight_decay=0.01)
    warm = cfg["warmup"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / warm, math.sqrt(warm / (s + 1))))
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_state, bad, log = float("inf"), None, 0, []
    reset_peak_mem()
    cuda_sync()
    t_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        t_ep, run_loss, run_tok = time.perf_counter(), 0.0, 0
        for idx in bucket_batches(train_data.lengths, cfg["batch_size"], seed=C.SEED + epoch):
            src, tin, tout = train_data.batch(idx, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(src, tin)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), tout.reshape(-1),
                                   ignore_index=PAD, label_smoothing=cfg["label_smoothing"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            n = tout.ne(PAD).sum().item()
            run_loss += loss.item() * n
            run_tok += n
        val_loss = evaluate_loss(model, val_data, device)
        entry = {"epoch": epoch, "train_loss": round(run_loss / run_tok, 4), "val_loss": round(val_loss, 4),
                 "val_ppl": round(math.exp(min(val_loss, 20)), 2), "epoch_time_s": round(time.perf_counter() - t_ep, 1)}
        log.append(entry)
        print(entry)
        if val_loss < best - 1e-4:
            best, bad = val_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                print("early stopping")
                break
    cuda_sync()
    train_time = time.perf_counter() - t_start
    model.load_state_dict(best_state)
    torch.save({"state": best_state, "cfg": dict(cfg), "vocab": sp.get_piece_size()}, CKPT)
    save_json(TRAIN_LOG, {"log": log, "train_time_s": round(train_time, 1), "best_val_loss": round(best, 4),
                          "n_train_used": len(train_data.src), "train_peak_gpu_mem_gb": peak_mem_gb(),
                          "epochs_run": len(log)})
    return model


# --------------------------------------------------------------------------- #
# Inference (greedy / temperature sampling, no KV-cache - the model is small)
# --------------------------------------------------------------------------- #

def make_translate_fn(model, sp, device):
    model.eval()
    use_amp = device.type == "cuda"

    @torch.no_grad()
    def decode_batch(texts, temperature):
        ids = [(sp.encode(t)[:128] or [UNK]) for t in texts]
        B, S = len(ids), max(map(len, ids))
        src = torch.full((B, S), PAD, dtype=torch.long)
        for b, s in enumerate(ids):
            src[b, : len(s)] = torch.tensor(s)
        src = src.to(device)
        src_pad = src.eq(PAD)
        max_steps = min(2 * S + 10, C.MAX_NEW_TOKENS)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            memory = model.encode(src, src_pad)
            ys = torch.full((B, 1), BOS, dtype=torch.long, device=device)
            done = torch.zeros(B, dtype=torch.bool, device=device)
            for _ in range(max_steps):
                logits = model.decode(ys, memory, src_pad)[:, -1].float()
                logits[:, [PAD, BOS, UNK]] = float("-inf")
                if temperature <= 0:
                    nxt = logits.argmax(-1)
                else:
                    nxt = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).squeeze(1)
                nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
                ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
                done |= nxt.eq(EOS)
                if done.all():
                    break
        out = []
        for row in ys[:, 1:].tolist():
            toks = []
            for t in row:
                if t in (EOS, PAD):
                    break
                toks.append(t)
            out.append(sp.decode(toks))
        return out

    def translate_fn(texts, temperature):
        return translate_in_batches(texts, lambda b: decode_batch(b, temperature), C.EVAL_BATCH_SIZE)

    return translate_fn


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=C.N_TRAIN_SCRATCH)
    ap.add_argument("--epochs", type=int, default=C.SCRATCH["epochs"])
    ap.add_argument("--skip_temp", action="store_true", help="skip the temperature sweep")
    ap.add_argument("--eval_only", action="store_true", help="load the saved checkpoint instead of training")
    ap.add_argument("--name", default="transformer")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    print("device:", device)

    tok_time = 0.0
    if args.eval_only:
        sp = spm.SentencePieceProcessor(model_file=str(SPM_PREFIX) + ".model")
        ck = torch.load(CKPT, map_location="cpu", weights_only=True)
        model = build_model(ck["vocab"], ck["cfg"]).to(device)
        model.load_state_dict(ck["state"])
        tlog = load_json(TRAIN_LOG)
    else:
        t0 = time.perf_counter()
        train_tokenizer(load_split("train", args.n_train), C.SCRATCH["vocab_size"])
        tok_time = time.perf_counter() - t0
        sp = spm.SentencePieceProcessor(model_file=str(SPM_PREFIX) + ".model")
        model = train(args, sp, device)
        tlog = load_json(TRAIN_LOG)

    total, trainable = count_params(model)
    meta = {
        "family": "Basic Transformer", "architecture": "Encoder-decoder",
        "adaptation": "Trained from scratch", "pretrained": False,
        "model_id": f"custom {C.SCRATCH['n_enc_layers']}+{C.SCRATCH['n_dec_layers']} layers, d={C.SCRATCH['d_model']}",
        "tokenizer": f"SentencePiece BPE (joint, {sp.get_piece_size()} tokens)",
        "total_params": total, "trainable_params": trainable,
        "n_train": tlog["n_train_used"], "train_time_s": tlog["train_time_s"],
        "tokenizer_train_time_s": round(tok_time, 1),
        "train_peak_gpu_mem_gb": tlog["train_peak_gpu_mem_gb"],
    }
    evaluate_model(args.name, make_translate_fn(model, sp, device), meta, skip_temp=args.skip_temp)


if __name__ == "__main__":
    main()
