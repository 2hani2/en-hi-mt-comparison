"""Tokenisation efficiency of each model on the test set (supports the tokenization discussion).

    python -m src.tokenizer_stats

Reports, for every tokenizer, the average number of sub-word tokens per Hindi word (reference
translations) and per English word (sources). A higher "tokens / Hindi word" means the model
needs more positions - more compute, more memory and longer generation - for the same sentence.
Tokenizers that cannot be downloaded (e.g. gated repos without a token) are skipped.
"""
from . import config as C
from .common import load_split, save_json


def avg_tokens_per_word(encode, texts):
    n_tok = sum(len(encode(t)) for t in texts)
    n_words = sum(len(t.split()) for t in texts)
    return round(n_tok / max(n_words, 1), 2)


def main():
    test = load_split("test")
    src = [r["src"] for r in test]
    tgt = [r["tgt"] for r in test]
    stats = {}

    spm_model = C.CKPT_DIR / "transformer_spm.model"
    if spm_model.exists():
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(model_file=str(spm_model))
        stats["Transformer (SentencePiece BPE, joint)"] = {
            "vocab_size": sp.get_piece_size(),
            "tokens_per_hi_word": avg_tokens_per_word(sp.encode, tgt),
            "tokens_per_en_word": avg_tokens_per_word(sp.encode, src)}

    try:
        from transformers import AutoTokenizer
    except ImportError:
        AutoTokenizer = None
        print("transformers not installed - skipping the Hugging Face tokenizers")
    hf = {"mT5": C.MT5["model_id"], "LLaMA": C.LLM_IDS["llama"], "Gemma": C.LLM_IDS["gemma"]} if AutoTokenizer else {}
    for name, model_id in hf.items():
        try:
            tok = AutoTokenizer.from_pretrained(model_id)
            enc = lambda t, tok=tok: tok(t, add_special_tokens=False)["input_ids"]
            stats[f"{name} ({model_id})"] = {
                "vocab_size": len(tok),
                "tokens_per_hi_word": avg_tokens_per_word(enc, tgt),
                "tokens_per_en_word": avg_tokens_per_word(enc, src)}
        except Exception as e:
            print(f"skipping {name}: {e}")

    save_json(C.RESULTS_DIR / "tokenizer_stats.json", stats)
    for k, v in stats.items():
        print(k, v)


if __name__ == "__main__":
    main()
