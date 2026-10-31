| System | Model | Architecture | Total params (M) | Trainable params (M) | Train pairs | Train time (min) | Inference (s, full test) | ms / sentence | BLEU | chrF |
|---|---|---|---|---|---|---|---|---|---|---|
| Basic Transformer - Trained from scratch | custom 3+3 layers, d=256 | Encoder-decoder | 9.6 | 9.63 | 49985 | 5.0 | 6.0 | 12.0 | 3.46 | 22.25 |
| mT5 - Full fine-tuning | google/mt5-small | Encoder-decoder | 300.2 | 300.18 | 20000 | 16.5 | 38.6 | 77.1 | 1.98 | 17.92 |
| LLaMA - LoRA fine-tuning (r=16) | meta-llama/Llama-3.2-1B-Instruct | Decoder-only | 1247.1 | 11.27 | 5000 | 3.6 | 155.9 | 311.7 | 9.19 | 34.62 |
| Gemma - LoRA fine-tuning (r=16) | google/gemma-3-1b-it | Decoder-only | 1012.9 | 13.05 | 5000 | 10.7 | 200.2 | 400.5 | 12.86 | 37.86 |
| Gemma - 3-shot prompting | google/gemma-3-1b-it | Decoder-only | 999.9 | 0 (frozen) | – | – | 183.5 | 366.9 | 13.0 | 39.33 |
| Gemma - Zero-shot prompting | google/gemma-3-1b-it | Decoder-only | 999.9 | 0 (frozen) | – | – | 142.3 | 284.6 | 11.34 | 35.04 |
| LLaMA - 3-shot prompting | meta-llama/Llama-3.2-1B-Instruct | Decoder-only | 1235.8 | 0 (frozen) | – | – | 134.0 | 268.0 | 9.2 | 34.33 |
| LLaMA - Zero-shot prompting | meta-llama/Llama-3.2-1B-Instruct | Decoder-only | 1235.8 | 0 (frozen) | – | – | 133.9 | 267.7 | 9.23 | 36.21 |
