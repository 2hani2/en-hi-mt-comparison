| System | T=0.0 | T=0.3 | T=0.7 | T=1.0 |
|---|---|---|---|---|
| Basic Transformer - Trained from scratch | 4.42 / 23.1 | 3.04 / 21.97 | 2.32 / 20.59 | 1.18 / 17.9 |
| mT5 - Full fine-tuning | 1.41 / 16.84 | 1.99 / 19.79 | 1.77 / 19.52 | 0.68 / 17.27 |
| LLaMA - LoRA fine-tuning (r=16) | 9.59 / 35.26 | 7.4 / 33.51 | 5.45 / 29.33 | 1.18 / 20.46 |
| Gemma - LoRA fine-tuning (r=16) | 12.06 / 37.75 | 8.72 / 36.29 | 8.13 / 33.39 | 2.35 / 21.28 |

Cells show BLEU / chrF on the first 100 test sentences. T=0.0 is greedy decoding; T>0 is pure temperature sampling (no top-k/top-p), one seed.
