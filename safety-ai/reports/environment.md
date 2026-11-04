# Training environment (docs/TRAINING_SPEC.md 6.1)

Detected 2026-11-04T17:36:11Z with `python -m safety_ai.training.environment`.

| item | value |
|---|---|
| OS | Windows-11-10.0.26200-SP0 |
| Python | 3.12.10 |
| torch | 2.14.1+cu126 (CUDA build 12.6, cuDNN 91002) |
| ultralytics | 8.4.201 |
| CPU | Intel(R) Core(TM) i5-10400F CPU @ 2.90GHz, 6 cores / 12 threads |
| RAM | 15.9 GiB total, 3.3 GiB available |
| CUDA available | yes |
| GPU 0 | NVIDIA GeForce GTX 1660 SUPER, 6143 MiB VRAM, 2319 MiB in use, driver 610.88, compute capability 7.5 |
| MPS | n/a (not an Apple Silicon machine) |
| **mode** | **GPU**, device `cuda:0`: CUDA device 0 (NVIDIA GeForce GTX 1660 SUPER) has 6.0 GiB VRAM >= 6 GB |

Rule (spec 6.1): GPU mode when CUDA has >= 6 GB VRAM, otherwise CPU mode (MPS counts as CPU mode with `device="mps"` tried first).
