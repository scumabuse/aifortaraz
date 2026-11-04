**Stage:** S0 done, S1 in progress (T1 agent; S4-S7 follow later).

- Done: environment detected (`python -m safety_ai.training.environment` -> `reports/environment.md`: GTX 1660 SUPER 6143 MiB, CUDA 12.6, torch 2.14.1+cu126, ultralytics 8.4.201, i5-10400F 6C/12T, 15.9 GiB RAM, mode GPU); archive downloaded and verified (647,666,037 bytes, CRCs ok, sha256 e5817aa9...1232, DECISIONS.md T-010) and extracted to `data/construction-ppe/`; registry aliases (T-008) with tests.
- In progress: S1 data verification, statistics, test-split carve, unified copy.
- Next command: `.venv/Scripts/python.exe -m safety_ai.training.cli_cppe data all`
- Open problems: none.
