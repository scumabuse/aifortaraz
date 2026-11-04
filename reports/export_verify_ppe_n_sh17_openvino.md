# Export verification — ppe_n_sh17 (openvino)

**PASSED**: mean IoU 1.0000 >= 0.9 and identical class ids.

| item | value |
|---|---|
| verified (UTC) | 2026-11-04T06:41:43Z |
| reference | models/ppe_n_sh17.pt (PyTorch) |
| export | models/ppe_n_sh17_openvino_model |
| images | 20 (SH17 val split (data/raw/sh17/val_files.txt, 1620 images), seeded sample (seed 0)) |
| input | 640x640 square letterbox, batch 1, same for both models |
| confidence threshold | 0.25 |
| device | cpu |
| matching | per class, greedy by descending IoU, a pair needs IoU >= 0.5; leftover overlapping boxes with different class ids count as class mismatches |
| pass criteria | mean IoU of matched boxes >= 0.9 and no class mismatch |
| matched boxes | 122 |
| mean IoU (min) | 1.0000 (1.0000) |
| class mismatches | 0 |
| unmatched .pt boxes | 0 |
| unmatched export boxes | 0 |

Unmatched boxes are reported but do not fail the check: detections whose confidence is within rounding of the threshold can appear in one backend and not the other.

## Per image

| # | image | .pt boxes | export boxes | matched | mean IoU | min IoU | class mismatches | unmatched .pt | unmatched export |
|---|---|---|---|---|---|---|---|---|---|
| 1 | sh17/pexels-photo-13820151.jpg | 8 | 8 | 8 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 2 | sh17/pexels-photo-15200454.jpg | 6 | 6 | 6 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 3 | sh17/pexels-photo-2886006.jpg | 4 | 4 | 4 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 4 | sh17/pexels-photo-3817825.jpg | 3 | 3 | 3 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 5 | sh17/pexels-photo-3846145.jpg | 6 | 6 | 6 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 6 | sh17/pexels-photo-3931399.jpg | 7 | 7 | 7 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 7 | sh17/pexels-photo-4487421.jpg | 7 | 7 | 7 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 8 | sh17/pexels-photo-4559761.jpg | 6 | 6 | 6 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 9 | sh17/pexels-photo-4820685.jpg | 6 | 6 | 6 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 10 | sh17/pexels-photo-5432294.jpg | 5 | 5 | 5 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 11 | sh17/pexels-photo-5668873.jpg | 7 | 7 | 7 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 12 | sh17/pexels-photo-5974004.jpg | 3 | 3 | 3 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 13 | sh17/pexels-photo-7089630.jpg | 7 | 7 | 7 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 14 | sh17/pexels-photo-7388908.jpg | 5 | 5 | 5 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 15 | sh17/pexels-photo-7788367.jpg | 11 | 11 | 11 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 16 | sh17/pexels-photo-7883930.jpg | 9 | 9 | 9 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 17 | sh17/pexels-photo-8985605.jpg | 4 | 4 | 4 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 18 | sh17/pexels-photo-8986140.jpg | 2 | 2 | 2 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 19 | sh17/pexels-photo-9363129.jpg | 6 | 6 | 6 | 1.0000 | 1.0000 | 0 | 0 | 0 |
| 20 | sh17/pexels-photo-9762097.jpg | 10 | 10 | 10 | 1.0000 | 1.0000 | 0 | 0 | 0 |
