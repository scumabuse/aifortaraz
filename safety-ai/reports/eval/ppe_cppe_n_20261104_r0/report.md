# Evaluation - ppe_cppe_n_20261104_r0 (val split)

| item | value |
|---|---|
| evaluated (UTC) | 2026-11-04T18:17:41Z |
| weights | `models/ppe_cppe_n_20261104_r0.pt` |
| dataset | `data/construction-ppe-unified/data.yaml`, split val (200 images, 2831 instances) |
| settings | imgsz 640, conf 0.001, NMS IoU 0.6 (spec 7.1) |
| device | cpu (Intel64 Family 6 Model 165 Stepping 3, GenuineIntel) |
| command | `python -m safety_ai.training.cli_cppe eval --model ppe_cppe_n_20261104_r0 --data data/construction-ppe-unified/data.yaml --split val --imgsz 640 --device cpu --speed-images 20` |

## Overall

| mAP50 | mAP50-95 | precision | recall |
|---|---|---|---|
| 0.0851 | 0.0520 | 0.2711 | 0.0783 |

Precision and recall are Ultralytics' values at the single confidence that maximizes the mean F1 over the classes.

## Per class

| class | taxonomy | images | instances | precision | recall | mAP50 | mAP50-95 | mean side at 640 (px) | small share |
|---|---|---|---|---|---|---|---|---|---|
| person | person | 200 | 481 | 0.944 | 0.807 | 0.906 | 0.560 | 145.8 | 25 % |
| head | head | 91 | 179 | 0.000 | 0.000 | 0.000 | 0.000 | 35.3 | 58 % |
| face | face | 56 | 82 | 0.000 | 0.000 | 0.000 | 0.000 | 15.1 | 99 % |
| hands | hands | 94 | 203 | 1.000 | 0.000 | 0.000 | 0.000 | 23.0 | 70 % |
| foot | foot | 71 | 178 | 0.000 | 0.000 | 0.000 | 0.000 | 32.7 | 52 % |
| helmet | helmet | 146 | 294 | 0.000 | 0.000 | 0.004 | 0.003 | 36.5 | 52 % |
| vest | vest | 157 | 340 | 0.000 | 0.000 | 0.006 | 0.002 | 78.0 | 41 % |
| glasses | glasses | 102 | 139 | 0.000 | 0.000 | 0.000 | 0.000 | 37.8 | 38 % |
| gloves | gloves | 136 | 381 | 1.000 | 0.000 | 0.000 | 0.000 | 32.4 | 52 % |
| boots | boots | 155 | 481 | 0.000 | 0.000 | 0.008 | 0.003 | 35.5 | 48 % |
| forklift | forklift | 46 | 73 | 0.038 | 0.055 | 0.012 | 0.004 | 230.7 | 4 % |

## Bare-class group (head, face, hands, foot)

| classes | instances | mean AP50 | mean AP50-95 | mean precision | mean recall | lowest AP50 |
|---|---|---|---|---|---|---|
| head, face, hands, foot | 642 | 0.000 | 0.000 | 0.250 | 0.000 | 0.000 |

Their recall is what makes "PPE missing" detectable (spec 7.1 a, 7.5: every bare class needs AP50 >= 0.3 for the production model).

## Worn versus bare confusions

From Ultralytics' confusion matrix (confidence 0.25, IoU 0.45, class-agnostic matching): per true class, how many boxes were found as the right class, as the paired class, or not at all.

| pair | true worn | worn found | worn -> bare | worn missed | true bare | bare found | bare -> worn | bare missed |
|---|---|---|---|---|---|---|---|---|
| helmet / head | 294 | 7 | 0 | 164 | 179 | 0 | 0 | 121 |
| glasses / face | 139 | 0 | 0 | 134 | 82 | 0 | 0 | 82 |
| gloves / hands | 381 | 0 | 2 | 182 | 203 | 0 | 0 | 132 |
| boots / foot | 481 | 147 | 1 | 289 | 178 | 0 | 79 | 89 |

## Small objects

Spec 7.1 c names gloves, glasses and boots; on this release the measured smallest classes (median label side < 32 px at 640) are helmet, head, face and hands.

| class | mean side at 640 (px) | small share | AP50 | AP50-95 | recall |
|---|---|---|---|---|---|
| gloves (gloves) | 32.4 | 52 % | 0.000 | 0.000 | 0.000 |
| glasses (glasses) | 37.8 | 38 % | 0.000 | 0.000 | 0.000 |
| boots (boots) | 35.5 | 48 % | 0.008 | 0.003 | 0.000 |
| helmet (helmet) | 36.5 | 52 % | 0.004 | 0.003 | 0.000 |
| head (head) | 35.3 | 58 % | 0.000 | 0.000 | 0.000 |
| face (face) | 15.1 | 99 % | 0.000 | 0.000 | 0.000 |
| hands (hands) | 23.0 | 70 % | 0.000 | 0.000 | 0.000 |

## Speed

On cpu (Intel64 Family 6 Model 165 Stepping 3, GenuineIntel) at imgsz 640:

| measure | ms per image |
|---|---|
| val preprocess | 1.85 |
| val inference | 68.53 |
| val loss | 0.00 |
| val postprocess | 12.75 |
| batch1 inference | 77.74 |
| batch1 total | 89.53 |

Batch-1 numbers are medians over 20 predictions after a warm-up of 3.

## Plots

- [confusion_matrix.png](confusion_matrix.png)
- [confusion_matrix_normalized.png](confusion_matrix_normalized.png)
- [BoxPR_curve.png](BoxPR_curve.png)
- [BoxF1_curve.png](BoxF1_curve.png)
- [BoxP_curve.png](BoxP_curve.png)
- [BoxR_curve.png](BoxR_curve.png)
