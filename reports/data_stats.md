# Construction-PPE data statistics

Ultralytics Construction-PPE, November 2026 release (CC BY 4.0), verified and remapped by `safety_ai.training.cppe_data` / `cppe_prepare` / `remap`. Produced by:

- `python -m safety_ai.training.cli_cppe data download`
- `python -m safety_ai.training.cli_cppe data verify`
- `python -m safety_ai.training.cli_cppe data prepare`

## Release facts versus the specification

The download URL serves the November 2026 release (DECISIONS.md T-001); the specification (4.1/4.2) described an earlier one. Verified facts of the release:

| item | specification 4.1/4.2 | downloaded release (verified) |
|---|---|---|
| images | 1,416 (1132 / 143 / 141) | 1,000 (800 / 200 / no test split) |
| classes | 11 incl. `none`, no `no_vest` | 12: `forklift` and `no_vest` added, no `none` |
| boxes | not stated | 14,863 |
| archive | about 178 MB | 647,666,037 bytes |
| licence | AGPL-3.0 | CC BY 4.0 |
| citation | Dalvi M., Singh N., Bhingarde S., Chalke K., v1.0.0, January 2025 | Ultralytics team, November 2026 (below) |
| ids | 0 helmet .. 5 none, 6 Person .. 10 no_boots | 0 helmet, 1 gloves, 2 vest, 3 boots, 4 goggles, 5 Person, 6 no_helmet, 7 no_goggle, 8 no_gloves, 9 no_boots, 10 forklift, 11 no_vest |

Citation (CC BY 4.0; credit the Ultralytics team and indicate changes):

```bibtex
@dataset{Ultralytics_Construction_PPE_2026,
    author = "{Ultralytics team}",
    title = {Ultralytics Construction-PPE Dataset},
    month = {November},
    year = {2026},
    license = {CC-BY-4.0},
    url = {https://docs.ultralytics.com/datasets/detect/construction-ppe},
    publisher = {Ultralytics}
}
```

Every published fact holds (800 / 200 images, 12 classes, 14,863 boxes).

## Acquisition

Method recorded in `reports/cppe/acquisition.json`: **present** (latest check: present). Archive `data/raw/cppe_t1/construction-ppe.zip`: 647666037 bytes, sha256 `e5817aa9c73f33e525d76ca6bc6af4e4eac9dcea9bcfef1cfd1a71c09fbc1232`, 647666037 bytes, 2004 members, all CRCs ok. See DECISIONS.md T-010 for the stalled downloads and the resumed curl that produced it.

## Verification (spec 4.4)

| split | images | label files | empty label files (backgrounds) | boxes |
|---|---:|---:|---:|---:|
| train | 800 | 800 | 0 | 11902 |
| val | 200 | 200 | 0 | 2961 |

- Label lines: every line parsed as `cls cx cy w h` with `cls` in 0..11, all values in [0, 1] and `w, h > 0`: no error.
- Images: every file opened with PIL `verify()` and then fully decoded: none corrupt or truncated.
- `manifest.jsonl` of the archive: 1000 rows; sha256 and annotation count of every image match.
- Warning only: 94 boxes reach past the image border by exactly 5e-6 (their six-decimal values sum to 1.000005), an export rounding artefact of boxes that touch the border; Ultralytics clips them, nothing is changed.

## Instances per class per split (release classes)

| id | class | train boxes | val boxes | total | share | train images | val images |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | helmet | 1282 | 294 | 1576 | 10.6 % | 613 | 146 |
| 1 | gloves | 1548 | 381 | 1929 | 13.0 % | 546 | 136 |
| 2 | vest | 1476 | 340 | 1816 | 12.2 % | 644 | 157 |
| 3 | boots | 2003 | 481 | 2484 | 16.7 % | 576 | 155 |
| 4 | goggles | 612 | 139 | 751 | 5.1 % | 437 | 102 |
| 5 | Person | 1962 | 481 | 2443 | 16.4 % | 791 | 200 |
| 6 | no_helmet | 634 | 179 | 813 | 5.5 % | 337 | 91 |
| 7 | no_goggle | 265 | 82 | 347 | 2.3 % | 227 | 56 |
| 8 | no_gloves | 839 | 203 | 1042 | 7.0 % | 340 | 94 |
| 9 | no_boots | 569 | 178 | 747 | 5.0 % | 244 | 71 |
| 10 | forklift | 275 | 73 | 348 | 2.3 % | 201 | 46 |
| 11 | no_vest | 437 | 130 | 567 | 3.8 % | 300 | 84 |

## Image sizes

| long side (px) | images | share |
|---|---:|---:|
| < 640 | 0 | 0.0 % |
| 640-1023 | 0 | 0.0 % |
| 1024-1279 | 2 | 0.2 % |
| 1280-1919 | 47 | 4.7 % |
| 1920-2559 | 951 | 95.1 % |
| >= 2560 | 0 | 0.0 % |

Orientation: landscape 577, portrait 282, square 141. Most common resolutions: 2048x1360 (361), 2048x1152 (141), 1360x2048 (141), 1152x2048 (141), 2048x2048 (141), 1536x1024 (42), 2048x1536 (26), 1448x1086 (5). Several resolutions occur exactly 141 times, consistent with the README's note that the release combines human-collected and generated images.

## Boxes per image

| boxes | images |
|---|---:|
| 0 | 0 |
| 1-5 | 13 |
| 6-10 | 473 |
| 11-20 | 290 |
| 21-30 | 154 |
| 31-50 | 67 |
| > 50 | 3 |

min / median / mean / max: 1.0 / 12.0 / 14.9 / 76.0

## Box size per class

Side = sqrt(box area); "at 640" scales the long image side to 640 px as the training letterbox does; a box is small when its side is < 32 px there (COCO's small).

| class | boxes | mean w x h (normalized) | mean w x h (px, native) | mean side at 640 | median side at 640 | small at 640 |
|---|---:|---:|---:|---:|---:|---:|
| helmet | 1576 | 0.079 x 0.067 | 130 x 108 | 37.1 | 27.3 | 52.2 % |
| gloves | 1929 | 0.059 x 0.071 | 99 x 115 | 32.9 | 32.1 | 49.8 % |
| vest | 1816 | 0.125 x 0.184 | 212 x 292 | 77.8 | 47.6 | 40.0 % |
| boots | 2484 | 0.071 x 0.067 | 118 x 108 | 34.7 | 33.0 | 49.0 % |
| goggles | 751 | 0.098 x 0.052 | 159 x 88 | 36.6 | 34.6 | 39.7 % |
| Person | 2443 | 0.180 x 0.447 | 302 x 724 | 145.6 | 88.9 | 27.2 % |
| no_helmet | 813 | 0.056 x 0.066 | 95 x 105 | 31.5 | 13.6 | 63.2 % |
| no_goggle | 347 | 0.042 x 0.020 | 69 x 32 | 14.7 | 13.8 | 98.3 % |
| no_gloves | 1042 | 0.044 x 0.050 | 72 x 80 | 23.4 | 19.5 | 70.7 % |
| no_boots | 747 | 0.088 x 0.058 | 143 x 95 | 35.9 | 34.9 | 44.4 % |
| forklift | 348 | 0.462 x 0.559 | 806 x 903 | 264.1 | 296.4 | 4.3 % |
| no_vest | 567 | 0.148 x 0.191 | 239 x 321 | 86.1 | 81.1 | 39.2 % |

Small-object classes (median side at 640 < 32 px): no_helmet, no_goggle, no_gloves, helmet. Borderline (32-40 px): gloves, boots, goggles, no_boots. The specification expected gloves, goggles and boots; on this release they sit just above the COCO limit at 640, while the bare-head, bare-face and bare-hand boxes are the smallest. Evaluation reports these classes separately (spec 7.1 c).

## Near-duplicates (pHash, Hamming distance <= 4)

1000 images fingerprinted (64-bit pHash of the grayscale image and of its mirror, `safety_ai.training.dedupe`); every pair compared. Exact duplicates (sha256): 0. Near-duplicate pairs: 0 (train-train 0, val-val 0, train-val 0).

No val image near-duplicates a train image, so no val image is excluded from the metric computation and no deduplicated val list is needed; the official splits are kept exactly (spec 4.4).

Closest pairs at any distance: train/source-efc23273c04e007083114c0f217335f0.jpg ~ val/source-9e10b0409f922546c35b276eb5a5756e.jpg at 6, train/ultralytics-ppe-nov2026-420.jpg ~ train/ultralytics-ppe-nov2026-556.jpg at 10, train/source-3978abe9cfed5d8a5faa5a98dfe4b00a.jpg ~ train/source-efc23273c04e007083114c0f217335f0.jpg at 12.

## Visual sanity check (24 random train images)

Annotated samples (all release classes, long side 640 px): `reports/samples/train/` (24 files, seed 0).

Inspection notes (2026-11-04, T1; under the deadline of DECISIONS.md T-019 only 3 of the 24 train
samples and 1 of the 12 no_vest samples were opened and looked at, the rest were generated but not
inspected):

- `00_train_ultralytics-ppe-nov2026-...jpg` (one of the release's generated images, file names
  `ultralytics-ppe-nov2026-*`): a worker beside orange forklifts. The Person box spans the whole
  body; helmet, goggles and vest sit exactly on the head, eyes and torso; both gloved hands and both
  boots have their own boxes; the two forklifts (one cut by the left border) are boxed. Class and
  box alignment is tight and correct.
- `14_train_source-903bd92aa69a20ccd.jpg` (a human-collected photo, file names `source-*`): a snowy
  construction site seen from a crane, five workers about 10-30 px tall at 640 px. Each worker
  still carries Person, helmet or no_helmet, vest, gloves or no_gloves and boots boxes; they are
  consistent with what can be seen, but most boxes are only a few pixels wide at the training
  input size: this is the small-object problem of `reports/data_stats.md` (box sizes per class).
- `no_vest/02_train_source-32b77e88696e2ba98.jpg` (only Person and no_vest boxes drawn): an atrium
  with about 25 small workers; no_vest boxes cover the torso of workers without a hi-vis vest,
  workers wearing yellow vests have none. This matches the numbers of the no_vest section (97 % of
  no_vest boxes inside a Person box, centred at 36 % of its height): a bare-torso label.
- No label noise was seen in these four images; the generated and the real images look different
  (close-up, clean composition versus wide shots with tiny people).

## Held-out test split (carved from train, DECISIONS.md T-002)

Method: greedy stratified carve of 100 images (seed 0, then 4000 seeded swap attempts) minimizing the squared relative deviation of every unified class's test instance count from 12.5 % of its train instances; pHash (Hamming <= 4, mirror included) clusters are atomic. Result: 100 test images, 700 train images; the file names are committed in `datasets/construction-ppe-split.json`. Train clusters: 800 (largest 1 image(s)). Near-duplicate pairs between the carved test split and the remaining train images: 0 (leakage-free by construction; asserted by `tests/unit/training/test_cppe_manifest.py` on the committed pHashes).

| unified class | train boxes | test boxes | train share | test share | val share |
|---|---:|---:|---:|---:|---:|
| person | 1713 | 249 | 17.1 % | 17.3 % | 17.0 % |
| head | 554 | 80 | 5.5 % | 5.6 % | 6.3 % |
| face | 232 | 33 | 2.3 % | 2.3 % | 2.9 % |
| hands | 734 | 105 | 7.3 % | 7.3 % | 7.2 % |
| foot | 498 | 71 | 5.0 % | 4.9 % | 6.3 % |
| helmet | 1121 | 161 | 11.2 % | 11.2 % | 10.4 % |
| vest | 1292 | 184 | 12.9 % | 12.8 % | 12.0 % |
| glasses | 535 | 77 | 5.3 % | 5.4 % | 4.9 % |
| gloves | 1354 | 194 | 13.5 % | 13.5 % | 13.5 % |
| boots | 1753 | 250 | 17.5 % | 17.4 % | 17.0 % |
| forklift | 241 | 34 | 2.4 % | 2.4 % | 2.6 % |

## Unified copy (`data/construction-ppe-unified/`, spec 5.1 and 5.3)

| release class | unified class | runtime meaning |
|---|---|---|
| 0 helmet | 5 helmet | PPE helmet PRESENT |
| 1 gloves | 8 gloves | PPE gloves PRESENT |
| 2 vest | 6 vest | PPE vest PRESENT |
| 3 boots | 9 boots | PPE boots PRESENT |
| 4 goggles | 7 glasses | PPE glasses PRESENT |
| 5 Person | 0 person | person box, tracked |
| 6 no_helmet | 1 head | bare head -> helmet MISSING |
| 7 no_goggle | 2 face | bare face -> glasses MISSING |
| 8 no_gloves | 3 hands | bare hands -> gloves MISSING |
| 9 no_boots | 4 foot | bare feet -> boots MISSING |
| 10 forklift | 10 forklift | forklift (vehicle proximity) |
| 11 no_vest | dropped | dropped (T-003) |

Images are hard links (linked 1000); every label file is rewritten with the unified ids, coordinates copied verbatim; `data.yaml` has relative paths and no `path` key. Dropped `no_vest` boxes: train 380, val 130, test 57.

| split | images | empty labels | boxes |
|---|---:|---:|---:|
| train | 700 | 0 | 10027 |
| val | 200 | 0 | 2831 |
| test | 100 | 0 | 1438 |

| unified class | source train | source val | unified train | unified val | unified test |
|---|---:|---:|---:|---:|---:|
| person | 1962 | 481 | 1713 | 481 | 249 |
| head | 634 | 179 | 554 | 179 | 80 |
| face | 265 | 82 | 232 | 82 | 33 |
| hands | 839 | 203 | 734 | 203 | 105 |
| foot | 569 | 178 | 498 | 178 | 71 |
| helmet | 1282 | 294 | 1121 | 294 | 161 |
| vest | 1476 | 340 | 1292 | 340 | 184 |
| glasses | 612 | 139 | 535 | 139 | 77 |
| gloves | 1548 | 381 | 1354 | 381 | 194 |
| boots | 2003 | 481 | 1753 | 481 | 250 |
| forklift | 275 | 73 | 241 | 73 | 34 |

Re-validation of the copy (spec 4.4 checks on the files on disk): no error; count identity (unified train + test = source train, unified val = source val, per class; no_vest absent): holds for every class. Asserted by `tests/unit/training/test_cppe_manifest.py`.

## Class imbalance and oversampling (spec 5.3)

Shares are of the unified train instances (700 images). A class below 5 % is repeated 2x, below 2 % 3x; an image with several minority classes takes the largest factor; val and test are never oversampled.

| class | train boxes | share before | factor | boxes in the list | share after |
|---|---:|---:|---:|---:|---:|
| person | 1713 | 17.1 % | 1 | 2685 | 16.7 % |
| head | 554 | 5.5 % | 1 | 941 | 5.9 % |
| face | 232 | 2.3 % | 2 | 464 | 2.9 % |
| hands | 734 | 7.3 % | 1 | 1294 | 8.1 % |
| foot | 498 | 5.0 % | 2 | 996 | 6.2 % |
| helmet | 1121 | 11.2 % | 1 | 1679 | 10.5 % |
| vest | 1292 | 12.9 % | 1 | 1976 | 12.3 % |
| glasses | 535 | 5.3 % | 1 | 825 | 5.1 % |
| gloves | 1354 | 13.5 % | 1 | 2092 | 13.0 % |
| boots | 1753 | 17.5 % | 1 | 2614 | 16.3 % |
| forklift | 241 | 2.4 % | 2 | 482 | 3.0 % |

`train_oversampled.txt`: 1142 entries for 700 train images; 442 images are repeated (minority classes: face 2x, foot 2x, forklift 2x). R2/R3 train on `data_oversampled.yaml`, which points at this list.

## The no_vest class (spec 5.2 applied to the release's extra class)

The specification's `none` class does not exist in this release; its inspection is applied to `no_vest` (DECISIONS.md T-003), the one release class the unified taxonomy has no target for.

| measure | value |
|---|---|
| boxes per split | train 437, val 130 |
| images containing it | 384 |
| mean size (normalized w x h) | 0.148 x 0.191 |
| mean size (px, native) | 239 x 321 |
| covered >= 90 % by one Person box | 97.0 % |
| overlapping no Person box (IoA < 10 %) | 0.0 % |
| median height / Person height | 0.40 |
| median width / Person width | 0.78 |
| median vertical centre in the Person box (0 top, 1 bottom) | 0.36 |
| Person boxes containing a no_vest box | 579 of 2443 |
| Person boxes containing a vest box | 1819 of 2443 |
| Person boxes containing both | 17 |

Annotated examples (only Person and no_vest boxes drawn): `reports/samples/no_vest/` (12 files).

**Decision (DECISIONS.md T-003, T-013): `no_vest` is dropped from the unified copy.** Its boxes are torso regions inside Person boxes (the evidence above), i.e. a bare-torso class. Neither the unified taxonomy nor the web contract has a torso class, and the runtime already decides vest MISSING from a tracked person without a vest, so the class carries no information the runtime can use; mapping it to `person` would teach a wrong person extent. The R1 baseline keeps it (the release's own 12 classes) so the baseline stays comparable with anyone training on the release.
