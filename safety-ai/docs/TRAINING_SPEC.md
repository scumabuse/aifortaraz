# PPE detector training specification (Construction-PPE)

Verbatim copy of the task specification this training work implements (received 2026-11-04).
Deviations forced by the data actually published at the download URL are recorded in DECISIONS.md (T-xxx).

2. PROMPT START — Mission and context
You are a senior computer-vision / ML engineer. Train, evaluate, export, document and integrate a production-quality PPE (personal protective equipment) detector on the Ultralytics Construction-PPE dataset for the “Digital Safety Inspector” platform — an industrial safety system where an AI module (safety-ai) watches camera streams, detects missing PPE and dangerous situations, and sends events to a web dashboard. The detector you produce is the ppe model of that module.
Context you must respect:
• The runtime consumes detections as normalized [x, y, w, h] boxes with class names from a unified taxonomy: person, head (bare), face (bare), hands (bare), foot (bare), ear (bare), helmet, vest, glasses, gloves, mask, boots, earmuffs, face_shield, harness, forklift, vehicle, tools. The PPE codes used by the web contract are helmet, vest, glasses, gloves, mask, boots, earmuffs, harness, face_shield. The runtime decides “PPE missing” from two kinds of evidence: the item is detected (PRESENT) or the bare body part is detected without the item (MISSING). Construction-PPE is valuable precisely because it labels the bare cases (no_helmet, no_gloves, no_boots, no_goggle).
• First check the working directory. If it contains the safety-ai repository (package safety_ai, models/registry.json, datasets/taxonomy.yaml, safety_ai/training/), work inside it: put training code under safety_ai/training/, register the model, update config and docs, and run its tests. If it does not exist, create a standalone project ppe-training/ with the structure of section 8.3 and the same conventions.
• The result is for a hackathon demo from a laptop: the final model must run in real time on CPU (nano/small variants, ONNX/OpenVINO exports) and be honest about its accuracy.
Deliverables (all mandatory): downloaded and verified dataset; the unified-taxonomy copy with mapping; training runs with logs and best weights; evaluation reports on val and test with per-class metrics, tuned per-class confidence thresholds and an error analysis; ONNX (and OpenVINO / TensorRT where possible) exports with parity checks and benchmarks; registry entry and integration (or the standalone inference script); a Colab/Kaggle training notebook; MODEL_CARD.md; docs; DECISIONS.md, PROGRESS.md, CHECKLIST.md, FINAL_REPORT.md.
Primary references (the dataset is also bundled with the ultralytics package, so nothing has to be downloaded by hand): documentation page https://docs.ultralytics.com/datasets/detect/construction-ppe , dataset YAML in the Ultralytics repository ultralytics/cfg/datasets/construction-ppe.yaml, archive https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip .
3. Autonomy and work protocol (binding)
You are running unattended. The user is away and will not answer until the work is finished.
• Never stop to ask a question, request confirmation or offer options. Never end your turn with “let me know”, “should I…” or a list of things for the user to do. Make the most reasonable decision, record it in DECISIONS.md (decision, reason, alternatives) and continue.
• No placeholders anywhere: no TODO, FIXME, pass bodies, NotImplementedError, skipped or xfail tests, “simplified” code paths. Everything in this specification is implemented and verified.
• Never reduce scope to finish sooner. When the primary way fails three times (download, install, CUDA), use the fallback defined for that step and continue.
• Training runs are the only long-running steps. While a run is in progress, do not idle: write the evaluation code, export code, tests, notebook and docs in parallel, and poll the run every 5 minutes (read results.csv, log the latest epoch, mAP and ETA to PROGRESS.md). Never kill a run before its time box unless it crashed; if it crashed, read the traceback, fix the cause, resume from last.pt.
• Time boxes (section 6) are hard limits enforced with the time= argument; everything else has no time limit — keep working until the Definition of Done is met.
• Use git from the first minute; commit after every stage; weights over 50 MB, datasets, runs and caches are git-ignored (make models restores weights from the registry).
• Keep PROGRESS.md current (every stage and at least every 30 minutes): done / in progress / exact next command / open problems. If your session is interrupted and restarted, your first action is to read PROGRESS.md, DECISIONS.md and CHECKLIST.md and continue from the first unfinished item — never start over, never retrain a finished run (resume from last.pt instead).
• Do not use the webcam or microphone. Do not install experiment trackers that need accounts (no wandb, comet, mlflow); disable Ultralytics sync and HUB (settings.update({"sync": False, "hub": False, "wandb": False, "comet": False, "mlflow": False})); local CSV logs and TensorBoard only.
• Verification is part of the work: run make check after every stage; before declaring completion run the Verification Protocol (section 9.3) until three consecutive passes have zero findings. Only then write FINAL_REPORT.md, set PROGRESS.md to DONE, commit, and stop.
• Be honest: report measured numbers only, with the exact command that produced them; name every fallback used.
4. Dataset: facts, acquisition, verification
4.1. Facts (verify them against the downloaded data and record any difference)
• Name: Ultralytics Construction-PPE. Object detection, Ultralytics YOLO label format. 1,416 images: 1,132 train, 143 val, 141 test. Archive about 178 MB.
• 11 classes with these ids: 0 helmet, 1 gloves, 2 vest, 3 boots, 4 goggles, 5 none, 6 Person, 7 no_helmet, 8 no_goggle, 9 no_gloves, 10 no_boots. Five worn-PPE classes, four missing-PPE classes, Person, and a generic none. There is no “no vest” class.
• Licence AGPL-3.0 (dataset published by Ultralytics). Citation: Dalvi M., Singh N., Bhingarde S., Chalke K., “Construction-PPE: Personal Protective Equipment Detection Dataset”, v1.0.0, January 2025, Ultralytics, https://docs.ultralytics.com/datasets/detect/construction-ppe/ . Put the citation and licence into MODEL_CARD.md and docs/TRAINING.md.
4.2. Dataset YAML (verbatim content of ultralytics/cfg/datasets/construction-ppe.yaml; write it to datasets/construction-ppe.yaml if the installed ultralytics does not ship it)
# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
# Construction-PPE dataset by Ultralytics
# Documentation: https://docs.ultralytics.com/datasets/detect/construction-ppe
# Example usage: yolo train data=construction-ppe.yaml

path: construction-ppe # dataset root dir
train: images/train # 1132 images
val: images/val # 143 images
test: images/test # 141 images

names:
  0: helmet
  1: gloves
  2: vest
  3: boots
  4: goggles
  5: none
  6: Person
  7: no_helmet
  8: no_goggle
  9: no_gloves
  10: no_boots

download: https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
4.3. Acquisition (in this order; stop at the first that works)
1. Point Ultralytics at the project: from ultralytics import settings; settings.update({"datasets_dir": "<project>/data", "runs_dir": "<project>/runs"}). Then from ultralytics.data.utils import check_det_dataset; check_det_dataset("construction-ppe.yaml") triggers the automatic download and extraction into data/construction-ppe/. If the installed ultralytics is too old to know this YAML, use the local copy from 4.2 (check_det_dataset("datasets/construction-ppe.yaml")); the download: key still works.
2. If the automatic download fails three times, download the archive directly with curl -L -o data/raw/construction-ppe.zip <url> and extract it to data/construction-ppe/.
3. If GitHub is unreachable, look for a user-provided data/raw/construction-ppe.zip and extract it. If nothing is available, keep working on everything that does not need the images (code, tests on a synthetic YOLO dataset of 200 images with the same 11 classes generated by training/synthetic.py, docs, notebook), record the blocker in DECISIONS.md and PROGRESS.md, and retry the download every 15 minutes until the end of the session.
Base weights: yolo26n.pt and yolo26s.pt (Ultralytics YOLO26); if they cannot be loaded with the installed version, use yolo11n.pt / yolo11s.pt; if downloads are blocked, use files from models/base/ when the user placed them there. Record which family was used.
4.4. Verification after download (safety-ai data validate or python -m ppe_training.data validate)
• Counts per split match 4.1 (images and label files; every image has a label file, empty label files are counted as background images and reported).
• Every label line is cls cx cy w h with cls in 0–10 and all coordinates in 0–1 with w, h > 0; corrupt or truncated images are detected with PIL verify() and listed.
• Per-class instance counts per split, image-size histogram, boxes-per-image histogram, mean box size per class (small objects: gloves, goggles, boots) → reports/data_stats.md with tables.
• Near-duplicate check across splits with perceptual hashing (Hamming distance ≤ 4): report the pairs; do not move images between the official splits (keep comparability with the published results), but exclude val/test images that duplicate a train image from the metric computation and say so in the report.
• Visual sanity: write 24 random annotated training images to reports/samples/train/ and look at the class/box alignment (describe what you see in the report: e.g. whether none boxes are meaningful).
• reports/data_stats.md ends with a decision on the none class (section 5.2).
5. Class mapping and dataset preparation
5.1. Mapping to the unified taxonomy
Construction-PPE class (id)
Unified class
Runtime meaning
helmet (0)
helmet
PPE code helmet PRESENT
gloves (1)
gloves
PPE code gloves PRESENT
vest (2)
vest
PPE code vest PRESENT
boots (3)
boots
PPE code boots PRESENT
goggles (4)
glasses
PPE code glasses PRESENT
none (5)
dropped (see 5.2)
—
Person (6)
person
person box, tracked
no_helmet (7)
head
bare head → helmet MISSING
no_goggle (8)
face
bare face → glasses MISSING
no_gloves (9)
hands
bare hands → gloves MISSING
no_boots (10)
foot
bare feet → boots MISSING
Unified ids for the training copy (10 classes, fixed order): 0 person, 1 head, 2 face, 3 hands, 4 foot, 5 helmet, 6 vest, 7 glasses, 8 gloves, 9 boots. Store the mapping as a construction_ppe entry in datasets/taxonomy.yaml (create the file in the standalone project) and ship models/<model>_class_map.json with the model so the runtime maps model class names to taxonomy names without code changes.
5.2. The none class
Inspect it before deciding: instance count per split, mean box size, where its boxes sit relative to Person boxes, and 12 annotated examples in reports/samples/none/. Default decision: drop it from the unified copy (it carries no PPE semantics for the runtime). Keep it in the baseline run on the original classes so the comparison with the published setting is fair. Write the decision and the evidence into DECISIONS.md and reports/data_stats.md.
5.3. Building the unified copy (data/construction-ppe-unified/)
• Copy or hard-link images; rewrite every label file with the new ids; drop dropped-class lines; keep the official train/val/test split exactly.
• Write data/construction-ppe-unified/data.yaml with the 10 names above and absolute-free relative paths.
• Re-run the validation of section 4.4 on the copy; instance counts per class must equal the source counts minus the dropped class — assert this in a test.
• Class imbalance: compute instance shares; for any class with < 5 % of all instances (expected candidates: boots, gloves, goggles and their bare counterparts — verify), build data/construction-ppe-unified/train_oversampled.txt listing train images with images containing those classes repeated 2× (3× when < 2 %). The production run trains on this list; val and test are never oversampled.
• Record in reports/data_stats.md: before/after class shares, number of repeated images, and the leakage pairs from section 4.4.
5.4. Optional extension (disabled by default)
Only after every item of the Definition of Done is ticked, and only if the environment variable EXTEND_WITH_SH17=1 is set, add the SH17 dataset (GitHub ahmadmughees/SH17dataset, CC BY-NC-SA 4.0) mapped into the same taxonomy (helmet, safety-vest → vest, glasses, gloves, face-mask-medical → mask, shoes → boots, earmuffs, face-guard → face_shield, person, head, face, hands, foot, ear; suits dropped) and train a second model ppe_merged_*, reported separately. Otherwise skip this section entirely and mention in FINAL_REPORT.md how to enable it.
6. Training plan
6.1. Environment detection (first thing after the dataset is verified)
Detect CUDA (torch.cuda.is_available(), GPU name, VRAM), MPS, CPU cores and RAM; write them to PROGRESS.md and reports/environment.md. Mode GPU when CUDA with ≥ 6 GB VRAM; mode CPU otherwise (MPS counts as CPU mode with device="mps" tried first and CPU as fallback if it errors). Use the Ultralytics Python API (from ultralytics import YOLO), never the shell CLI, so every run is a tested function in training/train.py with a typed config.
6.2. Runs (in this order; all with project="runs", exist_ok=True, seed=0, deterministic=True, plots=True, save_period=10, workers=min(8, cpu_count), amp=True on GPU)
Run
Purpose
Base
Data
Key settings
Time box
R0 smoke
prove the pipeline end to end
yolo26n.pt
unified
epochs=1, imgsz=320, fraction=0.1, batch=8
10 min
R1 baseline
reference numbers on the original 11 classes
yolo26n.pt
construction-ppe.yaml
epochs=60, imgsz=640, patience=20
GPU 0.75 h / CPU skipped
R2 production
the model to ship
yolo26s.pt (GPU) / yolo26n.pt (CPU)
unified, train_oversampled.txt
epochs=150, imgsz=640 (CPU: 512), patience=30, batch=-1 on GPU (auto) / 8 on CPU, cache="ram" if RAM ≥ 16 GB else "disk"
GPU 2.5 h / CPU 3 h
R3 variant
only if the GPU budget allows
yolo26s.pt with freeze=10, or yolo26m.pt
unified
same as R2
GPU 1.25 h / CPU skipped
Total GPU budget 5 hours; total CPU budget 3.5 hours. Enforce each box with time=<hours> (it overrides epochs). Choose the production model by val mAP50-95 (tie-break: mAP50 on the four bare classes head, face, hands, foot, because missing-PPE evidence is the product’s core). The test split is evaluated once, only for the chosen model, at the end.
6.3. Hyperparameters for R2 (also the defaults of training/train.py)
optimizer="auto", lr0 default, cos_lr=True, warmup_epochs=3, close_mosaic=15, hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, degrees=5, translate=0.1, scale=0.5, shear=2, fliplr=0.5, flipud=0.0, mosaic=1.0, mixup=0.1, copy_paste=0.0, erasing=0.3, label_smoothing=0.0, nbs=64. Keep hi-vis colours meaningful: do not add grayscale or strong hue augmentation beyond hsv_h=0.015. Resume: if runs/<name>/weights/last.pt exists and the run did not finish, YOLO(last_pt).train(resume=True).
6.4. During and after each run
• Poll results.csv every 5 minutes; log epoch, box/cls/dfl losses, mAP50, mAP50-95 and ETA to PROGRESS.md; detect divergence (NaN loss, mAP stuck at 0 after 10 epochs) and restart with lr0 halved and amp=False once, then record.
• After the run: copy best.pt to models/<name>.pt with the naming ppe_cppe_<variant>_<YYYYMMDD>.pt (variant n/s/m), save results.csv, results.png, confusion matrix and PR curves into reports/runs/<name>/, and append an entry to models/registry.json (name, role ppe, path, format pt, classes, imgsz, metrics val, trained_on construction-ppe-unified, base weights, epochs run, wall time, created_at, sha256).
• ppe_cppe_latest alias → the chosen production model.
6.5. CPU-only machines
Run R0 and the CPU variant of R2 only; write into docs/TRAINING.md and FINAL_REPORT.md the exact GPU command to reproduce R1–R3, plus the Colab path of section 6.6. State clearly which numbers come from the CPU-limited run.
6.6. Colab / Kaggle notebook (notebooks/train_colab.ipynb, mandatory)
Self-contained: installs ultralytics, downloads the dataset via the YAML, rebuilds the unified copy with the same mapping code (copied into the notebook as a cell, kept in sync by a test that diffs it against training/remap.py), runs R2 with a configurable time= cell, evaluates on val and test, exports ONNX, and zips best.pt, results.csv, plots and class_map.json into ppe_cppe_colab_<date>.zip for download. A final markdown cell explains how to register the zip locally (make register ZIP=...). Validate the notebook by executing it locally with jupyter nbconvert --execute in smoke mode (an environment variable SMOKE=1 sets fraction=0.05, epochs=1, imgsz=320).
7. Evaluation, threshold tuning, error analysis
7.1. Protocol
• Model selection and all tuning use the val split only. The test split is evaluated exactly once for the chosen production model and reported as the final number; never tune on it.
• Metrics via model.val(data=..., split="val"|"test", imgsz=640, conf=0.001, iou=0.6, plots=True): mAP50, mAP50-95, precision, recall overall and per class; confusion matrix; PR and F1 curves saved to reports/eval/<model>/.
• Report additionally: (a) the four bare classes (head, face, hands, foot) as a group — their recall is what makes “PPE missing” detectable; (b) pairwise confusion helmet↔head, glasses↔face, gloves↔hands, boots↔foot from the confusion matrix; (c) small-object classes (gloves, glasses, boots) with their mean box size; (d) speed in ms per image on the current device at imgsz 640.
• R1 (original 11 classes) is reported in its own table so the numbers are comparable with the published dataset page; R2/R3 in the unified table. Do not compare them to each other as if they were the same task.
7.2. Per-class confidence thresholds
For the production model sweep conf from 0.05 to 0.90 in steps of 0.05 on val and pick, per class, the value maximizing F1; clamp to [0.20, 0.70]; write models/<model>_thresholds.json ({"person": 0.35, "head": 0.30, ...}) and register it with the model. The runtime uses them as per-class conf overrides (global conf stays 0.25 as the floor for tracking). Report the chosen values and the F1 at each.
7.3. Error analysis (reports/eval/<model>/errors.md)
Run predictions on val at the tuned thresholds, match to ground truth at IoU 0.5, and dump: the 50 highest-confidence false positives and the 50 largest-area false negatives per class group (worn / bare / person) as annotated crops in reports/eval/<model>/errors/. Write what you actually see: small objects, occlusion, motion blur, label noise, duplicates, confusion between worn and bare classes, night scenes. Derive at most five concrete recommendations (e.g. imgsz 800 for gloves, more close-range data for boots, threshold changes) and state which of them you already applied.
7.4. Sanity inference
training/predict.py --weights models/ppe_cppe_latest.pt --source <12 test images> --out reports/samples/predict/ writes annotated images and a JSON per image with normalized [x, y, w, h] boxes, taxonomy class names and confidences — exactly the format the runtime consumes. Look at the images and describe three of them in the report.
7.5. Acceptance
There is no hard-coded accuracy pass mark for this dataset: the pass criteria are that every number in the report was produced by a logged command, that R2 beats R0 and (when run) R1-equivalent nano numbers are explained, that the four bare classes have non-trivial recall (if any of them has AP50 < 0.3, run the error analysis specifically on it and apply at least one remedy before finishing), and that the production model runs ≥ 8 FPS on the CPU of this machine via the ONNX or OpenVINO export at imgsz 640 (else at 512, documented).
8. Export, benchmark, registry, integration
8.1. Export and parity
Export the production model to ONNX (opset 17, imgsz 640, batch 1, simplify=True, dynamic=False) and, where the packages install, OpenVINO (FP32 and INT8 only if accuracy drop ≤ 1 mAP50 point on val) and TensorRT FP16 (CUDA only). Parity check for each export: run 20 val images through the .pt model and the export at the same thresholds; matched boxes (same class, IoU ≥ 0.5) must have mean IoU ≥ 0.9 and the same count ± 1 per image; write reports/export_parity.md. Register every export in models/registry.json as its own entry with format and parent.
8.2. Benchmark
safety-ai bench (or python -m ppe_training.bench): FPS, latency p50/p95 and peak RSS for each format on the current device over 200 frames of a 640×640 video; results to models/bench.json and a table in FINAL_REPORT.md. State which format the runtime should use on this machine.
8.3. Integration when the safety-ai repository exists
• Code lives in safety_ai/training/ (remap.py, train.py, evaluate.py, thresholds.py, errors.py, export.py, bench.py, predict.py, synthetic.py) and is reachable through the existing CLI (safety-ai data download|validate|remap, safety-ai train --run r2, safety-ai eval, safety-ai export, safety-ai bench).
• datasets/taxonomy.yaml gets the construction_ppe mapping; models/registry.json gets the entries; configs/default.yaml sets models.ppe: ppe_cppe_latest; the runtime loads <model>_class_map.json and <model>_thresholds.json from the registry entry (add this loading if it does not exist yet, with a unit test).
• Run the module’s own test suite (make check) and its end-to-end smoke with the new model; both must stay green. Run the 14 logic scenarios; they are model-independent and must stay green.
• Update docs/TRAINING.md (full recipe, commands, time boxes, Colab path), MODEL_CARD.md and README (one paragraph: which model ships, metrics, licence).
8.4. Standalone project when the repository does not exist (ppe-training/)
ppe-training/
  pyproject.toml  Makefile  requirements.lock  README.md  MODEL_CARD.md
  ppe_training/   data.py (download, validate, stats)  remap.py  train.py  evaluate.py  thresholds.py  errors.py  export.py  bench.py  predict.py  synthetic.py  registry.py
  datasets/construction-ppe.yaml  datasets/taxonomy.yaml
  models/registry.json  models/base/  (weights git-ignored)
  notebooks/train_colab.ipynb
  tests/  (remap correctness, label validation, synthetic smoke training, parity, predict output schema)
  reports/  runs/  data/  (git-ignored except reports)
  docs/TRAINING.md  DECISIONS.md  PROGRESS.md  CHECKLIST.md  FINAL_REPORT.md
predict.py must produce, per frame, {"frame_size": [w, h], "objects": [{"label": "<taxonomy name>", "bbox": [x, y, w, h], "conf": 0.93}]} with normalized coordinates — the format the safety-ai runtime and the website contract use — so the weights plus class_map.json plus thresholds.json can be dropped into the module later without changes. Write INTEGRATION.md with the three files to copy and the registry entry to add.
8.5. Venue adaptation recipe (document, do not run now)
capture 10–15 minutes of video at the presentation venue → sample a frame every 2 s → auto-label with the production model → correct labels in Label Studio or CVAT (30–60 minutes of human work) → fine-tune from ppe_cppe_latest.pt for 10 epochs at lr0 0.002 with freeze=10, mixing 20 % of the original train set to avoid forgetting → evaluate on the original val split (must not drop more than 2 mAP50 points) and on 50 held-out venue frames → register as ppe_cppe_venue_<date>. Give exact commands.
9. Stages, Definition of Done, Verification Protocol — PROMPT END
9.1. Stages (commit after each; update PROGRESS.md and CHECKLIST.md)
1. S0 Bootstrap: detect the repository or create the standalone project; environment report; git; DECISIONS/PROGRESS/CHECKLIST with every item of 9.2 unticked; make check skeleton.
2. S1 Data: download, validate, stats, samples, none decision, unified copy, oversampling list, tests for remap and validation.
3. S2 Smoke: R0 end to end (train → eval → export → predict) on the real data, or on the synthetic dataset if the download is still blocked.
4. S3 Training: R1 (GPU only), R2, R3 (GPU budget permitting), with polling, resume handling, registry entries.
5. S4 Evaluation: val and test reports, thresholds, error analysis, sanity inference, acceptance checks of 7.5.
6. S5 Export, parity, benchmark.
7. S6 Integration or standalone deliverables, notebook validated in smoke mode, docs, MODEL_CARD.md.
8. S7 Verification Protocol three clean passes; FINAL_REPORT.md; PROGRESS.md = DONE.
9.2. Definition of Done (every line in CHECKLIST.md with the command and key output as evidence)
[ ] Dataset downloaded and verified: 1,132 / 143 / 141 images with labels, all label lines valid, reports/data_stats.md written, 24 annotated samples inspected and described.
[ ] Unified copy built with the mapping of 5.1; instance counts match source minus dropped classes (asserted by a test); none decision recorded with evidence; oversampling list built and reported.
[ ] R0 smoke passed end to end; R2 completed within its time box (and R1/R3 on GPU); results.csv, plots and best.pt saved; registry entries valid JSON with sha256.
[ ] Val report with per-class metrics, bare-class group, pairwise confusions, speed; test report for the production model only; thresholds.json with F1-optimal per-class values.
[ ] Error analysis with 100 dumped examples and at most five recommendations, at least one applied.
[ ] predict.py outputs the contract-shaped JSON for 12 test images; images inspected.
[ ] ONNX export verified (parity ≥ 0.9 mean IoU, counts ± 1); OpenVINO / TensorRT exported where installable; bench.json written; CPU ≥ 8 FPS at 640 or at 512 with the reason.
[ ] Integration: safety-ai make check and e2e smoke green with models.ppe: ppe_cppe_latest (or, standalone: tests green and INTEGRATION.md complete).
[ ] notebooks/train_colab.ipynb executes in smoke mode via nbconvert; its remap cell is identical to remap.py (test).
[ ] docs/TRAINING.md, MODEL_CARD.md (dataset, licence, citation, metrics, intended use, limitations), README updated; every documented command executed during verification.
[ ] grep -rn "TODO\|FIXME\|NotImplementedError\|xfail\|skip(" <package> tests/ returns nothing.
[ ] DECISIONS.md complete; FINAL_REPORT.md written; PROGRESS.md says DONE with the final commit hash.
9.3. Verification Protocol (three consecutive clean passes required)
1. Fresh virtual environment from the lock file; make check (lint, types, unit tests, remap and validation tests, synthetic smoke training of 1 epoch at imgsz 320).
2. Re-run evaluate from the registered weights on val and compare with the numbers in the reports: mAP50 and mAP50-95 must match to three decimals (deterministic eval); any mismatch is a finding.
3. Re-run the export parity check and predict.py on the 12 images; compare JSON output with the stored one (identical boxes within 1e-3).
4. Review pass: re-read sections 4–8 of this specification bullet by bullet and check the code and reports implement each; write reports/review_pass_<n>.md; fix every finding.
5. Docs pass: execute every command in README.md, docs/TRAINING.md and INTEGRATION.md exactly as written.
6. Static pass: placeholder grep, --help of every command, registry JSON schema validation, licence and citation present.
7. Any finding → fix, commit, restart from step 1. A pass is clean only when steps 1–6 had zero findings. Record each pass in reports/verification_pass_<n>.md.
9.4. FINAL_REPORT.md
One page of what was built; environment used (GPU or CPU mode) and total training time; dataset summary and the none decision; table of all runs (base, data, epochs reached, wall time, val mAP50 / mAP50-95); the production model with per-class val and test metrics, bare-class group, thresholds; export and benchmark table; error-analysis conclusions; every fallback and deviation; known limitations (1,416 construction images, no “no vest” label, goggles only, small objects at distance, lighting); exact commands to run inference, to reproduce training on a GPU locally and in Colab, and to do the venue adaptation; next steps for the team. Then set PROGRESS.md to DONE, commit, stop.
PROMPT END.
