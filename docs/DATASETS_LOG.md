# Dataset download log

Every download attempt made by `safety-ai data download`, newest last (spec 6.2: record what was
used, with licences). "On disk" is the size of `data/raw/<source>/` after the attempt. This log
feeds docs/TRAINING.md.

| Date (UTC) | Source | URL | Result | Transferred | On disk | Items | Licence | Detail |
|---|---|---|---|---|---|---|---|---|
| 2026-11-03 23:06 | sh17 | https://www.kaggle.com/api/v1/datasets/download/mugheesahmad/sh17-dataset-for-ppe-detection | ok | 1.58 GB | 1.61 GB | 8099 | CC BY-NC-SA 4.0 | 8099/8099 images available (pexels 8098, archive fallback 1); dropped 0, failed 0 |
| 2026-11-03 23:23 | css | https://www.kaggle.com/api/v1/datasets/download/snehilsanyal/construction-site-safety-image-dataset-roboflow?datasetVersionNumber=3 | ok | 154.86 MB | 156.17 MB | 2801 | CC BY 4.0 | 5605 of the archive's members selected (2801 images); 5605 written now, 0 already present; archive 216.0 MB, fetched 154.9 MB |
| 2026-11-03 23:23 | harness | https://www.kaggle.com/api/v1/datasets/download/manoj223/safety-harness?datasetVersionNumber=1 | ok | 141.42 MB | 142.66 MB | 566 | MIT | 1133 of the archive's members selected (566 images); 1133 written now, 0 already present; archive 141.6 MB, fetched 141.4 MB |
| 2026-11-03 23:24 | urfall | https://fenix.ur.edu.pl/~mkepski/ds/uf.html | ok | 97.87 MB | 97.89 MB | 70 | CC BY-NC-SA 4.0 (non-commercial academic use; cite Kwolek & Kepski 2014) | 70 videos (30 falls, 40 ADL) |
| 2026-11-03 23:24 | chv | https://drive.google.com/file/d/1fdGn67W0B7ShpBDbbQpUF0ScPQa4DR0a/view | exists | 0 B | 448.30 MB | 1330 | free use with citation (Wang et al., Sensors 2021, 21(10):3478) | 1330 images extracted from CHV_dataset.zip (archive deleted) |
| 2026-11-03 23:25 | sh17 | https://www.kaggle.com/api/v1/datasets/download/mugheesahmad/sh17-dataset-for-ppe-detection | ok | 0 B | 1.61 GB | 8099 | CC BY-NC-SA 4.0 | 8099/8099 images available (pexels 8098, archive fallback 1); dropped 0, failed 0 |
| 2026-11-04 07:03 | hardhat | https://www.kaggle.com/api/v1/datasets/download/andrewmvd/hard-hat-detection?datasetVersionNumber=1 | ok | 243.95 MB | 1.32 GB | 5000 | CC0 1.0 | 10000 of the archive's members selected (5000 images); 933 written now, 9067 already present; archive 1314.2 MB, fetched 243.9 MB |
| 2026-11-04 07:06 | forklift | https://www.kaggle.com/api/v1/datasets/download/manoj223/forklift-dataset?datasetVersionNumber=1 | ok | 282.88 MB | 296.87 MB | 159 | MIT | labels and data.yaml for the images already fetched (bandwidth-limited subset); 319 of the archive's members selected (159 images); 233 written now, 86 already present; archive 9288.0 MB, fetched 282.9 MB |
| 2026-11-04 07:07 | forklift | https://www.kaggle.com/api/v1/datasets/download/manoj223/forklift-dataset?datasetVersionNumber=1 | ok | 166.08 MB | 393.37 MB | 330 | MIT | 661 of the archive's members selected (330 images); 356 written now, 305 already present; archive 9288.0 MB, fetched 166.1 MB |
