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
