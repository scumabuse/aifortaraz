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
