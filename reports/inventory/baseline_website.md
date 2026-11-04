# Website baseline (before any integration change)

Run 2026-11-04 ~19:50 on aifortaraz commit 83a018c (website code identical to 91e7b9b).

Commands (from safety-inspector/): `cd backend && ../.venv/Scripts/python -m pytest -q`, `npm --prefix frontend test`, `npm --prefix frontend run lint`, `npm --prefix frontend run build`, `.venv/Scripts/python -m ruff check backend scripts`.

```text
=== backend pytest
..........................................................               [100%]
============================== warnings summary ===============================
..\.venv\Lib\site-packages\starlette\testclient.py:51
  C:\Users\arkhi\Downloads\ocherednoiAIdlyahackathona\aifortaraz\safety-inspector\.venv\Lib\site-packages\starlette\testclient.py:51: DeprecationWarning: The anyio.abc.BlockingPortal alias is deprecated, use anyio.from_thread.BlockingPortal instead.
    _PortalFactoryType = Callable[[], AbstractContextManager[anyio.abc.BlockingPortal]]

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
58 passed, 1 warning in 13.14s
=== npm test
✔ zero avoided cost has no finite payback (0.3129ms)
✔ invalid counts and percentages are rejected (0.2422ms)
✔ 4:3 video is pillarboxed inside a 16:9 tile (1.2808ms)
✔ wide video is letterboxed inside a square tile (0.184ms)
✔ normalized bbox includes the black-bar offset (0.1681ms)
✔ invalid media dimensions cannot produce NaN coordinates (0.1297ms)
✔ detections fade after one second and disappear at three (0.1723ms)
ℹ tests 8
ℹ suites 0
ℹ pass 8
ℹ fail 0
ℹ cancelled 0
ℹ skipped 0
ℹ todo 0
ℹ duration_ms 183.7559
=== npm lint

> digital-safety-inspector@0.1.0 lint
> eslint .

=== npm build
dist/assets/index-BrlXND47.css          42.60 kB │ gzip:  10.09 kB
dist/assets/PlanMap-BtqWCt_U.js          2.75 kB │ gzip:   1.44 kB
dist/assets/Controls-DP6vGeWd.js         3.41 kB │ gzip:   1.67 kB
dist/assets/DashboardPage-BtHrrn1S.js    4.70 kB │ gzip:   1.94 kB
dist/assets/WallPage-Dpvqei4Y.js         5.79 kB │ gzip:   2.55 kB
dist/assets/AnalyticsPage-3MrcQ3jG.js    7.08 kB │ gzip:   2.71 kB
dist/assets/PlanPage-CzN-KzKc.js         7.68 kB │ gzip:   3.14 kB
dist/assets/AboutPage-CvVy7MX2.js        8.51 kB │ gzip:   3.29 kB
dist/assets/ReportsPage-9wxv-g2R.js      9.84 kB │ gzip:   3.47 kB
dist/assets/SettingsPage-C8uEIT-y.js    21.46 kB │ gzip:   6.44 kB
dist/assets/ZoneEditor-D88mNT6i.js     315.82 kB │ gzip:  98.35 kB
dist/assets/index-CA3Xibk7.js          342.13 kB │ gzip: 108.15 kB
dist/assets/LivePage-Dqw2QdeB.js       396.71 kB │ gzip: 126.81 kB
dist/assets/Charts-CS2Iy5iG.js         436.53 kB │ gzip: 117.34 kB
✓ built in 5.98s
=== ruff
All checks passed!
```
