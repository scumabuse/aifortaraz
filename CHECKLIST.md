# Checklist — Definition of Done (integration spec 10.1)

Each item is ticked only with the command that proves it and the key lines of its output.

- [ ] INTEGRATION_INVENTORY.md complete: website and AI mapped with file paths, baseline test results of both repos, real responses captured, conformance matrix with ≥ 60 rows and every row OK or fixed (commit hash in the last column), duplicate code resolved and archived.
- [ ] contract/CONTRACT.md and both repositories’ docs/CONTRACT.md byte-identical; schema parity test green; the ten malformed payloads rejected with 422.
- [ ] Both repositories’ own make check green after all integration changes, with new tests for every fix.
- [ ] make init + make doctor green on this machine; .env is the single source of the API key and URLs.
- [ ] make dev starts backend, frontend and AI, prints URLs and logins, and Ctrl+C stops all three cleanly; make up does the same with Docker (if Docker is unavailable here, compose files are complete and reviewed, and this is recorded).
- [ ] Seeded cameras cam-1/cam-2 are AI-enabled with MJPEG URLs and zones; GET /ingest/config/cam-1 matches section 4.4 exactly.
- [ ] All flow tests of 8.2 green; chaos tests green; mock switching verified both ways.
- [ ] Real-model smoke of 8.3 passed: 120 s with zero ERROR lines, overlays visible in reports/screenshots/live_cam1.png, incidents page screenshot shows at least one AI-sourced incident or, when the synthetic video produces none, the fake-scenario incident with the “AI” badge; screenshots inspected and described.
- [ ] Performance budget measured and recorded; demo config tuned to this machine.
- [ ] make demo-preset idempotent and verified twice in a row.
- [ ] docs/INTEGRATION.md, docs/TROUBLESHOOTING.md, docs/DEMO_SCRIPT.md, root README written; every documented command executed during verification.
- [ ] grep -rn "TODO\|FIXME\|NotImplementedError\|xfail\|skip(" over integration-tests/, scripts/ and every file touched in the two repositories returns nothing.
- [ ] Commits: integration:-prefixed commits in both repositories, root repository committed, PROGRESS.md = DONE with the three commit hashes, FINAL_REPORT.md complete.
