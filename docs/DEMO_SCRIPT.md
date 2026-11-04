# Live demo script (one laptop, about 12 minutes)

Setup: the product runs with `make dev` on the presenter's laptop, the browser on the projector
shows <http://localhost:5173> (admin / admin), cam-1 «Пресс №2» is the laptop webcam, cam-2
«Сварочный участок» plays its demo video, cam-3 and cam-4 run on mock data. The presenter needs a
helmet and a safety vest. Details of every step are in [INTEGRATION.md](INTEGRATION.md); when
something breaks, see the fallbacks at the end and [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Pre-flight

30 minutes before the talk (`make demo-preset` prints this list):

1. Laptop on mains power, heavy programs closed, webcam plugged in, helmet and vest at hand.
2. `make doctor` in the root directory: every row OK (Docker may show WARN).
3. cam-1 to the webcam: add the line `SAFETY_AI_CAMERAS__0__SOURCE=0` at the end of the root
   `.env` (removing it switches back to the demo video).
4. `make dev` and wait for the banner "Digital Safety Inspector is running"
   (already running: `make restart-ai`).
5. `make demo-preset` in a second terminal: "demo-preset: applied" or "already applied (no
   changes)"; confirm_frames 6, cooldown 20 s, demo mode on, camera cards on the AI video.
6. Telegram: the test message from `make demo-preset` arrived in the chat (needs
   `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_IDS` in `.env`); otherwise check
   Настройки > Уведомления > «Журнал отправок».
7. Projector: log in as admin / admin, open Live-мониторинг and cam-1
   (http://localhost:5173/live/cam-1): badge «AI», «Онлайн», boxes on your own image.
8. http://127.0.0.1:9100/health: cam-1 and cam-2 healthy, processed_fps close to 10.
9. Browser tabs ready: Дашборд, Live cam-1, Инциденты (filter «Внешний ИИ-модуль»),
   Камеры > «Пресс №2» > Зоны, Аналитика.
10. Walk through the zone once with and without the helmet; acknowledge the test incident.
11. Keep one terminal open in the root directory for the fallbacks; do not run `make e2e` or
    `safety-ai selftest --backend` from now on.

## On stage

| Time | Action | What the audience sees | What to say |
|---|---|---|---|
| 0:00 | Дашборд | KPIs (incidents, «Соответствие СИЗ», «Камер онлайн»), the latest incidents with «AI» and «демо» badges | four cameras of a press shop: two analysed by our AI module, two on the demo generator |
| 1:00 | Live-мониторинг → cam-1 | your webcam image with boxes, track id, ✓/✗/? per required item, the zone polygon, badge «AI» | the AI runs on this laptop; the site receives only metadata, the video goes straight from the AI to the browser |
| 2:00 | Stand in the zone wearing helmet and vest, then take the helmet off | the box turns from «Норма» to «Нарушение», the helmet item shows ✗ | the model sees a bare head where a helmet is required |
| 2:10 | Wait | the incident «Нарушение СИЗ» appears with sound and banner, severity «Высокая» (helmet in a danger zone) | expected within about 3 s of the bare head being visible: about 9 processed frames (the smoothed evidence crosses the threshold in 3 frames, then 6 confirming frames) ≈ 0.9 s at 10 fps, plus delivery to the site (target ≤ 2 s); scale the first part with the processed fps of `/health` |
| 2:30 | Show the phone / Telegram chat | the message with the snapshot and a link to the incident | notifications have filters, quiet hours and a per-camera cooldown |
| 3:00 | Put the helmet back on, open the incident | the incident page: snapshot with the boxes drawn by the site, missing «каска», present «жилет», start, end and duration once the helmet is back | the event is closed by the AI about 1.5 s after the helmet is back on |
| 4:00 | «Принять» | status «Принят», the timeline entry, the reaction time on the dashboard | one click for the shift master; «Ложное срабатывание» is next to it |
| 5:00 | Камеры → «Пресс №2» → Зоны → «Нарисовать зону» → click four points → Enter → name, «Обязательные СИЗ» (for example «Каска» and «Жилет») → «Сохранить» | «Зоны сохранены», «Версия» +1 | the AI receives config.updated at once and applies it within about 3 s |
| 6:00 | Back to Live cam-1, switch the site overlay «Зоны» off | the new polygon with its name is drawn by the AI into the video; «Объекты в кадре» lists the new zone for you | the zone travelled site → AI → video without a restart (version also in http://127.0.0.1:9100/health) |
| 7:00 | Optional: repeat the violation after 20 s | a second incident; Telegram stays silent if the first message is younger than 120 s | the same person can trigger again only after the cooldown (20 s in the preset) |
| 8:00 | Аналитика | trends, incidents by type and zone, compliance per camera from the AI's minute statistics | KPIs come from real AI statistics for AI cameras and from mocks for the others |
| 9:30 | Incident → «Составить акт» → «Сгенерировать описание» → «Предпросмотр» → «Печать / PDF» → «Сохранить как PDF» (optionally «Сохранить в историю») | the act about the violation with the snapshot, as PDF | paperwork in one minute |
| 11:00 | Questions | | privacy and false alarms: see below |

## What to say about privacy

- Video is processed locally on this laptop; nothing goes to a cloud. The website does not
  analyse or store video: the annotated stream goes from the AI module to the browser, the site
  keeps only metadata, the last detections in memory and one snapshot per incident.
- No identification: people are anonymous track numbers, there is no face recognition.
- Face blurring is one setting: `privacy.blur_faces: true` (or `SAFETY_AI_PRIVACY__BLUR_FACES=true`)
  blurs every head in the video and in the snapshots sent to the site.
- Snapshots are re-encoded without metadata and deleted after `snapshot_retention_days` (30); the
  AI's local event clips have a 2 GB cap and can be switched off (`output.save_event_clips`).
- Telegram is optional and receives only the incident text, snapshot and link.

## What to say about false-alarm control

- A violation needs evidence: "helmet missing" means a bare head is seen, not that the helmet is
  merely not found; when nothing is visible the item is unknown («?»), never a violation.
- Temporal confirmation with hysteresis: smoothed evidence plus `confirm_frames` consecutive
  frames (6 in the preset) to start, the same to end; persons below `min_confidence` are ignored.
- Rules per zone: required PPE, forbidden entry, max persons, vehicle distance, schedules,
  per-zone confirmation and cooldown, severity override.
- Cooldowns and deduplication on both sides: the AI does not re-raise the same person within the
  cooldown, the site deduplicates by event id and suppresses repeats.
- The model reports what it can detect (`supported_ppe`); the zone editor warns «не детектируется
  текущей моделью» instead of producing silent misses.
- A person decides: every incident has a snapshot with boxes and can be marked «Ложное
  срабатывание» with a reason, which feeds the analytics.

## Fallbacks

1. **AI crashes → the website continues in demo mode automatically.** A crashed camera worker is
   restarted by the AI's own supervisor; meanwhile cam-1 shows «ИИ-модуль офлайн» after 15 s and
   its «ДЕМО» mocks resume 60 s after the last heartbeat, with no action. Say "the site switched
   cam-1 to demo mode". If the whole AI process exits, `make dev` stops the website too (it treats
   the exit of any of its three processes as fatal): run
   `aifortaraz\safety-inspector\.venv\Scripts\python aifortaraz\safety-inspector\scripts\dev.py`
   to bring the website back alone in demo mode (mocks on all cameras), or `make dev` for
   everything.
2. **Website crashes → the AI alone with its preview window.** `make dev` stops the AI as well,
   which frees the webcam. Then:

   ```powershell
   cd safety-ai
   .venv\Scripts\safety-ai.exe demo --source 0 --zones tools/demo_zones_cam1.yaml --show
   ```

   A window shows the webcam with boxes and the zone (keys: q quit, z zones, b boxes, p pose,
   s screenshot); events are printed as JSON lines and saved to `safety-ai/events/`. The zones
   file requires helmet and vest and confirms after 8 frames with a 60 s cooldown. Add
   `--config configs/demo_integration.yaml` to use the show's model (without it the demo uses the
   default `ppe_s_latest`).
3. **Webcam fails → cam-1 back to the demo file with one config line.** Delete the line
   `SAFETY_AI_CAMERAS__0__SOURCE=0` from the root `.env` (or, if the YAML variant was used, put
   `source: data/samples/cam1.mp4` back in `safety-ai/configs/demo_integration.yaml`), then
   `make restart-ai`. cam-1 is «Офлайн» for the restart; the player retries every 5 s and shows
   the demo video with AI overlays again.
