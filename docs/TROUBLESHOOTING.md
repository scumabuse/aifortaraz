# Troubleshooting

Start with `make doctor`: it checks versions, `.env` and its copies, the ports, the AI key against
the website, the model weights and alias, and the demo videos, and names the failing piece. The
commands below are for PowerShell on Windows, run from the root directory; `curl.exe` is the real
curl (in Windows PowerShell plain `curl` is an alias of `Invoke-WebRequest`).

## Diagnostic commands

```powershell
make doctor
# the AI key from the root .env
$key = (Select-String -Path .env -Pattern '^SAFETY_AI_API_KEY=(.*)$').Matches[0].Groups[1].Value
# website up, number of AI modules connected over /ws/ingest
curl.exe -s http://127.0.0.1:8000/health
# the configuration the AI receives (401 = key rejected)
curl.exe -s -i -H "X-API-Key: $key" http://127.0.0.1:8000/api/v1/ingest/config/cam-1
# AI health: per camera stream_state, processed_fps, config_version, backend_unauthorized, error
curl.exe -s http://127.0.0.1:9100/health
# AI video reachable: prints the response headers (multipart/x-mixed-replace), stops after 3 s
curl.exe -s -D - -o NUL --max-time 3 http://localhost:8001/stream/cam-1.mjpeg
# an admin token, then which cameras run mocks and which are real
$token = (Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/v1/auth/login -ContentType 'application/json' -Body '{"username":"admin","password":"admin"}').access_token
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/demo -Headers @{Authorization = "Bearer $token"}
# camera status as the website sees it
Invoke-RestMethod -Uri http://127.0.0.1:8000/api/v1/cameras/cam-1/status -Headers @{Authorization = "Bearer $token"}
# recent AI problems
Select-String -Path safety-ai\logs\safety-ai*.log -Pattern '"level": *"(ERROR|WARNING)"' | Select-Object -Last 20
Get-ChildItem safety-ai\logs\rejected -ErrorAction SilentlyContinue | Select-Object -Last 5
# who holds a port
Get-NetTCPConnection -State Listen -LocalPort 8000,5173,8001,9100 -ErrorAction SilentlyContinue
```

Under Docker use `localhost` instead of `127.0.0.1` from the host and `make logs` for the logs.
`safety-ai selftest --backend http://127.0.0.1:8000` (from `safety-ai/`, key from
`SAFETY_AI_API_KEY`) sends one message of every type to the real website and fails on any 422 —
it creates test incidents and marks the first camera as real for a minute, so do not run it
during a presentation.

## Symptom → cause → fix

| Symptom | Cause | Diagnose | Fix |
|---|---|---|---|
| Camera stays «Офлайн» / «ИИ-модуль офлайн» on the Live page | Wrong or rotated API key: every heartbeat gets 401 | AI log: `backend rejected the API key (HTTP 401) for POST /api/v1/ingest/heartbeat — check SAFETY_AI_API_KEY / the website's AI key`; `/health` `backend_unauthorized: true`; the curl of the config above answers 401 | `make init` (copies the same key to both sides), restart (`make dev` or both terminals); the website re-seeds `ai-module` from `SEED_AI_API_KEY` on start. The AI retries every 10 s and keeps its events spooled, so it recovers by itself once the key matches |
| Camera stays offline | Wrong backend URL in the AI (`SAFETY_AI_BACKEND_URL`), the backend not running, or the AI pointed at the mock on 8010 | AI log `sending POST … failed (connection error …)`; `/health` `extra.backend.reachable: false` | `SAFETY_AI_BACKEND_URL=http://127.0.0.1:8000` on the host (`http://backend:8000` inside compose), `make init`, `make restart-ai` |
| Camera stays offline | AI and website on different machines: firewall or a 127.0.0.1-only backend | `curl.exe http://<website-host>:8000/health` from the AI machine | the website backend binds 127.0.0.1 in `make dev`; for two machines run the website with Docker (0.0.0.0:8000) or behind a proxy, open the port in Windows Defender Firewall, set `SAFETY_AI_BACKEND_URL` to that host |
| Camera stays offline | Camera ids differ: the AI runs `cam-5`, the website card has a UUID | AI log 404 / `Camera not found`; `/health` lists the AI's ids; Камеры shows the card id («Скопировать camera_id») | use the same id on both sides (DECISIONS.md I-007; INTEGRATION.md section 8) |
| Camera shows «ИИ-модуль офлайн» although `/health` is green | The browser's clock differs from the server's by more than 15 s (the Live page compares `last_heartbeat_at` with the browser time) | compare the clocks of both machines | synchronise the clock (Windows: Settings → Time & language → Date & time → Sync now), reload |
| 401 on ingest (`{"error": {"code": "unauthorized", "message": "API-ключ отсутствует, недействителен или отозван"}}`) | Missing `X-API-Key`, a key from another database, a revoked key, or `.env` copies out of sync | `make doctor` (row ".env (single source, copies in sync)" and the key row); `python scripts/init_env.py --check` | `make init`; restart the website so the seed applies `SEED_AI_API_KEY`; for `/ws/ingest` the key goes in `?api_key=` (a wrong key closes the socket before accept) |
| 422 with field errors | The payload breaks the contract; `details.fields[].location` names the field, e.g. `["body", "details", "frame_size"]`, `["body", "snapshot"]`, `["body", "external_id"]` | AI: `safety-ai/logs/rejected/<utc>_<external_id>_post.json` holds the payload, the status and the website's response body; AI log `POST … rejected: …; response body: …`. Website: `reports/runtime/backend.log` | the side whose output does not match contract 4.1 is wrong: if the field and value follow the contract, the website validation is at fault, otherwise the AI producer. `make e2e PYTEST_ARGS="-k schema_parity"` compares both sides' models; `422 Зона не найдена или принадлежит другой камере` means a zone was deleted after the AI read it (wait for the config refresh) |
| No overlays but the video plays | `/ws/live` not connected or not subscribed (header shows «Переподключение…» / «Нет связи»; an expired session closes the socket) | browser devtools → Network → WS `/ws/live?token=…`: a `subscribe` with `cam-1` must be sent and `detections` must arrive | reload the page or log in again; under Vite the `/ws` proxy needs the backend on :8000 |
| No overlays but the video plays | Camera id mismatch: the AI's detections are refused | AI `/health` `extra.websocket.server_errors` > 0; AI log `Invalid detections: … Camera not found` | same ids on both sides (section 8 of INTEGRATION.md) |
| No overlays but the video plays | CORS: the page is served from an origin not in `CORS_ORIGINS` and calls the backend directly | browser console `blocked by CORS policy` | open the site through Vite (<http://localhost:5173>) or nginx (<http://localhost:8080>), or add the origin to `CORS_ORIGINS`, `make init`, restart |
| No overlays but the video plays | Overlay toggles off, or the AI sends nothing for the camera (`backend.ws: false`) | Live toggles «Рамки», «Зоны», «Только нарушения»; `/health` `extra.websocket.connected` | switch the toggles on; keep `backend.ws: true` |
| Video black: «Нет сигнала» / «Повторная попытка через 5 секунд» | The MJPEG URL is not reachable from the browser: the card points at another host or port | `curl.exe -i http://localhost:8001/stream/cam-1.mjpeg`; Камеры → cam-1 → «Адрес потока» | `AI_MJPEG_BASE_URL` must be the address the browser can reach (`http://localhost:8001` on one machine), then `make demo-preset` (the seed never edits existing cards) |
| Video black | `output.expose: false` while the browser is on another machine (servers bound to 127.0.0.1) | `/health` `extra.bind` | `output.expose: true` (demo config) or `safety-ai run --expose`; open TCP 8001 in the firewall |
| Video black | Port 8001 busy (another AI, a `safety-ai demo --mjpeg`, another program) | `make doctor` row "port 8001"; `Get-NetTCPConnection -LocalPort 8001` | stop the other program; one AI per machine |
| Video black, `/health` shows `stream_state` `failed` / `ended` | Missing demo file, webcam busy or absent, or the website `stream_url` points at the AI's own MJPEG for a camera without `source:` | `/health` `cameras[].error`; `make doctor` demo video rows | `make videos`; free the webcam; give the camera a local `source:` |
| «Проверить поток» says «Превью недоступно…» although the video plays | The check runs on the backend, not in the browser; inside Docker `localhost:8001` is the backend container | — | expected under Docker; the browser check is what matters |
| Mocks and real data overlap («ДЕМО» on an AI camera, mock incidents on cam-1) | No real AI signal for the camera in the last 60 s (AI stopped, key rejected, camera failed) — mocks resume by design | `GET /api/v1/demo` → `real_camera_ids`; `/health` | fix the AI side; mocks stop within one generator tick (≤ 2 s) after the next real heartbeat |
| Mock incidents of cam-1 in the journal | The 30-day mock history (360 incidents) and mock incidents from before the AI came online stay in the journal | source column «демо» | filter «Внешний ИИ-модуль», or Настройки → Демо-режим → «Сбросить и пересоздать демо-данные» (real AI records are kept) |
| `config.updated` not received (a saved zone does not reach the AI) | The AI's WebSocket is down (key, URL, backend restarted) | `/health` `extra.websocket.connected`; `config_version` of the camera vs `version` of `GET /api/v1/ingest/config/cam-1` | the AI also re-reads every 60 s and on every reconnect, so it converges within a minute; fix the WebSocket cause; `make restart-ai` reloads at once |
| `config.updated` received but nothing changes | The zone is disabled, outside its schedule (camera time `TZ`), or requires an item the model cannot detect («не детектируется текущей моделью») | zone editor; `cameras[].timezone` in the AI config | enable the zone, fix the schedule or `TZ`, choose PPE the model supports |
| Duplicates in the incident list | Two AI modules for the same camera (`make dev` plus a manual `safety-ai run`, or Docker plus host) | website `/health` `connected_ai_modules` > 1; `Get-NetTCPConnection -LocalPort 8001` | run one AI per camera |
| Duplicates in the incident list | The tracker lost the person and gave a new track id: cooldown and deduplication are per track | `track_id` of the incidents differ | expected; reduce occlusion, keep the person in frame; repeated `external_id`s are always answered 200 with the same incident |
| Duplicates in the incident list | `safety-ai selftest --backend …` or «Симулировать инцидент» was used | source and type of the incidents | mark them «Ложное срабатывание» |
| Timezone off by hours | The website shows times in its setting «Часовой пояс IANA» (Настройки → Общие настройки), not in the browser's zone | the journal shows «Часовой пояс: …» | set the plant zone there (`Asia/Yekaterinburg` by default) |
| Schedules switch at the wrong hour | AI camera timezone from `TZ` (`cameras[].timezone`) | `TZ` in `.env` | set `TZ`, `make init`, `make restart-ai` |
| Event times wrong by hours | A client sent local time with `Z`; or the machine clock is wrong (the AI sends UTC with `Z`, the website rejects timestamps without a zone with 422) | `make e2e PYTEST_ARGS="-k timestamps"` | fix the system clock or the client |
| Slow laptop: `processed_fps` far below 10, late incidents, choppy video | CPU or GPU too slow for two cameras at `imgsz` 640 | `/health` `processed_fps`; Task Manager | lower `processing.target_fps` (e.g. `SAFETY_AI_PROCESSING__TARGET_FPS=6`) or `models.imgsz` (`SAFETY_AI_MODELS__IMGSZ=480`) in `.env`, or disable cam-2 (`SAFETY_AI_CAMERAS__1__ENABLED=false`), then `make restart-ai`; confirmation time is about 9 processed frames, so it grows as fps drops |

## Docker on Windows

Docker is not installed on the integration machine (DECISIONS.md I-004); these notes come from
reviewing the compose file, not from a run.

| Symptom | Cause | Fix |
|---|---|---|
| Containers start but the website is empty, or `database is locked` / disk I/O errors | The SQLite database (WAL) lives on a bind mount from the Windows file system (`./aifortaraz/safety-inspector/data:/app/data`), where file locking is unreliable and slow | keep the checkout inside the WSL 2 file system (`\\wsl$\…`) or run the backend with `make dev`; do not run `make dev` and `make up` on the same database at the same time |
| Bind mounts empty (`models`, `data/samples`) | The drive is not shared with Docker Desktop, or the path contains characters Docker cannot mount | Docker Desktop → Settings → Resources → File sharing; a checkout path without spaces |
| `exec … safety-ai-entrypoint: no such file or directory` | Shell scripts checked out with CRLF | both repositories set `eol=lf` in `.gitattributes`; re-checkout with `git config core.autocrlf false` |
| AI container unhealthy for minutes | First start downloads missing weights (`make models` in the entrypoint), or `safety-ai/data/samples` is empty | `make videos` on the host before `make up`; give the healthcheck its 120 s start period |
| Changes to `configs/demo_integration.yaml` have no effect | The configuration is copied into the image | `docker compose -p inspector up -d --build ai` |
| Webcam source fails in the container | Docker Desktop does not pass a USB webcam into Linux containers | use `make dev` for the webcam |
| Telegram links open the wrong port | `make demo-preset` stores `http://localhost:5173` as the site URL | under Docker set the base URL to `http://localhost:8080` in Настройки → Уведомления |
