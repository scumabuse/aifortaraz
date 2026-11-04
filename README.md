# Digital Safety Inspector — website + AI module

## How to run

```
make init     # once: writes the root .env (API key, URLs) and copies it into both repositories
make dev      # starts the website backend, the website frontend and the AI module; Ctrl+C stops all three
```

## Logins

| Login / password | Role |
|---|---|
| admin / admin | everything, including users and API keys |
| engineer / engineer | cameras, zones, incidents, reports, settings |
| master / master | incidents of the area «Прессовый участок» |
| viewer / viewer | read only |

## URLs

| What | URL |
|---|---|
| Website | http://localhost:5173 |
| API docs | http://localhost:5173/api/docs |
| Backend health | http://127.0.0.1:8000/health |
| AI video (MJPEG) | http://localhost:8001/stream/cam-1.mjpeg, http://localhost:8001/stream/cam-2.mjpeg |
| AI health / metrics | http://127.0.0.1:9100/health, http://127.0.0.1:9100/metrics |
