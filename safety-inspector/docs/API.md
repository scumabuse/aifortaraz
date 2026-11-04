# API — этап 3

Префикс REST: `/api/v1`; Swagger: `/api/docs`; OpenAPI: `/api/openapi.json`.
Swagger UI поставляется локально. Контракт интеграции с модулем ИИ — единый источник
истины — в [CONTRACT.md](CONTRACT.md) (побайтная копия contract/CONTRACT.md корня интеграции).

## Авторизация и ошибки

Браузер: `Authorization: Bearer <access_token>`. ИИ: `X-API-Key: <ключ>`.
Каналы независимы: JWT не заменяет API-ключ и наоборот.

- POST `/auth/login` — JSON `{ "username": "admin", "password": "admin" }`.
- POST `/auth/refresh` — JSON `{ "refresh_token": "..." }`.
- GET `/auth/me` — текущий пользователь без password_hash.
- GET `/users` — список пользователей, только admin.

Login/refresh возвращают access_token, refresh_token, token_type=bearer и user
(id, username, full_name, role, area_id). Время жизни по умолчанию 30 минут / 7 дней.
Ошибки 401 / 403 / 404 / 409 / 422:

```json
{"error":{"code":"unauthorized","message":"API-ключ отсутствует, недействителен или отозван","details":{}}}
```

## Камеры

Чтение — все вошедшие пользователи. Изменение — admin, engineer.

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/cameras` | массив камер с участком, статусом, FPS, моделью, числом зон и событий за 7 дней |
| POST | `/cameras` | создание, 201 |
| GET / PUT / DELETE | `/cameras/{id}` | просмотр / полная замена полей / удаление |
| GET | `/cameras/{id}/status` | статус по последнему heartbeat |
| POST | `/cameras/{id}/check-stream` | ограниченная по времени проверка HTTP(S)-превью/MJPEG |
| GET | `/cameras/{id}/config` | JSON ingest/config для редактора зон: дни расписания в нумерации редактора (Пн=0) и supported_ppe |

Тело POST/PUT:

```json
{
  "name": "Пресс №2",
  "area_id": "<UUID участка>",
  "stream_type": "rtsp",
  "stream_url": "rtsp://192.168.1.10/live",
  "preview_url": "http://192.168.1.10/frame.jpg",
  "plan_x": 0.3,
  "plan_y": 0.7,
  "description": "Прессовый участок",
  "enabled": true,
  "ai_enabled": true
}
```

stream_type: rtsp / mjpeg / hls / file / webcam. Координаты 0…1 или null.
Камера в ответах GET содержит supported_ppe — коды СИЗ из extra.supported_ppe
последнего heartbeat ИИ (null, если ИИ их не сообщал).
Для webcam stream_url может быть пустым. Для file допустим путь `/demo/cam1.mp4`.
Удаление камеры с инцидентами запрещено (409); без инцидентов удаляются также
зоны, статус и compliance_stats. Проверка потока возвращает `{ok,message}`;
успех превью не означает, что ИИ онлайн. Декодирования видео нет.

## Зоны и правила

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/cameras/{id}/zones` | все зоны, включая отключённые |
| PUT | `/cameras/{id}/zones` | атомарная замена массива зон |
| POST | `/zones` | создание, тело зоны + camera_id, 201 |
| PUT / DELETE | `/zones/{id}` | изменение / удаление |
| GET | `/zone-responsibles` | id и full_name для выбора ответственного, admin/engineer |

PUT массива сохраняет существующие UUID, создаёт новые и удаляет отсутствующие.
UUID из другой камеры отклоняется; удаление зоны с инцидентами запрещено (409).
Невалидный элемент откатывает всю замену. Максимум 100 зон и 100 вершин на зону.
POST/PUT зон доступны admin/engineer, GET доступен всем вошедшим.

```json
{
  "id": "<UUID, необязательно для новой зоны>",
  "name": "Пресс №2",
  "color": "#eab54d",
  "zone_type": "danger",
  "polygon": [[0.1,0.2],[0.55,0.18],[0.6,0.8],[0.12,0.85]],
  "enabled": true,
  "sort_order": 0,
  "rules": {
    "required_ppe": ["helmet","glasses"],
    "entry_forbidden": false,
    "max_persons": 2,
    "min_vehicle_distance_m": null,
    "schedule": [{"days":[0,1,2,3,4],"from":"08:00","to":"20:00"}],
    "confirm_frames": null,
    "cooldown_sec": null,
    "severity_override": null,
    "regulation_ref": "п. 4.2 ИОТ-12",
    "responsible_user_id": null
  }
}
```

В polygon первая точка не дублируется в конце; минимум 3 различные вершины,
ненулевая площадь, без самопересечений, координаты 0…1. zone_type:
ppe / danger / vehicle / counting. schedule=null — круглосуточно; дни Пн=0…Вс=6,
время HH:mm в часовом поясе предприятия. Интервалы через полночь допустимы.
ИИ получает дни по контракту 4.4: 1…7, Пн=1 — преобразование выполняется только
в GET /ingest/config.
confirm_frames: 1…10000, cooldown_sec: 0…86400; null наследуется из settings.
max_persons: 1…10000 или null; дистанция: (0,1000] или null.
regulation_ref и responsible_user_id сохраняются в правилах браузерного API;
они не добавляются в контракт ИИ, чтобы сохранить точную структуру раздела 6.2.

## Конфигурация ИИ — точный формат раздела 6.2

GET `/ingest/config/{camera_id}` — один объект; GET `/ingest/config` — массив
таких объектов для камер с ai_enabled=true. Обязателен действующий X-API-Key.
Включение камеры enabled и флаг ai_enabled — разные поля: коллекция фильтруется
по ai_enabled, как указано в контракте. Отключённые зоны не выдаются ИИ.
Дни расписания здесь 1…7 (Пн=1), как в контракте 4.4.

```json
{
  "camera": {
    "id": "<UUID камеры>",
    "name": "Пресс №2",
    "stream_url": "rtsp://192.168.1.10/live",
    "stream_type": "rtsp"
  },
  "frame_size_hint": [1280,720],
  "thresholds": {"confirm_frames":8,"cooldown_sec":60,"min_confidence":0.5},
  "ppe_types": ["helmet","vest","glasses","gloves","mask","boots","earmuffs","harness","face_shield"],
  "zones": [
    {
      "id": "<UUID зоны>",
      "name": "Пресс №2",
      "type": "danger",
      "polygon": [[0.1,0.2],[0.55,0.18],[0.6,0.8],[0.12,0.85]],
      "rules": {
        "required_ppe": ["helmet","glasses"],
        "entry_forbidden": false,
        "max_persons": 2,
        "min_vehicle_distance_m": null,
        "schedule": null,
        "confirm_frames": 8,
        "cooldown_sec": 60,
        "severity_override": null
      }
    }
  ],
  "version": 42,
  "updated_at": "2026-11-03T12:00:00Z"
}
```

frame_size_hint — подсказка 1280×720, не результат анализа кадра. ИИ должен
использовать нормализованные координаты относительно своего реального кадра.
version — глобальный монотонный номер конфигурации. Он повышается на каждое
сохранение зон/порогов и изменение камер, updated_at — время последнего сохранения
в UTC. confirm_frames/cooldown_sec зон наследуются до сериализации ответа;
явный cooldown_sec=0 сохраняется. Цвет и поле enabled не добавляются в config.

GET `/settings` — просмотр текущих настроек; GET `/settings/ppe-types`,
`/settings/incident-types`, `/settings/areas` — справочники. PUT `/settings` в этапе 2
сохраняет только три порога (admin/engineer), JSON:
`{"confirm_frames":8,"cooldown_sec":60,"min_confidence":0.5}`.

## API-ключи

Только admin:

- GET `/api-keys` — id, name, created_at, revoked_at; без секрета и хеша.
- POST `/api-keys`, JSON `{"name":"ИИ-модуль цеха"}` — 201 с теми же полями
  и дополнительным `key`, показанным единственный раз.
- DELETE `/api-keys/{id}` — отзыв (повторный отзыв безопасен), запись остаётся
  в журнале. Повторное чтение config отозванным ключом даёт 401.

Ключ модуля ИИ (`ai-module`) создаёт сид при старте: из SEED_AI_API_KEY (ключ
заменяется при смене значения, в журнал пишется только факт), иначе один раз
случайный — он печатается в stdout и сохраняется в `.secrets/ai_api_key`.


## Ingest событий — раздел 6.1

POST `/ingest/events` поддерживает два Content-Type:

1. application/json: поля события ниже и необязательное snapshot_base64.
2. multipart/form-data: payload — JSON-строка события без snapshot_base64,
   snapshot — JPEG-файл. Не больше одного файла, неизвестные поля формы отклоняются.

```json
{
  "external_id": "cam-1-1762179667-17-ppe_violation",
  "camera_id": "<UUID камеры>",
  "zone_id": "<UUID зоны или null>",
  "type": "ppe_violation",
  "severity": null,
  "confidence": 0.91,
  "track_id": 17,
  "started_at": "2026-11-03T14:21:07Z",
  "ended_at": null,
  "details": {
    "missing_ppe": ["helmet"],
    "present_ppe": ["vest"],
    "boxes": [{"track_id":17,"label":"person","bbox":[0.41,0.22,0.12,0.46],"conf":0.93}],
    "frame_size": [1280,720]
  },
  "snapshot_base64": "<base64 JPEG, необязательно>"
}
```

X-API-Key обязателен. Время требует timezone и приводится к UTC. zone_id — null
или зона этой камеры. details проверяются по типу с ключами раздела 4.1
контракта, лишние ключи отклоняются; boxes (непустой список, label из словаря
контракта) и frame_size обязательны. external_id =
`{camera_id}-{unix_seconds}-{track_id}-{type}`, track_id — целое. Целые числа
ограничены 64 битами; снимок должен начинаться с FF D8 FF. Ошибки — 422 с полями.
Коды типов инцидентов берутся из включённого справочника.
severity выбирается: явное значение → severity_override зоны → значение типа;
для отсутствия каски в danger-зоне дефолт повышается до high. Бэкенд не вычисляет
рамки, не анализирует СИЗ и не распознаёт людей: он проверяет форму присланных данных.

- **201**: `{"id":"...","status":"new","severity":"medium","deduplicated":false}`.
- **200**, повтор external_id: тот же id и текущие status/severity, deduplicated=true.
- **202**, кулдаун: `{"deduplicated":true,"id":"<существующий инцидент>"}`.

Кулдаун — camera + zone + type + track_id: событие подавляется, если его started_at
отличается от started_at такого инцидента меньше чем на cooldown_sec. Сравнивается
время события, а не приёма: очередь ИИ после сбоя связи доставляется без потерь.
null сравнивается как null. cooldown_sec=0 отключает правило.
Проверка и запись сериализуются блокировкой камеры, external_id защищён уникальным
ключом. Внешние ID, подавленные кулдауном, записываются в служебную ingest_receipts:
повтор такого ID всегда даёт 200 даже после кулдауна, а PATCH находит исходную запись.
Дедупликация не создаёт снимков, комментариев, записей таймлайна или новых уведомлений.

JPEG: максимум 5 МБ и 16 мегапикселей, общий HTTP-запрос до 8 МБ. Принимается raw
base64 и префикс data:image/jpeg;base64,. Снимок проверяется, перекодируется без
EXIF, создаётся миниатюра до 320 px. Ошибки: 413 для размера, 422 для невалидных
JSON/base64/JPEG/рамок, 415 для неизвестного Content-Type; общий JSON-конверт ошибки
сохраняется. Заголовок Content-Type и имя файла не заменяют проверку содержимого.

PATCH `/ingest/events/{external_id}`, JSON:

```json
{"ended_at":"2026-11-03T14:21:19Z"}
```

Обновляет ended_at и duration_sec, рассылает incident.updated, возвращает инцидент.
Время окончания не может быть раньше начала. Это завершение наблюдения, а не
закрытие оператором: status не меняется. PATCH поддерживает external_id со слешами;
клиент должен URL-кодировать идентификатор. Неизвестный external_id — 404.

## Heartbeat и compliance — раздел 6.1

POST `/ingest/heartbeat`, каждые 5 секунд:

```json
{"camera_id":"<UUID>","fps":24.5,"model_name":"yolo26n-ppe","model_version":"0.3","status":"ok","extra":{"device":"cuda:0"}}
```

Требуется API-ключ. ok/online → online, error → degraded; также принимаются
explicit offline/degraded. last_heartbeat_at задаётся сервером. Сохраняются FPS,
модель/версия/extra, рассылается camera.status. Ответ — тот же статус камеры.
После 15 секунд тишины (настройка heartbeat_timeout_sec) фоновый монитор переводит
камеру в offline и рассылает переход; отсутствие heartbeat не создаёт инцидент.

POST `/ingest/stats`, раз в минуту:

```json
{"camera_id":"<UUID>","period_start":"2026-11-03T14:20:00Z","period_end":"2026-11-03T14:21:00Z","persons_total":42,"persons_compliant":39,"by_ppe":{"helmet":{"checked":42,"compliant":40},"vest":{"checked":42,"compliant":41}}}
```

Требуется API-ключ. compliant <= total, количества неотрицательны, end > start.
Повтор ровно того же camera/start/end заменяет запись. Ответ 201:
`{"id":"...","compliance_percent":92.86}`. При total=0 процент null.
Для обновления оперативной сводки также отправляется актуальное camera.status;
клиент по этому событию перечитывает summary. Сырые кадры/детекции в БД не пишутся.

## Журнал и жизненный цикл инцидентов — раздел 6.5

Везде требуется JWT. GET `/incidents`:

| Параметр | Значение |
|---|---|
| from / to | ISO 8601 с timezone, включительно, по started_at |
| camera_id / area_id / zone_id | фильтры расположения |
| type / severity / status | код типа / low,medium,high,critical / статус |
| assignee_id | UUID или unassigned |
| q | поиск по id, external_id, имени камеры и зоны |
| only_unacknowledged | true: только new |
| source | ai, mock, manual; несколько через запятую |
| page / page_size | страница от 1 / размер 1…100, по умолчанию 20 |
| sort | started_at, created_at, severity, status; префикс - для обратного порядка |

Ответ: `{items,total,page,page_size}`. Инциденты содержат ссылки snapshot_url,
thumbnail_url, clip_url (null, если не задан), имена камеры/зоны/ответственного,
reaction_sec и исходные details. Время UTC сериализуется с Z. person_ref не
заполняется и не принимается ingest; идентификатор человека — только track_id.

GET `/incidents/{id}` дополнительно возвращает timeline, comments, previous_id,
next_id. Навигация соседей — по started_at/id; не зависит от фильтра журнала.
GET `/incidents/export.csv` принимает те же фильтры и sort, выгружает все строки,
а не страницу. CSV в UTF-8 с BOM и разделителем ;, формулы в строках экранируются.
GET `/incident-assignees` — пользователи с правами работы с инцидентами.

| Действие POST | Тело | Переход |
|---|---|---|
| `/incidents/{id}/acknowledge` | без тела | new → acknowledged, повтор безопасен |
| `/incidents/{id}/assign` | `{user_id:"UUID или null"}` | назначить / снять ответственного |
| `/incidents/{id}/resolve` | `{note:"принятая мера"}` | acknowledged → resolved |
| `/incidents/{id}/false-positive` | `{reason:"причина"}` | любой → false_positive |
| `/incidents/{id}/reopen` | без тела | resolved/false_positive → acknowledged |
| `/incidents/{id}/comments` | `{text:"комментарий"}` | добавить комментарий и запись таймлайна |

GET `/incidents/{id}/comments` — список комментариев. Пустые причины/меры/комментарии
отклоняются (422), недопустимый переход — 409, недостаточные права — 403.
Первое acknowledged_at/acknowledged_by сохраняется после переоткрытия; поле реакции
считается от created_at. PATCH ended_at на время реакции не влияет.

Admin/engineer — все действия. Master — acknowledge/resolve/comment только своего
участка. Viewer — только чтение. Мастера другого участка и viewer не могут быть
назначены ответственными. Все проверки выполняются сервером.

POST `/incidents/bulk`, JSON `{ids:["..."],action:"acknowledge"}` либо
`{ids:["..."],action:"false_positive",reason:"..."}`. До 100 уникальных id.
Операция атомарна: если одно действие запрещено или конфликтует, откатывается всё.
Ответ `{items,updated}`, после коммита рассылаются incident.updated для изменённых.

GET `/analytics/summary` — оперативная SQL-сводка по тем же фильтрам:
incidents_total, open_incidents, critical_incidents, mean_reaction_sec,
compliance_percent, online_cameras, total_cameras. Compliance фильтруется по
периоду/camera/area: для камеры с реальной статистикой (source=ai) за период
учитывается только она, демо-статистика — только для камер без реальной.
Средняя реакция — по инцидентам с acknowledged_at.
Полные аналитические графики, отчёты и печатные акты — последующие этапы.

## In-app уведомления

POST события создаёт Notification для каждого admin/engineer/viewer и мастеров
его участка. channel=inapp, status=sent означает наличие уведомления в центре,
а не доказательство фактического прочтения. Таймлайн содержит notified с каналом.

- GET `/notifications/inbox?page=1&page_size=30`: `{items,unread_count,total,page,page_size}`.
  Только собственные записи, с read_at. Состояние прочтения хранится в БД.
- POST `/notifications/read-all`: отметить собственные записи, ответ `{updated}`.
- GET `/notifications/log`: журнал отправок с пагинацией; admin/engineer видят все,
  остальные — только свои. Telegram в этапе 3 не отправляется.

## WebSocket — разделы 6.3–6.4

Браузер: `ws://host/ws/live?token=<JWT>`. ИИ: `ws://host/ws/ingest?api_key=<ключ>`.
Неавторизованный/просроченный/отозванный доступ закрывается с 4401. Query-секреты
редактируются фильтром access-логов. Проверка токена/ключа выполняется перед отправкой
и перед приёмом метаданных от ИИ. Используйте wss при сетевом развёртывании с HTTPS.

Браузер посылает:

```json
{"type":"subscribe","cameras":["<UUID>"]}
```

unsubscribe с теми же полями снимает подписку. Она относится только к detections;
инциденты и статусы доставляются глобально, notification — своему пользователю.

События сервера (формат раздела 6.4):

```json
{"type":"incident.created","incident":{"id":"..."}}
{"type":"incident.updated","incident":{"id":"..."}}
{"type":"camera.status","camera_id":"...","status":"online","fps":24.5,"model_name":"...","last_heartbeat_at":"..."}
{"type":"config.updated","camera_id":"...","version":43}
{"type":"notification","text":"Нарушение СИЗ · Пресс №2","incident_id":"..."}
```

В incident передаётся полный сериализованный объект. Изменения публикуются только
после коммита БД; дубликаты не рассылаются. notification содержит также id/created_at.

ИИ передаёт detections, heartbeat и JPEG-кадры по ws/ingest. Detections проверяются по схеме, не сохраняются в БД,
пересылаются только подписанным клиентам. Сообщение соответствует разделу 6.3:
type, camera_id, ts, frame_id, objects (track_id,label,bbox,conf,ppe,zones,state).
state — ok / violation / danger. Значения ppe: true / false / null (null — неизвестно),
ключи — коды СИЗ; null пересылается браузеру без изменений. При подписке выдаётся
последнее состояние камеры. Некорректный JSON, бинарный кадр или неверное сообщение
дают ответ `{"type":"error","message":"..."}`; соединение не закрывается.

Ping каждые 20 с, ответ `{type:"pong"}`; клиентский ping тоже получает pong.
После 60 с отсутствия сообщений клиент закрывается с 4408. У медленного клиента
коалесцируются детекции по камере. Если переполняется очередь из 64 событий,
соединение закрывается с 1013: после переподключения надо перечитать REST.
Браузер реализует backoff 1,2,4,8,16,30 с, обновление JWT и восстановление списков,
KPI, непрочитанных уведомлений и критических тревог. Хаб в памяти одной реплики;
несколько независимых workers без общего брокера не поддерживаются.

GET `/health` (без префикса) — API, БД, число активных ИИ-WebSocket и stage=7.


## Live / демо (этап 4)

WebSocket `/ws/ingest` использует тот же API-ключ, что и ingest HTTP.
Дополнительное сообщение (JPEG без data-URL префикса):

```json
{"type":"frame","camera_id":"UUID","ts":"2026-11-04T08:00:00Z","jpeg_base64":"/9j/..."}
```

Не более 5 MiB декодированного JPEG и 16 млн пикселей. Кадры не пишутся на диск.
GET `/api/v1/stream/{camera_id}.mjpeg`: multipart/x-mixed-replace, последний кадр,
до 15 кадров/с без накопления очереди. Авторизация: Bearer JWT, query `token`
для img либо X-API-Key. Нет авторизации — 401, камеры — 404, свежего кадра
(5 секунд) — 503; открытый поток завершается после 5 секунд тишины.
Клиент должен переподключаться. Токены query скрыты в access-логах приложения.

Исходящие `detections` и `camera.status` содержат дополнительное поле source:
`ai` / `mock`. Статус камеры также содержит relay_frame_age_sec (null без кадра).
Детекции сохраняются только в оперативной памяти; подписка `/ws/live`:
`{"type":"subscribe","cameras":["UUID"]}`. Для снятия используйте unsubscribe с тем же списком.
`demo.updated` означает необходимость перечитать статус демо/списки.
GET `/api/v1/incidents?camera_ids=UUID1,UUID2` фильтрует Live-журнал.

Все demo HTTP требуют JWT:

| Метод / путь | Роль | Тело / результат |
|---|---|---|
| GET /api/v1/demo | все | enabled, rate, seeding, error, active_camera_ids, real_camera_ids |
| PUT /api/v1/demo | admin, engineer | {"enabled":true,"rate":1}; rate 0.1…10 |
| POST /api/v1/demo/simulate | admin, engineer | {"camera_id":"UUID","type":"fall"}; 201 полный инцидент |
| POST /api/v1/demo/reset | admin | пересоздание source=mock; 409 если демо выключено |

Симуляция использует общий ingest-путь, cooldown, снимки, аудит и уведомления.
При активном реальном ИИ создание моков блокируется. Настройки сохраняются в БД.
`ComplianceStat.source` добавлен миграцией 81dea44abe81; существующие строки — ai.
Runtime, relay и подписки рассчитаны на одну реплику API.


## Аналитика и план (этап 5)

| Метод и путь | Доступ | Результат |
|---|---|---|
| GET /api/v1/analytics/overview | Все роли | Показатели текущего и предыдущего периодов, серии, распределения и сводка по камерам |
| GET /api/v1/plan | Все роли | Планы участков, камеры, зоны, статусы и число инцидентов на камеру |
| PUT /api/v1/plan/{area_id}/image | admin, engineer | multipart, поле file. Принимает PNG/JPG до 10 МиБ и 16 Мп. Возвращает url, width, height |
| PATCH /api/v1/plan/cameras/{camera_id} | admin, engineer | Координаты {"x":0.25,"y":0.75}, обе в диапазоне 0…1 |

У GET-запросов есть параметры period=today|shift|7d|30d|custom и area_id.
Для custom передайте date_from и date_to в ISO-формате YYYY-MM-DD. Последняя
дата включена целиком, максимальный диапазон 366 дней. start/end в ответе
передают UTC-границы полуоткрытого интервала. События учитываются по started_at,
смены начинаются в 08:00 и 20:00 в часовом поясе предприятия.

Overview также принимает camera_id. При dashboard=true ответ возвращает
24 часовых столбца за сегодня и статусы камер вместо подробных серий.
Для сравнения поля previous_start и previous_end описывают предыдущий интервал.
Соответствие СИЗ считается как 100 × SUM(persons_compliant) / SUM(persons_total)
по интервалам статистики, целиком попавшим в выбранный период. При отсутствии
наблюдений возвращается null. Средняя реакция считается по времени от создания
до принятия. GET /api/v1/analytics/summary сохранён для прежних клиентов.


## Уведомления, документы и настройки (этап 6)

| Метод и путь | Доступ | Назначение |
|---|---|---|
| GET /api/v1/settings/telegram | admin, engineer | Настройки с has_token, без значения токена |
| PUT /api/v1/settings/telegram | admin, engineer | token, recipients [{chat_id,label}], rules |
| POST /api/v1/settings/telegram/test | admin, engineer | {chat_id}, постановка в очередь (202) |
| GET /api/v1/settings/telegram/journal?page=1 | admin, engineer | 25 записей, статусы pending/sent/failed/skipped, попытки и ошибка |
| POST /api/v1/reports | admin, engineer | Создать summary или prescription за period с area_id/camera_ids |
| GET /api/v1/reports?page=1 | Все роли | 20 документов и total |
| GET /api/v1/reports/{id}/document | Все роли | HTML, download=true добавляет скачивание |
| GET /api/v1/reports/{id}/data.csv | Все роли | CSV данных сохранённой сводки или предписания |
| POST /api/v1/incidents/{id}/act/draft | admin, engineer | Поля акта с шаблонным описанием |
| POST /api/v1/incidents/{id}/act/preview | admin, engineer | HTML предпросмотра по отредактированным полям |
| POST /api/v1/incidents/{id}/act | admin, engineer | Присвоить номер и сохранить акт |
| PUT /api/v1/settings/general | admin, engineer | Предприятие, цех, тема, часовой пояс, звук, пороги и срок хранения |
| PUT /api/v1/settings/ppe-types/{code} | admin, engineer | Создать или обновить тип СИЗ |
| PUT /api/v1/settings/incident-types/{code} | admin, engineer | Создать или обновить тип инцидента |
| POST /api/v1/settings/areas | admin, engineer | Создать участок |
| PUT /api/v1/settings/areas/{id} | admin, engineer | Обновить участок |
| POST /api/v1/users | admin | Создать пользователя |
| PUT /api/v1/users/{id} | admin | Обновить пользователя. Пароль null сохраняет прежний |

В PUT Telegram token=null оставляет текущий токен, пустая строка удаляет его.
Поля rules — enabled, include_demo, types, severities, quiet_start, quiet_end,
cooldown_sec, base_url. При включённой автоотправке обязательны токен, получатель
и URL. Тихие часы задаются парой HH:MM. Отправка снимка использует Telegram Bot API,
а journal сохраняет результат и число попыток. Ключи ИИ и контракт ingest прежние.

Периоды отчёта совпадают с аналитикой. Для предписания нужны area_id, measures и
deadline в формате YYYY-MM-DD. Документ поддерживает до 500 инцидентов.
Поля акта берутся из draft, затем отправляются в preview или act без изменения
формата. HTML и CSV сохранённых документов доступны с JWT. PDF создаётся
из HTML через системный диалог печати браузера.


## Страница проекта и видеостена (этап 7)

- GET /api/v1/public/project — публичное имя предприятия и model_metrics
  с полями map_percent, fps, latency_ms. У незаполненных метрик значение null.
- PUT /api/v1/settings/model-metrics — JWT admin/engineer, принимает тот же
  объект model_metrics. Допустимые диапазоны 0…100 %, 0…10000 FPS и 0…60000 мс.
- /wall использует GET /api/v1/cameras, GET /api/v1/analytics/overview
  с period=today&dashboard=true, журнал инцидентов и обычную подписку /ws/live.
  Контракт ИИ остаётся общим для Live и видеостены.

Тестовый клиент вызывается через scripts/fake_ai_client.py с --camera UUID и
--api-key либо AI_API_KEY. Перед запуском он читает конфигурацию по REST.
Heartbeat, detections и необязательный frame идут через /ws/ingest, события
и снимки — POST /api/v1/ingest/events. --video и --webcam выбирают источник
снимков через FFmpeg. По умолчанию используется сгенерированный кадр.
