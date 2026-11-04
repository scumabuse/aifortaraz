import { useState, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, readSession } from "../../api";
import { Button } from "../../components/ui/button";
import { QueryState } from "../analytics/Controls";
import { useLiveStore } from "../realtime/store";
import { DemoSettings } from "../live/DemoSettings";
import { ApiKeys } from "./ApiKeys";
import { ModelSettings } from "./ModelSettings";
import { TelegramSettings } from "./TelegramSettings";
import ru from "../../locales/ru.json";
import "./settings.css";

type SettingRow = {
  id?: string;
  code?: string;
  name?: string;
  name_ru?: string;
  description?: string;
  icon?: string;
  default_severity?: string;
  color?: string;
  enabled?: boolean;
  sort_order?: number;
  username?: string;
  full_name?: string;
  role?: keyof typeof ru.roles;
  area_id?: string;
};
type Catalog = "ppe-types" | "incident-types" | "areas" | "users";
const catalogs = {
  "ppe-types": "Типы СИЗ",
  "incident-types": "Типы инцидентов",
  areas: "Участки",
  users: "Пользователи",
};
const severityLabels = {
  low: "Низкая",
  medium: "Средняя",
  high: "Высокая",
  critical: "Критическая",
};
type General = {
  enterprise_name: string;
  workshop_name: string;
  timezone: string;
  theme: string;
  alarm_sound: boolean;
  confirm_frames: number;
  cooldown_sec: number;
  min_confidence: number;
  heartbeat_timeout_sec: number;
  snapshot_retention_days: number;
};

export default function SettingsPage() {
  const role = readSession()?.user.role;
  const edit = role === "admin" || role === "engineer";
  const tabs = [
    ["general", "Общие"],
    ...Object.entries(catalogs).filter(
      ([key]) => key !== "users" || role === "admin",
    ),
    ...(edit
      ? [
          ["model", "Метрики модели"],
          ["telegram", "Уведомления"],
          ["demo", "Демо-режим"],
        ]
      : []),
    ...(role === "admin" ? [["api-keys", "API-ключи"]] : []),
  ];
  const [tab, setTab] = useState("general");
  return (
    <>
      <div className="page-heading">
        <div>
          <h1>Настройки</h1>
          <p>Предприятие, пороги ИИ, справочники и доступ</p>
        </div>
      </div>
      <div className="tabs">
        {tabs.map(([key, label]) => (
          <button
            key={key}
            className={tab === key ? "selected" : ""}
            onClick={() => setTab(key)}
          >
            {label}
          </button>
        ))}
      </div>
      <section className="panel">
        {tab === "general" ? (
          <GeneralSettings editable={edit} />
        ) : tab === "model" ? (
          <ModelSettings />
        ) : tab === "telegram" ? (
          <TelegramSettings />
        ) : tab === "demo" ? (
          <DemoSettings />
        ) : tab === "api-keys" ? (
          <ApiKeys />
        ) : (
          <CatalogTable key={tab} catalog={tab as Catalog} editable={edit} />
        )}
      </section>
    </>
  );
}

function GeneralSettings({ editable }: { editable: boolean }) {
  const client = useQueryClient();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [error, setError] = useState("");
  const data = useQuery({
    queryKey: ["enterprise-settings", readSession()?.user.id],
    queryFn: () => api<General>("/api/v1/settings"),
  });
  async function save(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const form = new FormData(e.currentTarget);
    setBusy(true);
    setError("");
    setMessage("");
    const config = {
      enterprise_name: form.get("enterprise_name"),
      workshop_name: form.get("workshop_name"),
      timezone: form.get("timezone"),
      theme: form.get("theme"),
      alarm_sound: form.has("alarm_sound"),
      confirm_frames: Number(form.get("confirm_frames")),
      cooldown_sec: Number(form.get("cooldown_sec")),
      min_confidence: Number(form.get("min_confidence")),
      heartbeat_timeout_sec: Number(form.get("heartbeat_timeout_sec")),
      snapshot_retention_days: Number(form.get("snapshot_retention_days")),
    };
    try {
      await api("/api/v1/settings/general", {
        method: "PUT",
        body: JSON.stringify(config),
      });
      useLiveStore.getState().setSound(config.alarm_sound);
      await client.invalidateQueries({ queryKey: ["enterprise-settings"] });
      setMessage("Настройки сохранены");
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }
  if (!data.data) return <QueryState query={data} />;
  const config = data.data;
  return (
    <>
      <h2>Общие настройки</h2>
      <form className="edit-form" onSubmit={save}>
        <fieldset className="form-grid" disabled={!editable || busy}>
          <label>
            Предприятие
            <input
              name="enterprise_name"
              required
              maxLength={200}
              defaultValue={config.enterprise_name}
            />
          </label>
          <label>
            Цех
            <input
              name="workshop_name"
              required
              maxLength={200}
              defaultValue={config.workshop_name}
            />
          </label>
          <label>
            Часовой пояс IANA
            <input
              name="timezone"
              required
              maxLength={100}
              defaultValue={config.timezone}
            />
          </label>
          <label>
            Тема
            <select name="theme" defaultValue={config.theme}>
              <option value="dark">Тёмная</option>
              <option value="light">Светлая</option>
            </select>
          </label>
          <label>
            Кадры подтверждения
            <input
              type="number"
              name="confirm_frames"
              min={1}
              max={10000}
              required
              defaultValue={config.confirm_frames}
            />
          </label>
          <label>
            Кулдаун, с
            <input
              type="number"
              name="cooldown_sec"
              min={0}
              max={86400}
              required
              defaultValue={config.cooldown_sec}
            />
          </label>
          <label>
            Минимальная уверенность
            <input
              type="number"
              name="min_confidence"
              min={0}
              max={1}
              step="any"
              required
              defaultValue={config.min_confidence}
            />
          </label>
          <label>
            Таймаут heartbeat, с
            <input
              type="number"
              name="heartbeat_timeout_sec"
              min={1}
              max={3600}
              required
              defaultValue={config.heartbeat_timeout_sec}
            />
          </label>
          <label>
            Хранить снимки, дней
            <input
              type="number"
              name="snapshot_retention_days"
              min={1}
              max={3650}
              required
              defaultValue={config.snapshot_retention_days}
            />
          </label>
          <label className="check-label">
            <input
              type="checkbox"
              name="alarm_sound"
              defaultChecked={config.alarm_sound}
            />{" "}
            Звук тревоги по умолчанию
          </label>
        </fieldset>
        {editable && (
          <div>
            <Button disabled={busy}>Сохранить настройки</Button>
          </div>
        )}
        {message && <p role="status">{message}</p>}
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
      </form>
    </>
  );
}

function CatalogTable({
  catalog,
  editable,
}: {
  catalog: Catalog;
  editable: boolean;
}) {
  const path =
    catalog === "users" ? "/api/v1/users" : `/api/v1/settings/${catalog}`;
  const client = useQueryClient();
  const list = useQuery({
    queryKey: ["settings-catalog", catalog],
    queryFn: () => api<SettingRow[]>(path),
  });
  const areas = useQuery({
    queryKey: ["analytics-areas"],
    queryFn: () => api<SettingRow[]>("/api/v1/settings/areas"),
    enabled: catalog === "users",
  });
  const [row, setRow] = useState<SettingRow>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const coded = catalog === "ppe-types" || catalog === "incident-types";
  async function save(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    if (!row) return;
    setBusy(true);
    setError("");
    const form = new FormData(e.currentTarget);
    const get = (key: string) => String(form.get(key) ?? "").trim();
    const values =
      catalog === "users"
        ? {
            username: get("username"),
            full_name: get("full_name"),
            role: get("role"),
            area_id: get("area_id") || null,
            password: form.get("password") || null,
          }
        : catalog === "areas"
          ? { name: get("name"), description: get("description") || null }
          : {
              name_ru: get("name_ru"),
              enabled: form.has("enabled"),
              ...(catalog === "ppe-types"
                ? {
                    icon: get("icon") || null,
                    sort_order: Number(get("sort_order")),
                  }
                : {
                    default_severity: get("default_severity"),
                    color: get("color"),
                  }),
            };
    const id = row.id ?? (coded ? get("code") : undefined);
    try {
      await api(id ? `${path}/${encodeURIComponent(id)}` : path, {
        method: id ? "PUT" : "POST",
        body: JSON.stringify(values),
      });
      setRow(undefined);
      for (const key of [
        "settings-catalog",
        "analytics-areas",
        "areas",
        "ppe-types",
        "incident-types",
        "cameras",
        "cameras-options",
        "summary",
        "plan",
        "assignees",
        "zone-responsibles",
      ])
        void client.invalidateQueries({ queryKey: [key] });
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="panel-heading">
        <h2>{catalogs[catalog]}</h2>
        {editable && !row && (
          <Button
            onClick={() => {
              setError("");
              setRow({});
            }}
          >
            Добавить
          </Button>
        )}
      </div>
      {!list.data ? (
        <QueryState query={list} />
      ) : list.data.length ? (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>
                  {coded ? "Код" : catalog === "users" ? "Логин" : "Участок"}
                </th>
                <th>Данные</th>
                {editable && <th>Действия</th>}
              </tr>
            </thead>
            <tbody>
              {list.data.map((item) => (
                <tr key={item.id ?? item.code}>
                  <td>{item.code ?? item.username ?? item.name}</td>
                  <td>
                    {item.name_ru ?? item.full_name ?? item.description ?? "—"}
                    {item.icon && ` · ${item.icon}`}
                    {item.default_severity &&
                      ` · ${severityLabels[item.default_severity as keyof typeof severityLabels]} · ${item.color}`}
                    {item.role && ` · ${ru.roles[item.role]}`}
                    {item.area_id &&
                      ` · ${areas.data?.find((a) => a.id === item.area_id)?.name ?? item.area_id}`}
                    {item.enabled === false && " · отключён"}
                  </td>
                  {editable && (
                    <td>
                      <Button
                        variant="outline"
                        disabled={!!row}
                        onClick={() => {
                          setError("");
                          setRow(item);
                        }}
                      >
                        Изменить
                      </Button>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted">Справочник пуст.</p>
      )}
      {row && (
        <form className="edit-form" onSubmit={save}>
          <h3>{row.id || row.code ? "Редактирование" : "Новая запись"}</h3>
          {catalog === "users" && (
            <>
              <label>
                Логин
                <input
                  name="username"
                  required
                  maxLength={100}
                  defaultValue={row.username}
                />
              </label>
              <label>
                Имя
                <input
                  name="full_name"
                  required
                  maxLength={200}
                  defaultValue={row.full_name}
                />
              </label>
              <label>
                Пароль {row.id && "(пусто — оставить прежний)"}
                <input
                  name="password"
                  type="password"
                  autoComplete="new-password"
                  minLength={8}
                  maxLength={256}
                  required={!row.id}
                />
              </label>
              <label>
                Роль
                <select name="role" defaultValue={row.role ?? "viewer"}>
                  {Object.entries(ru.roles).map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Участок
                <select name="area_id" defaultValue={row.area_id ?? ""}>
                  <option value="">Не указан</option>
                  {areas.data?.map((a) => (
                    <option key={a.id} value={a.id}>
                      {a.name}
                    </option>
                  ))}
                </select>
              </label>
              {areas.isError && <QueryState query={areas} />}
            </>
          )}
          {catalog === "areas" && (
            <>
              <label>
                Название
                <input
                  name="name"
                  required
                  maxLength={200}
                  defaultValue={row.name}
                />
              </label>
              <label>
                Описание
                <textarea
                  name="description"
                  maxLength={2000}
                  defaultValue={row.description}
                />
              </label>
            </>
          )}
          {coded && (
            <>
              <label>
                Код
                <input
                  name="code"
                  required
                  maxLength={50}
                  pattern="[a-z][a-z0-9_]*"
                  defaultValue={row.code}
                  readOnly={!!row.code}
                />
              </label>
              <label>
                Название
                <input
                  name="name_ru"
                  required
                  maxLength={200}
                  defaultValue={row.name_ru}
                />
              </label>
              {catalog === "ppe-types" ? (
                <>
                  <label>
                    Иконка
                    <input name="icon" maxLength={50} defaultValue={row.icon} />
                  </label>
                  <label>
                    Порядок
                    <input
                      type="number"
                      name="sort_order"
                      min={0}
                      max={10000}
                      defaultValue={row.sort_order ?? 0}
                    />
                  </label>
                </>
              ) : (
                <>
                  <label>
                    Критичность
                    <select
                      name="default_severity"
                      defaultValue={row.default_severity ?? "medium"}
                    >
                      {Object.entries(severityLabels).map(([key, label]) => (
                        <option key={key} value={key}>
                          {label}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Цвет
                    <input
                      type="color"
                      name="color"
                      defaultValue={row.color ?? "#eab54d"}
                    />
                  </label>
                </>
              )}
              <label className="check-label">
                <input
                  type="checkbox"
                  name="enabled"
                  defaultChecked={row.enabled ?? true}
                />{" "}
                Включён
              </label>
            </>
          )}
          <div className="action-row">
            <Button disabled={busy}>Сохранить</Button>
            <Button
              type="button"
              variant="outline"
              disabled={busy}
              onClick={() => setRow(undefined)}
            >
              Отмена
            </Button>
          </div>
        </form>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </>
  );
}
