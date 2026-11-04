import { useState, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../../api";
import { Button } from "../../components/ui/button";
import { QueryState } from "../analytics/Controls";
import { formatDate } from "../incidents/utils";

const severityNames = {
  low: "Низкая",
  medium: "Средняя",
  high: "Высокая",
  critical: "Критическая",
};
const states = {
  pending: "В очереди",
  sent: "Отправлено",
  failed: "Ошибка",
  skipped: "Пропущено",
};
type Rules = {
  enabled: boolean;
  include_demo: boolean;
  types: string[];
  severities: string[];
  quiet_start: string | null;
  quiet_end: string | null;
  cooldown_sec: number;
  base_url: string;
};
type Recipient = { chat_id: string; label: string };
type Config = { has_token: boolean; recipients: Recipient[]; rules: Rules };
type Log = {
  id: string;
  incident_id: string | null;
  recipient: string;
  label: string;
  status: keyof typeof states;
  attempts: number;
  error: string | null;
  created_at: string;
  sent_at: string | null;
};

export function TelegramSettings() {
  const config = useQuery({
    queryKey: ["telegram-config"],
    queryFn: () => api<Config>("/api/v1/settings/telegram"),
  });
  const types = useQuery({
    queryKey: ["incident-types"],
    queryFn: () =>
      api<{ code: string; name_ru: string }[]>(
        "/api/v1/settings/incident-types",
      ),
  });
  const [page, setPage] = useState(1);
  const log = useQuery({
    queryKey: ["telegram-journal", page],
    queryFn: () =>
      api<{ items: Log[]; total: number }>(
        `/api/v1/settings/telegram/journal?page=${page}`,
      ),
    refetchInterval: 3000,
  });
  return (
    <>
      <h2>Telegram</h2>
      <p className="muted">
        Укажите получателей, которым бот может писать. Пустой фильтр включает
        все значения.
      </p>
      {!config.data ? (
        <QueryState query={config} />
      ) : !types.data ? (
        <QueryState query={types} />
      ) : (
        <TelegramForm initial={config.data} types={types.data} />
      )}
      <section className="settings-log">
        <h2>Журнал отправок</h2>
        {!log.data ? (
          <QueryState query={log} />
        ) : !log.data.items.length ? (
          <p className="muted">Отправок на этой странице нет.</p>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Время</th>
                  <th>Канал</th>
                  <th>Получатель</th>
                  <th>Событие</th>
                  <th>Статус</th>
                  <th>Попытки</th>
                </tr>
              </thead>
              <tbody>
                {log.data.items.map((row) => (
                  <tr key={row.id}>
                    <td>{formatDate(row.sent_at ?? row.created_at)}</td>
                    <td>Telegram</td>
                    <td>
                      {row.label} · {row.recipient}
                    </td>
                    <td>
                      {row.incident_id ? (
                        <Link
                          className="accent"
                          to={`/incidents/${row.incident_id}`}
                        >
                          Инцидент
                        </Link>
                      ) : (
                        "Тест"
                      )}
                    </td>
                    <td>
                      {states[row.status]}
                      {row.error && <div className="muted">{row.error}</div>}
                    </td>
                    <td>{row.attempts}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="action-row">
          <Button
            variant="outline"
            disabled={page <= 1}
            onClick={() => setPage(page - 1)}
          >
            Назад
          </Button>
          <span>Страница {page}</span>
          <Button
            variant="outline"
            disabled={!log.data || page * 25 >= log.data.total}
            onClick={() => setPage(page + 1)}
          >
            Далее
          </Button>
        </div>
      </section>
    </>
  );
}

function TelegramForm({
  initial,
  types,
}: {
  initial: Config;
  types: { code: string; name_ru: string }[];
}) {
  const [config, setConfig] = useState(initial);
  const [recipients, setRecipients] = useState(
    initial.recipients.map((r, index) => ({ ...r, key: index })),
  );
  const [key, setKey] = useState(initial.recipients.length);
  const [testChat, setTestChat] = useState(
    initial.recipients[0]?.chat_id ?? "",
  );
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const client = useQueryClient();
  async function save(e: FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const target = e.currentTarget;
    const form = new FormData(target);
    setBusy(true);
    setError("");
    setMessage("");
    try {
      const result = await api<Config>("/api/v1/settings/telegram", {
        method: "PUT",
        body: JSON.stringify({
          token: form.has("clear_token") ? "" : form.get("token") || null,
          recipients: recipients.map(({ chat_id, label }) => ({
            chat_id: chat_id.trim(),
            label: label.trim(),
          })),
          rules: {
            enabled: form.has("enabled"),
            include_demo: form.has("include_demo"),
            base_url: form.get("base_url"),
            cooldown_sec: Number(form.get("cooldown_sec")),
            quiet_start: form.get("quiet_start") || null,
            quiet_end: form.get("quiet_end") || null,
            types: form.getAll("types"),
            severities: form.getAll("severities"),
          },
        }),
      });
      const token = target.elements.namedItem("token");
      const clear = target.elements.namedItem("clear_token");
      if (token instanceof HTMLInputElement) token.value = "";
      if (clear instanceof HTMLInputElement) clear.checked = false;
      setConfig(result);
      setTestChat(
        result.recipients.some((r) => r.chat_id === testChat)
          ? testChat
          : (result.recipients[0]?.chat_id ?? ""),
      );
      client.setQueryData(["telegram-config"], result);
      setMessage("Настройки Telegram сохранены");
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }
  async function sendTest() {
    setBusy(true);
    setError("");
    setMessage("");
    try {
      await api("/api/v1/settings/telegram/test", {
        method: "POST",
        body: JSON.stringify({ chat_id: testChat }),
      });
      setMessage("Тест поставлен в очередь. Результат появится в журнале.");
      void client.invalidateQueries({ queryKey: ["telegram-journal"] });
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const rules = initial.rules;
  return (
    <>
      <form className="edit-form" onSubmit={save}>
        <fieldset disabled={busy} className="settings-fields">
          <label>
            Токен бота
            <input
              name="token"
              type="password"
              autoComplete="new-password"
              maxLength={200}
              placeholder={
                config.has_token
                  ? "Сохранён · введите для замены"
                  : "Введите токен"
              }
            />
          </label>
          {config.has_token && (
            <label className="check-label">
              <input name="clear_token" type="checkbox" /> Удалить сохранённый
              токен
            </label>
          )}
          <label className="check-label">
            <input
              name="enabled"
              type="checkbox"
              defaultChecked={rules.enabled}
            />{" "}
            Автоотправка инцидентов
          </label>
          <label className="check-label">
            <input
              name="include_demo"
              type="checkbox"
              defaultChecked={rules.include_demo}
            />{" "}
            Включать демо-инциденты
          </label>
          <label>
            Базовый URL для ссылок на инциденты
            <input
              name="base_url"
              type="url"
              maxLength={400}
              defaultValue={rules.base_url}
              placeholder="https://inspector.example"
            />
          </label>
          <div className="form-grid">
            <label>
              Тихие часы с
              <input
                name="quiet_start"
                type="time"
                defaultValue={rules.quiet_start?.slice(0, 5)}
              />
            </label>
            <label>
              Тихие часы до
              <input
                name="quiet_end"
                type="time"
                defaultValue={rules.quiet_end?.slice(0, 5)}
              />
            </label>
          </div>
          <p className="muted">
            Критические события проходят вне зависимости от тихих часов.
          </p>
          <label>
            Кулдаун камеры и типа, с
            <input
              name="cooldown_sec"
              type="number"
              min={0}
              max={86400}
              required
              defaultValue={rules.cooldown_sec}
            />
          </label>
          <fieldset>
            <legend>Типы инцидентов</legend>
            <div className="settings-checks">
              {types.map((type) => (
                <label className="check-label" key={type.code}>
                  <input
                    type="checkbox"
                    name="types"
                    value={type.code}
                    defaultChecked={rules.types.includes(type.code)}
                  />
                  {type.name_ru}
                </label>
              ))}
            </div>
          </fieldset>
          <fieldset>
            <legend>Критичность</legend>
            <div className="settings-checks">
              {Object.entries(severityNames).map(([value, label]) => (
                <label className="check-label" key={value}>
                  <input
                    type="checkbox"
                    name="severities"
                    value={value}
                    defaultChecked={rules.severities.includes(value)}
                  />
                  {label}
                </label>
              ))}
            </div>
          </fieldset>
          <h3>Получатели</h3>
          {recipients.map((recipient) => (
            <div className="settings-recipient" key={recipient.key}>
              <label>
                Chat ID
                <input
                  value={recipient.chat_id}
                  pattern="-?[1-9][0-9]{0,18}"
                  required
                  onChange={(e) =>
                    setRecipients((items) =>
                      items.map((item) =>
                        item.key === recipient.key
                          ? { ...item, chat_id: e.target.value }
                          : item,
                      ),
                    )
                  }
                />
              </label>
              <label>
                Подпись
                <input
                  value={recipient.label}
                  required
                  maxLength={100}
                  onChange={(e) =>
                    setRecipients((items) =>
                      items.map((item) =>
                        item.key === recipient.key
                          ? { ...item, label: e.target.value }
                          : item,
                      ),
                    )
                  }
                />
              </label>
              <Button
                type="button"
                variant="outline"
                onClick={() =>
                  setRecipients((items) =>
                    items.filter((item) => item.key !== recipient.key),
                  )
                }
              >
                Убрать
              </Button>
            </div>
          ))}
          <div className="action-row">
            <Button
              type="button"
              variant="outline"
              disabled={recipients.length >= 30}
              onClick={() => {
                setRecipients((items) => [
                  ...items,
                  { key, chat_id: "", label: "" },
                ]);
                setKey(key + 1);
              }}
            >
              Добавить получателя
            </Button>
            <Button>Сохранить</Button>
          </div>
        </fieldset>
      </form>
      {config.has_token && config.recipients.length > 0 && (
        <div className="edit-form">
          <label>
            Получатель теста
            <select
              value={testChat}
              onChange={(e) => setTestChat(e.target.value)}
            >
              {config.recipients.map((item) => (
                <option value={item.chat_id} key={item.chat_id}>
                  {item.label} · {item.chat_id}
                </option>
              ))}
            </select>
          </label>
          <div>
            <Button
              disabled={busy || !testChat}
              variant="outline"
              onClick={() => void sendTest()}
            >
              Отправить тест
            </Button>
          </div>
        </div>
      )}
      {message && <p role="status">{message}</p>}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </>
  );
}
