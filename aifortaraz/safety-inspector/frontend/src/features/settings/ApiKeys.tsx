import { useEffect, useState } from "react";
import { api } from "../../api";
import { Button } from "../../components/ui/button";
import ru from "../../locales/ru.json";
interface ApiKey {
  id: string;
  name: string;
  created_at: string;
  revoked_at: string | null;
}
const t = ru.keyUi;
export function ApiKeys() {
  const [rows, setRows] = useState<ApiKey[]>([]);
  const [name, setName] = useState("");
  const [key, setKey] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [pending, setPending] = useState(false);
  const [attempt, setAttempt] = useState(0);
  useEffect(() => {
    let active = true;
    setLoading(true);
    api<ApiKey[]>("/api/v1/api-keys")
      .then((data) => {
        if (active) {
          setRows(data);
          setError("");
        }
      })
      .catch((e) => {
        if (active) setError(e.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [attempt]);
  async function create(e: React.FormEvent) {
    e.preventDefault();
    setPending(true);
    setError("");
    try {
      const created = await api<ApiKey & { key: string }>("/api/v1/api-keys", {
        method: "POST",
        body: JSON.stringify({ name }),
      });
      setKey(created.key);
      setName("");
      setAttempt((x) => x + 1);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  async function revoke(row: ApiKey) {
    if (!confirm(`${t.revokeConfirm} «${row.name}»?`)) return;
    setPending(true);
    try {
      await api(`/api/v1/api-keys/${row.id}`, { method: "DELETE" });
      setAttempt((x) => x + 1);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  return (
    <div>
      <h2>{t.title}</h2>
      <p className="muted camera-description">{t.hint}</p>
      <form className="inline-form edit-form" onSubmit={create}>
        <label>
          {t.name}
          <input
            required
            maxLength={100}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <Button disabled={pending || !!key}>{t.create}</Button>
      </form>
      {key && (
        <div className="secret-box" role="status">
          <strong>{t.once}</strong>
          <code>{key}</code>
          <div className="action-row">
            <Button
              variant="outline"
              onClick={() => {
                void navigator.clipboard
                  .writeText(key)
                  .catch(() => setError(t.copyError));
              }}
            >
              {t.copy}
            </Button>
            <Button variant="outline" onClick={() => setKey("")}>
              {t.saved}
            </Button>
          </div>
        </div>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
          <Button variant="outline" onClick={() => setAttempt((x) => x + 1)}>
            {ru.retry}
          </Button>
        </p>
      )}
      {loading ? (
        <div className="skeleton">{ru.loading}</div>
      ) : rows.length ? (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>{t.name}</th>
                <th>{t.created}</th>
                <th>{t.status}</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id}>
                  <td>{row.name}</td>
                  <td>{new Date(row.created_at).toLocaleString("ru-RU")}</td>
                  <td>{row.revoked_at ? t.revoked : t.active}</td>
                  <td>
                    {!row.revoked_at && (
                      <Button
                        variant="outline"
                        disabled={pending}
                        onClick={() => void revoke(row)}
                      >
                        {t.revoke}
                      </Button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted">{t.empty}</p>
      )}
    </div>
  );
}
