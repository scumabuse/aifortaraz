import { useState, type FormEvent } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api";
import { Button } from "../../components/ui/button";
import { QueryState } from "../analytics/Controls";
import type { ModelMetrics, ProjectInfo } from "../project/AboutPage";

const fields = [
  ["map_percent", "mAP, %", 100],
  ["fps", "Скорость, FPS", 10000],
  ["latency_ms", "Задержка, мс", 60000],
] as const;

export function ModelSettings() {
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["public-project"],
    queryFn: () => api<ProjectInfo>("/api/v1/public/project"),
  });
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const [saved, setSaved] = useState(false);
  async function save(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const metrics = Object.fromEntries(
      fields.map(([key]) => {
        const value = form.get(key);
        return [key, value === "" ? null : Number(value)];
      }),
    ) as ModelMetrics;
    setPending(true);
    setSaved(false);
    setError("");
    try {
      await api("/api/v1/settings/model-metrics", {
        method: "PUT",
        body: JSON.stringify(metrics),
      });
      await client.invalidateQueries({ queryKey: ["public-project"] });
      setSaved(true);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setPending(false);
    }
  }
  if (!query.data) return <QueryState query={query} />;
  return (
    <>
      <h2>Метрики модели</h2>
      <p className="muted">
        Значения публикуются на странице проекта. Пустые поля скрывают числовые
        значения.
      </p>
      <form className="edit-form" onSubmit={save}>
        <div className="form-grid">
          {fields.map(([key, label, max]) => (
            <label key={key}>
              {label}
              <input
                type="number"
                name={key}
                min={0}
                max={max}
                step="any"
                defaultValue={query.data.model_metrics[key] ?? ""}
              />
            </label>
          ))}
        </div>
        <div>
          <Button disabled={pending}>Сохранить</Button>
        </div>
        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}
        {saved && <p role="status">Метрики обновлены</p>}
      </form>
    </>
  );
}
