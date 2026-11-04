import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api, readSession } from "../../api";
import { Button } from "../../components/ui/button";
import { PeriodControls, QueryState } from "./Controls";
import { usePeriod, usePlans } from "./data";
import { incidentLink, PlanMap } from "./PlanMap";
import "./metrics.css";

export default function PlanPage() {
  const filter = usePeriod("7d");
  const plans = usePlans(filter.params, filter.valid);
  const data = plans.data;
  const area =
    data?.areas.find((a) => a.id === filter.search.get("area_id")) ??
    data?.areas[0];
  const [live, setLive] = useState(true);
  const [editing, setEditing] = useState(false);
  const role = readSession()?.user.role;
  const editable = role === "admin" || role === "engineer";
  const client = useQueryClient();
  const upload = useMutation({
    mutationFn: ({ id, file }: { id: string; file: File }) => {
      const body = new FormData();
      body.append("file", file);
      return api(`/api/v1/plan/${id}/image`, { method: "PUT", body });
    },
    onSuccess: () => client.invalidateQueries({ queryKey: ["plan"] }),
  });
  const move = useMutation({
    mutationFn: ({
      id,
      position,
    }: {
      id: string;
      position: { x: number; y: number };
    }) =>
      api(`/api/v1/plan/cameras/${id}`, {
        method: "PATCH",
        body: JSON.stringify(position),
      }),
    onSuccess: async () => {
      await Promise.all([
        client.invalidateQueries({ queryKey: ["plan"] }),
        client.invalidateQueries({ queryKey: ["cameras"] }),
        client.invalidateQueries({ queryKey: ["cameras-options"] }),
      ]);
    },
  });
  const max = Math.max(0, ...(area?.cameras.map((c) => c.count) ?? []));
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">Механический цех №3</span>
          <h1>План цеха</h1>
          <p>Расположение камер и концентрация инцидентов по участкам</p>
        </div>
      </div>
      <PeriodControls filter={filter}>
        <label>
          Участок{" "}
          <select
            value={area?.id ?? ""}
            onChange={(e) => filter.change("area_id", e.target.value)}
          >
            {!data?.areas.length && <option value="">Нет участков</option>}
            {data?.areas.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
          </select>
        </label>
        <label className="metric-check">
          <input
            type="checkbox"
            checked={live}
            onChange={(e) => setLive(e.target.checked)}
          />{" "}
          Live-статус
        </label>
      </PeriodControls>
      {!filter.valid ? (
        <p className="muted">Выберите начало и конец периода.</p>
      ) : !data ? (
        <QueryState query={plans} />
      ) : !area ? (
        <div className="panel">
          Добавьте участок и камеры, чтобы заполнить план.
        </div>
      ) : (
        <section className="panel plan-panel">
          <div className="panel-heading">
            <h2>{area.name}</h2>
            {editable && (
              <Button
                variant="outline"
                aria-pressed={editing}
                onClick={() => setEditing(!editing)}
              >
                {editing ? "Завершить размещение" : "Разместить камеры"}
              </Button>
            )}
          </div>
          {!area.image && (
            <p className="muted">
              План участка ещё не загружен. На схеме показаны сохранённые
              позиции камер.
            </p>
          )}
          {editable && (
            <label className="plan-upload">
              Загрузить план PNG/JPG
              <input
                type="file"
                accept="image/png,image/jpeg"
                disabled={upload.isPending}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) upload.mutate({ id: area.id, file });
                  e.target.value = "";
                }}
              />
            </label>
          )}
          {editing && (
            <p className="muted">
              Перетащите маркер или используйте стрелки на выбранной камере.
            </p>
          )}
          {(move.error || upload.error) && (
            <p className="error" role="alert">
              {move.error?.message ?? upload.error?.message}
            </p>
          )}
          {(upload.isPending || move.isPending) && (
            <p role="status">Сохранение…</p>
          )}
          <PlanMap
            key={area.id}
            area={area}
            data={data}
            max={max}
            live={live}
            move={
              editing && editable
                ? (id, position) => {
                    if (!move.isPending) move.mutate({ id, position });
                  }
                : undefined
            }
          />
          <div className="plan-legend">
            <span>Инциденты</span>
            <span>0</span>
            <i />
            <span>{max}</span>
            {live && (
              <>
                <span className="plan-status-key online">Онлайн</span>
                <span className="plan-status-key degraded">Нестабильно</span>
                <span className="plan-status-key offline">Офлайн</span>
                <span className="plan-status-key recent">
                  Инцидент за последний час
                </span>
              </>
            )}
          </div>
          <ul className="plan-camera-list">
            {area.cameras.map((c) => (
              <li key={c.id}>
                <Link className="accent" to={incidentLink(c.id, data)}>
                  {c.name}
                </Link>
                <span>
                  {c.count} инц.
                  {c.x == null || c.y == null ? " · без позиции" : ""}
                </span>
                {editing && editable && (c.x == null || c.y == null) && (
                  <Button
                    variant="outline"
                    disabled={move.isPending}
                    onClick={() =>
                      move.mutate({ id: c.id, position: { x: 0.5, y: 0.5 } })
                    }
                  >
                    На план
                  </Button>
                )}
              </li>
            ))}
          </ul>
          {!area.cameras.length && (
            <p className="muted">На этом участке нет камер.</p>
          )}
        </section>
      )}
    </>
  );
}
