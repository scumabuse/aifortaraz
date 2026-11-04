import { useState, type FormEvent } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { api, apiBlob, readSession } from "../../api";
import { Button } from "../../components/ui/button";
import { QueryState } from "../analytics/Controls";
import { periodOptions, type Period } from "../analytics/data";
import type { CameraRecord } from "../cameras/types";
import { formatDate } from "../incidents/utils";
import { download, PrintDocument, useDocument } from "./Document";

type Row = {
  id: string;
  title: string;
  from_: string;
  to: string;
  author: string;
  created_at: string;
  has_csv: boolean;
  incident_id: string | null;
  number: string | null;
};
type Areas = { id: string; name: string }[];
type History = { items: Row[]; total: number };

export default function ReportsPage() {
  const { reportId } = useParams();
  const viewer = useDocument(reportId);
  const client = useQueryClient();
  const [page, setPage] = useState(1);
  const [kind, setKind] = useState("summary");
  const [period, setPeriod] = useState<Period>("7d");
  const [area, setArea] = useState("");
  const [cameraIds, setCameraIds] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const editable = ["admin", "engineer"].includes(
    readSession()?.user.role ?? "",
  );
  const history = useQuery({
    queryKey: ["reports", page],
    queryFn: () => api<History>(`/api/v1/reports?page=${page}`),
    enabled: !reportId,
  });
  const areas = useQuery({
    queryKey: ["analytics-areas"],
    queryFn: () => api<Areas>("/api/v1/settings/areas"),
    enabled: editable && !reportId,
  });
  const cameras = useQuery({
    queryKey: ["cameras-options"],
    queryFn: () => api<CameraRecord[]>("/api/v1/cameras"),
    enabled: editable && !reportId,
  });
  async function generate(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setBusy(true);
    setError("");
    const form = new FormData(event.currentTarget);
    try {
      const row = await api<Row>("/api/v1/reports", {
        method: "POST",
        body: JSON.stringify({
          kind,
          period,
          area_id: area || null,
          camera_ids: cameraIds,
          date_from: form.get("date_from") || null,
          date_to: form.get("date_to") || null,
          measures: form.get("measures") || "",
          deadline: form.get("deadline") || null,
        }),
      });
      await client.invalidateQueries({ queryKey: ["reports"] });
      setSelected(row.id);
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }
  const [selected, setSelected] = useState<string>();
  const generated = useDocument(selected);
  async function file(id: string, csv = false) {
    setError("");
    try {
      await download(
        `/api/v1/reports/${id}/${csv ? "data.csv" : "document?download=true"}`,
        `${id}.${csv ? "csv" : "html"}`,
      );
    } catch (reason) {
      setError((reason as Error).message);
    }
  }
  if (reportId)
    return (
      <>
        <Link className="accent" to="/reports">
          ← К истории документов
        </Link>
        {viewer.document?.html ? (
          <PrintDocument
            key={reportId}
            title="Документ"
            html={viewer.document.html}
          />
        ) : viewer.document?.error ? (
          <div className="panel" role="alert">
            <p>{viewer.document.error}</p>
            <Button onClick={viewer.retry}>Повторить</Button>
          </div>
        ) : (
          <div className="skeleton">Загрузка документа…</div>
        )}
      </>
    );
  return (
    <>
      <div className="page-heading">
        <div>
          <h1>Отчёты и акты</h1>
          <p>Сводки, предписания и история документов</p>
        </div>
      </div>
      {editable && (
        <section className="panel">
          <h2>Сформировать документ</h2>
          {(areas.isError || cameras.isError) && (
            <QueryState query={areas.isError ? areas : cameras} />
          )}
          <form className="edit-form" onSubmit={generate}>
            <div className="form-grid">
              <label>
                Документ
                <select value={kind} onChange={(e) => setKind(e.target.value)}>
                  <option value="summary">Сводка за период</option>
                  <option value="prescription">Предписание</option>
                </select>
              </label>
              <label>
                Период
                <select
                  value={period}
                  onChange={(e) => setPeriod(e.target.value as Period)}
                >
                  {Object.entries(periodOptions).map(([key, label]) => (
                    <option key={key} value={key}>
                      {label}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {period === "custom" && (
              <div className="form-grid">
                <label>
                  С<input type="date" name="date_from" required />
                </label>
                <label>
                  По
                  <input type="date" name="date_to" required />
                </label>
              </div>
            )}
            <label>
              Участок
              <select
                required={kind === "prescription"}
                value={area}
                onChange={(e) => {
                  setArea(e.target.value);
                  setCameraIds([]);
                }}
              >
                <option value="">
                  {kind === "prescription" ? "Выберите участок" : "Все участки"}
                </option>
                {areas.data?.map((a) => (
                  <option key={a.id} value={a.id}>
                    {a.name}
                  </option>
                ))}
              </select>
            </label>
            <fieldset>
              <legend>Камеры · без выбора включаются все</legend>
              <div className="report-checks">
                {cameras.data
                  ?.filter((c) => !area || c.area_id === area)
                  .map((c) => (
                    <label className="check-label" key={c.id}>
                      <input
                        type="checkbox"
                        checked={cameraIds.includes(c.id)}
                        onChange={(e) =>
                          setCameraIds((previous) =>
                            e.target.checked
                              ? [...previous, c.id]
                              : previous.filter((id) => id !== c.id),
                          )
                        }
                      />
                      {c.name}
                    </label>
                  ))}
              </div>
            </fieldset>
            {kind === "prescription" && (
              <>
                <label>
                  Требуемые меры
                  <textarea name="measures" required maxLength={3000} />
                </label>
                <label>
                  Срок устранения
                  <input type="date" name="deadline" required />
                </label>
              </>
            )}
            <div>
              <Button disabled={busy}>
                {busy ? "Формирование…" : "Сформировать"}
              </Button>
            </div>
          </form>
        </section>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {selected &&
        (generated.document?.html ? (
          <PrintDocument
            key={selected}
            html={generated.document.html}
            title="Сформированный документ"
          />
        ) : generated.document?.error ? (
          <div className="panel">
            <p role="alert">{generated.document.error}</p>
            <Button onClick={generated.retry}>Повторить</Button>
          </div>
        ) : (
          <div className="skeleton">Загрузка документа…</div>
        ))}
      <section className="panel report-history">
        <h2>История документов</h2>
        {!history.data ? (
          <QueryState query={history} />
        ) : !history.data.items.length ? (
          <p className="muted">На этой странице нет документов.</p>
        ) : (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Создан</th>
                  <th>Документ</th>
                  <th>Период</th>
                  <th>Автор</th>
                  <th>Файлы</th>
                </tr>
              </thead>
              <tbody>
                {history.data.items.map((row) => (
                  <tr key={row.id}>
                    <td>{formatDate(row.created_at)}</td>
                    <td>
                      <Link className="accent" to={`/reports/${row.id}`}>
                        {row.number ?? row.title}
                      </Link>
                    </td>
                    <td>
                      {formatDate(row.from_)} — {formatDate(row.to)}
                    </td>
                    <td>{row.author}</td>
                    <td>
                      <div className="action-row">
                        <Button
                          variant="outline"
                          onClick={() => void file(row.id)}
                        >
                          HTML
                        </Button>
                        {row.has_csv && (
                          <Button
                            variant="outline"
                            onClick={() => void file(row.id, true)}
                          >
                            CSV
                          </Button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <div className="action-row report-pagination">
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
            disabled={!history.data || page * 20 >= history.data.total}
            onClick={() => setPage(page + 1)}
          >
            Далее
          </Button>
        </div>
      </section>
    </>
  );
}

const fields = {
  enterprise: "Предприятие",
  workshop: "Цех",
  area: "Участок",
  time: "Дата и время",
  camera: "Камера",
  zone: "Зона",
  violation: "Вид нарушения",
  description: "Описание",
  regulation: "Пункт инструкции",
  responsible: "Ответственный",
  measures: "Принятые меры",
  deadline: "Срок устранения",
};
type Act = Record<keyof typeof fields, string>;

export function ActPage() {
  const { id } = useParams();
  const [act, setAct] = useState<Act>();
  const [preview, setPreview] = useState("");
  const [saved, setSaved] = useState<Row>();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const client = useQueryClient();
  const document = useDocument(saved?.id);
  const path = `/api/v1/incidents/${id}/act`;
  const draft = useQuery({
    queryKey: ["act-draft", id],
    queryFn: () => api<Act>(path + "/draft", { method: "POST" }),
    retry: false,
  });
  const model = act ?? draft.data;
  async function action(kind: "preview" | "save" | "describe") {
    setError("");
    setBusy(true);
    try {
      if (kind === "describe") {
        const fresh = await api<Act>(path + "/draft", { method: "POST" });
        setAct({ ...model!, description: fresh.description });
      } else if (kind === "preview") {
        setPreview(
          await (
            await apiBlob(path + "/preview", {
              method: "POST",
              body: JSON.stringify(model),
            })
          ).text(),
        );
      } else {
        const row = await api<Row>(path, {
          method: "POST",
          body: JSON.stringify(model),
        });
        setPreview("");
        setSaved(row);
        await client.invalidateQueries({ queryKey: ["reports"] });
      }
    } catch (reason) {
      setError((reason as Error).message);
    } finally {
      setBusy(false);
    }
  }
  return (
    <>
      <div className="page-heading">
        <div>
          <Link className="accent" to={`/incidents/${id}`}>
            ← К инциденту
          </Link>
          <h1>Акт о нарушении</h1>
        </div>
      </div>
      {!model ? (
        <QueryState query={draft} />
      ) : (
        <section className="panel">
          <form
            className="edit-form"
            onSubmit={(e) => {
              e.preventDefault();
              void action("save");
            }}
          >
            <div className="form-grid">
              {(Object.keys(fields) as (keyof Act)[]).map((key) => {
                const props = {
                  value: model[key],
                  onChange: (
                    e: React.ChangeEvent<
                      HTMLInputElement | HTMLTextAreaElement
                    >,
                  ) => {
                    setAct({ ...model, [key]: e.target.value });
                    setPreview("");
                  },
                };
                return (
                  <label key={key}>
                    {fields[key]}
                    {key === "description" || key === "measures" ? (
                      <textarea {...props} required={key === "description"} />
                    ) : (
                      <input {...props} />
                    )}
                  </label>
                );
              })}
            </div>
            <div className="action-row">
              <Button
                type="button"
                variant="outline"
                disabled={busy}
                onClick={() => void action("describe")}
              >
                Сгенерировать описание
              </Button>
              <Button
                type="button"
                variant="outline"
                disabled={busy || !model.description.trim()}
                onClick={() => void action("preview")}
              >
                Предпросмотр
              </Button>
              <Button disabled={busy}>Сохранить в историю</Button>
            </div>
          </form>
        </section>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {preview && <PrintDocument html={preview} title="Предпросмотр акта" />}
      {saved && (
        <p>
          <Link className="accent" to={`/reports/${saved.id}`}>
            Сохранён {saved.number}
          </Link>
        </p>
      )}
      {saved && document.document?.html && (
        <PrintDocument
          key={saved.id}
          html={document.document.html}
          title={saved.number ?? "Акт"}
        />
      )}
      {saved && document.document?.error && (
        <p className="error" role="alert">
          {document.document.error}
        </p>
      )}
    </>
  );
}
