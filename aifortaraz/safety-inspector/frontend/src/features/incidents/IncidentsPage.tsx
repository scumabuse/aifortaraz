import { useEffect, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Download, LayoutGrid, List, Search } from "lucide-react";
import { api, apiBlob, readSession, type DictionaryItem } from "../../api";
import { Button } from "../../components/ui/button";
import type { CameraRecord, ZoneRecord } from "../cameras/types";
import { invalidateIncidents } from "../realtime/Realtime";
import { useLiveStore } from "../realtime/store";
import { IncidentActions } from "./IncidentActions";
import { Snapshot } from "./Snapshot";
import {
  INCIDENT_SOURCES,
  normalizeSourceFilter,
  sourceBadgeKind,
  type IncidentSource,
} from "./details";
import {
  canAct,
  duration,
  formatDate,
  getTimezone,
  incidentDescription,
  incidentParams,
  relativeTime,
} from "./utils";
import type { Assignee, IncidentPage, IncidentRecord } from "./types";
import ru from "../../locales/ru.json";
import "./incidents.css";

const t = ru.incidentUi;
export function SeverityBadge({
  value,
}: {
  value: IncidentRecord["severity"];
}) {
  return (
    <span className={`severity-badge severity-${value}`}>
      {t.severities[value]}
    </span>
  );
}
export function StatusBadge({ value }: { value: IncidentRecord["status"] }) {
  return (
    <span className={`incident-status status-${value}`}>
      {t.statuses[value]}
    </span>
  );
}
export function SourceBadge({ value }: { value: IncidentRecord["source"] }) {
  const kind = sourceBadgeKind(value);
  return kind ? (
    <span className={`source-badge source-${kind}`} title={t.sources[kind]}>
      {t.sourceBadges[kind]}
    </span>
  ) : null;
}
export function IncidentsPage() {
  const [search, setSearch] = useSearchParams();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const [bulkDialog, setBulkDialog] = useState(false);
  const [reason, setReason] = useState("");
  const view = search.get("view") ?? "table";
  const params = incidentParams(search).toString();
  const page = Number(search.get("page") ?? 1) || 1;
  const cameraId = search.get("camera_id") ?? "";
  const sourceFilter = normalizeSourceFilter(search.get("source"));
  const writable = readSession()?.user.role !== "viewer";
  useLiveStore((s) => s.clock);
  const list = useQuery({
    queryKey: ["incidents", "list", params],
    queryFn: ({ signal }) =>
      api<IncidentPage>(`/api/v1/incidents?${params}`, { signal }),
    placeholderData: keepPreviousData,
  });
  const cameras = useQuery({
    queryKey: ["cameras-options"],
    queryFn: () => api<CameraRecord[]>("/api/v1/cameras"),
  });
  const types = useQuery({
    queryKey: ["incident-types"],
    queryFn: () => api<DictionaryItem[]>("/api/v1/settings/incident-types"),
  });
  const assignees = useQuery({
    queryKey: ["assignees"],
    queryFn: () => api<Assignee[]>("/api/v1/incident-assignees"),
  });
  const zones = useQuery({
    queryKey: ["zone-options", cameraId],
    queryFn: () => api<ZoneRecord[]>(`/api/v1/cameras/${cameraId}/zones`),
    enabled: !!cameraId,
  });
  useEffect(() => setSelected(new Set()), [params]);
  function change(key: string, value: string) {
    setSearch(
      (current) => {
        const next = new URLSearchParams(current);
        if (value) next.set(key, value);
        else next.delete(key);
        if (key !== "page") next.delete("page");
        if (key === "camera_id") next.delete("zone_id");
        return next;
      },
      { replace: true },
    );
  }
  function toggle(id: string) {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }
  const selectedRows =
    list.data?.items.filter((row) => selected.has(row.id)) ?? [];
  const allowAck =
    !!selectedRows.length &&
    selectedRows.every(
      (row) =>
        ["new", "acknowledged"].includes(row.status) &&
        canAct(row, "acknowledge"),
    );
  const allowFalse =
    !!selectedRows.length &&
    selectedRows.every((row) => canAct(row, "false_positive"));
  async function bulk(action: "acknowledge" | "false_positive") {
    setPending(true);
    setError("");
    try {
      await api("/api/v1/incidents/bulk", {
        method: "POST",
        body: JSON.stringify({
          ids: [...selected],
          action,
          ...(action === "false_positive" ? { reason } : {}),
        }),
      });
      setSelected(new Set());
      setBulkDialog(false);
      setReason("");
      invalidateIncidents();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  async function exportCsv() {
    setPending(true);
    setError("");
    try {
      const blob = await apiBlob(`/api/v1/incidents/export.csv?${params}`);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "incidents.csv";
      link.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  return (
    <>
      <div className="page-heading">
        <div>
          <span className="eyebrow">{ru.enterprise}</span>
          <h1>{ru.incidents}</h1>
          <p>{t.subtitle}</p>
        </div>
        <Button
          variant="outline"
          disabled={pending}
          onClick={() => void exportCsv()}
        >
          <Download size={17} />
          {t.csv}
        </Button>
      </div>
      <section className="panel incident-filters">
        <div className="filter-search">
          <Search size={18} />
          <input
            aria-label={t.search}
            placeholder={t.search}
            value={search.get("q") ?? ""}
            onChange={(e) => change("q", e.target.value)}
            maxLength={250}
          />
        </div>
        <div className="filter-grid">
          <label>
            {t.from}
            <input
              type="date"
              value={search.get("fromDate") ?? ""}
              onChange={(e) => change("fromDate", e.target.value)}
            />
          </label>
          <label>
            {t.to}
            <input
              type="date"
              value={search.get("toDate") ?? ""}
              onChange={(e) => change("toDate", e.target.value)}
            />
          </label>
          <label>
            {t.camera}
            <select
              value={cameraId}
              onChange={(e) => change("camera_id", e.target.value)}
            >
              <option value="">{t.allCameras}</option>
              {cameras.data?.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            {t.zone}
            <select
              disabled={!cameraId || zones.isPending}
              value={search.get("zone_id") ?? ""}
              onChange={(e) => change("zone_id", e.target.value)}
            >
              <option value="">{t.allZones}</option>
              {zones.data?.map((z) => (
                <option key={z.id} value={z.id}>
                  {z.name}
                </option>
              ))}
            </select>
          </label>
          <label>
            {t.type}
            <select
              value={search.get("type") ?? ""}
              onChange={(e) => change("type", e.target.value)}
            >
              <option value="">{t.allTypes}</option>
              {types.data?.map((kind) => (
                <option key={kind.code} value={kind.code}>
                  {kind.name_ru}
                </option>
              ))}
            </select>
          </label>
          <label>
            {t.severity}
            <select
              value={search.get("severity") ?? ""}
              onChange={(e) => change("severity", e.target.value)}
            >
              <option value="">{t.allSeverities}</option>
              {Object.entries(t.severities).map(([code, label]) => (
                <option value={code} key={code}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label>
            {t.status}
            <select
              value={search.get("status") ?? ""}
              onChange={(e) => change("status", e.target.value)}
            >
              <option value="">{t.allStatuses}</option>
              {Object.entries(t.statuses).map(([code, label]) => (
                <option value={code} key={code}>
                  {label}
                </option>
              ))}
            </select>
          </label>
          <label>
            {t.assignee}
            <select
              value={search.get("assignee_id") ?? ""}
              onChange={(e) => change("assignee_id", e.target.value)}
            >
              <option value="">{t.allAssignees}</option>
              <option value="unassigned">{t.unassigned}</option>
              {assignees.data?.map((a) => (
                <option value={a.id} key={a.id}>
                  {a.full_name}
                </option>
              ))}
            </select>
          </label>
          <label>
            {t.source}
            <select
              value={sourceFilter}
              onChange={(e) => change("source", e.target.value)}
            >
              <option value="">{t.allSources}</option>
              {INCIDENT_SOURCES.map((code) => (
                <option value={code} key={code}>
                  {t.sourceBadges[code]}
                </option>
              ))}
              {sourceFilter.includes(",") && (
                <option value={sourceFilter}>
                  {sourceFilter
                    .split(",")
                    .map((code) => t.sourceBadges[code as IncidentSource])
                    .join(" + ")}
                </option>
              )}
            </select>
          </label>
        </div>
        <div className="filter-bottom">
          <label className="check-label">
            <input
              type="checkbox"
              checked={search.get("only_unacknowledged") === "true"}
              onChange={(e) =>
                change("only_unacknowledged", e.target.checked ? "true" : "")
              }
            />
            {t.onlyNew}
          </label>
          <small>
            {t.timezone}: {getTimezone()}
          </small>
          <Button
            variant="ghost"
            onClick={() => setSearch(new URLSearchParams())}
          >
            {t.reset}
          </Button>
        </div>
        {(cameras.isError ||
          types.isError ||
          assignees.isError ||
          zones.isError) && (
          <div className="error">
            {t.filterError}
            <Button
              variant="outline"
              onClick={() => {
                void cameras.refetch();
                void types.refetch();
                void assignees.refetch();
                if (cameraId) void zones.refetch();
              }}
            >
              {ru.retry}
            </Button>
          </div>
        )}
      </section>
      <div className="incident-list-toolbar">
        <span className="muted">
          {t.found}: {list.data?.total ?? "—"}
          {list.isFetching && !list.isPending ? ` · ${t.updating}` : ""}
        </span>
        <label className="sort-control">
          {t.sort}
          <select
            value={search.get("sort") ?? "-started_at"}
            onChange={(e) => change("sort", e.target.value)}
          >
            {Object.entries(t.sortOptions).map(([key, label]) => (
              <option key={key} value={key}>
                {label}
              </option>
            ))}
          </select>
        </label>
        <div className="action-row">
          <Button
            variant={view === "table" ? "default" : "outline"}
            title={t.table}
            aria-label={t.table}
            onClick={() => change("view", "table")}
          >
            <List size={17} />
          </Button>
          <Button
            variant={view === "cards" ? "default" : "outline"}
            title={t.cards}
            aria-label={t.cards}
            onClick={() => change("view", "cards")}
          >
            <LayoutGrid size={17} />
          </Button>
        </div>
      </div>
      {writable && (
        <div className="bulk-bar">
          <span>
            {t.selected}: {selected.size}
          </span>
          <Button
            variant="outline"
            disabled={!allowAck || pending || list.isPlaceholderData}
            onClick={() => void bulk("acknowledge")}
          >
            {t.acknowledge}
          </Button>
          <Button
            variant="outline"
            disabled={!allowFalse || pending || list.isPlaceholderData}
            onClick={() => setBulkDialog(true)}
          >
            {t.falsePositive}
          </Button>
        </div>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {list.isPending ? (
        <div className="panel skeleton" role="status">
          {ru.loading}
        </div>
      ) : list.isError ? (
        <section className="panel">
          <p className="error" role="alert">
            {list.error.message}
          </p>
          <Button onClick={() => void list.refetch()}>{ru.retry}</Button>
        </section>
      ) : !list.data.items.length ? (
        <section className="panel empty">
          <h2>{t.empty}</h2>
          <p>{t.emptyHint}</p>
          <Button
            variant="outline"
            onClick={() => setSearch(new URLSearchParams())}
          >
            {t.reset}
          </Button>
        </section>
      ) : view === "cards" ? (
        <div className="incident-cards">
          {list.data.items.map((row) => (
            <article className="panel incident-card" key={row.id}>
              <Link to={`/incidents/${row.id}`}>
                <Snapshot incident={row} thumbnail />
              </Link>
              <div className="action-row">
                {writable && (
                  <input
                    type="checkbox"
                    checked={selected.has(row.id)}
                    onChange={() => toggle(row.id)}
                    aria-label={`${t.select} ${row.id}`}
                  />
                )}
                <SeverityBadge value={row.severity} />
                <StatusBadge value={row.status} />
                <SourceBadge value={row.source} />
              </div>
              <Link to={`/incidents/${row.id}`}>
                <h2>{row.type_name}</h2>
              </Link>
              <p>
                {row.camera_name} → {row.zone_name ?? t.noZone}
              </p>
              <p className="muted">{incidentDescription(row)}</p>
              <small title={formatDate(row.started_at)}>
                {relativeTime(row.started_at)}
              </small>
              <IncidentActions incident={row} compact />
            </article>
          ))}
        </div>
      ) : (
        <section className="panel table-scroll incident-table">
          <table>
            <thead>
              <tr>
                {writable && (
                  <th>
                    <input
                      type="checkbox"
                      aria-label={t.selectPage}
                      checked={
                        list.data.items.length > 0 &&
                        list.data.items.every((row) => selected.has(row.id))
                      }
                      onChange={(e) =>
                        setSelected(
                          e.target.checked
                            ? new Set(list.data.items.map((row) => row.id))
                            : new Set(),
                        )
                      }
                    />
                  </th>
                )}
                {[
                  t.snapshot,
                  t.id,
                  t.time,
                  t.type,
                  t.cameraZone,
                  t.details,
                  t.severity,
                  t.status,
                  t.assignee,
                  t.reaction,
                  t.actions,
                ].map((label) => (
                  <th key={label}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {list.data.items.map((row) => (
                <tr key={row.id}>
                  {writable && (
                    <td>
                      <input
                        type="checkbox"
                        checked={selected.has(row.id)}
                        onChange={() => toggle(row.id)}
                        aria-label={`${t.select} ${row.id}`}
                      />
                    </td>
                  )}
                  <td>
                    <Link to={`/incidents/${row.id}`}>
                      <Snapshot incident={row} thumbnail />
                    </Link>
                  </td>
                  <td>
                    <Link
                      className="accent id-short"
                      title={row.id}
                      to={`/incidents/${row.id}`}
                    >
                      {row.id.slice(0, 8)}
                    </Link>
                  </td>
                  <td>
                    <time title={formatDate(row.started_at)}>
                      {relativeTime(row.started_at)}
                    </time>
                    <small>{formatDate(row.started_at)}</small>
                  </td>
                  <td>
                    <Link to={`/incidents/${row.id}`}>{row.type_name}</Link>{" "}
                    <SourceBadge value={row.source} />
                  </td>
                  <td>
                    {row.camera_name}
                    <small>→ {row.zone_name ?? t.noZone}</small>
                  </td>
                  <td>{incidentDescription(row)}</td>
                  <td>
                    <SeverityBadge value={row.severity} />
                  </td>
                  <td>
                    <StatusBadge value={row.status} />
                  </td>
                  <td>{row.assignee_name ?? t.unassigned}</td>
                  <td>{duration(row.reaction_sec)}</td>
                  <td>
                    <IncidentActions incident={row} compact />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
      <div className="pagination">
        <label>
          {t.pageSize}
          <select
            value={search.get("page_size") ?? "20"}
            onChange={(e) => change("page_size", e.target.value)}
          >
            {[20, 50, 100].map((size) => (
              <option key={size} value={size}>
                {size}
              </option>
            ))}
          </select>
        </label>
        <span>
          {t.page} {page} /{" "}
          {Math.max(
            1,
            Math.ceil((list.data?.total ?? 0) / (list.data?.page_size ?? 20)),
          )}
        </span>
        <Button
          variant="outline"
          disabled={page <= 1 || list.isFetching}
          onClick={() => change("page", String(page - 1))}
        >
          {t.previous}
        </Button>
        <Button
          variant="outline"
          disabled={
            !list.data ||
            page * list.data.page_size >= list.data.total ||
            list.isFetching
          }
          onClick={() => change("page", String(page + 1))}
        >
          {t.next}
        </Button>
      </div>
      {bulkDialog && (
        <div className="modal-backdrop">
          <section
            className="modal panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="bulk-title"
          >
            <h2 id="bulk-title">
              {t.falsePositive} · {selected.size}
            </h2>
            <form
              className="edit-form"
              onSubmit={(e) => {
                e.preventDefault();
                void bulk("false_positive");
              }}
            >
              <label>
                {t.reason}
                <textarea
                  required
                  autoFocus
                  maxLength={4000}
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                />
              </label>
              {error && (
                <p className="error" role="alert">
                  {error}
                </p>
              )}
              <div className="action-row">
                <Button disabled={pending}>{t.confirm}</Button>
                <Button
                  variant="outline"
                  type="button"
                  disabled={pending}
                  onClick={() => setBulkDialog(false)}
                >
                  {t.cancel}
                </Button>
              </div>
            </form>
          </section>
        </div>
      )}
    </>
  );
}
