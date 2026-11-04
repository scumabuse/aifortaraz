import { useState } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Radio, Send } from "lucide-react";
import { api, readSession } from "../../api";
import { Button } from "../../components/ui/button";
import { invalidateIncidents } from "../realtime/Realtime";
import { useLiveStore } from "../realtime/store";
import { IncidentActions } from "./IncidentActions";
import { Snapshot } from "./Snapshot";
import { SeverityBadge, StatusBadge } from "./IncidentsPage";
import { canAct, duration, formatDate, incidentDescription } from "./utils";
import type { Assignee, IncidentDetail } from "./types";
import ru from "../../locales/ru.json";

const t = ru.incidentUi;
export function IncidentPage() {
  const { id } = useParams();
  const [text, setText] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  useLiveStore((s) => s.clock);
  const detail = useQuery({
    queryKey: ["incident", id],
    queryFn: ({ signal }) =>
      api<IncidentDetail>(`/api/v1/incidents/${id}`, { signal }),
  });
  const assignees = useQuery({
    queryKey: ["assignees"],
    queryFn: () => api<Assignee[]>("/api/v1/incident-assignees"),
  });
  async function comment(e: React.FormEvent) {
    e.preventDefault();
    setPending(true);
    setError("");
    try {
      await api(`/api/v1/incidents/${id}/comments`, {
        method: "POST",
        body: JSON.stringify({ text }),
      });
      setText("");
      invalidateIncidents();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  if (detail.isPending)
    return (
      <div className="panel skeleton" role="status">
        {ru.loading}
      </div>
    );
  if (detail.isError)
    return (
      <section className="panel">
        <p className="error" role="alert">
          {detail.error.message}
        </p>
        <Button onClick={() => void detail.refetch()}>{ru.retry}</Button>
        <Button asChild variant="ghost">
          <Link to="/incidents">{ru.incidents}</Link>
        </Button>
      </section>
    );
  const incident = detail.data;
  return (
    <>
      <div className="page-heading">
        <div>
          <Link className="accent" to="/incidents">
            ← {ru.incidents}
          </Link>
          <h1>{incident.type_name}</h1>
          <p className="incident-id">{incident.id}</p>
        </div>
        <div className="action-row">
          {["admin", "engineer"].includes(readSession()?.user.role ?? "") && (
            <Button variant="outline" asChild>
              <Link to={`/incidents/${id}/act`}>Составить акт</Link>
            </Button>
          )}
          <SeverityBadge value={incident.severity} />
          <StatusBadge value={incident.status} />
        </div>
      </div>
      <div className="incident-detail-layout">
        <div className="incident-detail-main">
          <section className="panel">
            <Snapshot incident={incident} />
            <div className="snapshot-caption">
              <span>
                {incident.camera_name} → {incident.zone_name ?? t.noZone}
              </span>
              <time>{formatDate(incident.started_at)}</time>
            </div>
            {incident.clip_url && (
              <video
                src={incident.clip_url}
                controls
                preload="metadata"
                className="incident-clip"
              />
            )}
            <p className="incident-description">
              {incidentDescription(incident)}
            </p>
            <IncidentActions incident={incident} />
            <div className="incident-links action-row">
              <Button asChild variant="outline">
                <Link to={`/live/${incident.camera_id}`}>
                  <Radio size={16} />
                  {t.openCamera}
                </Link>
              </Button>
              <Button variant="outline" disabled title={t.actHint}>
                {t.act}
              </Button>
            </div>
          </section>
          <section className="panel comments-panel">
            <h2>
              {t.comments} · {incident.comments.length}
            </h2>
            {incident.comments.length ? (
              <div className="comment-list">
                {incident.comments.map((item) => (
                  <article key={item.id}>
                    <div>
                      <strong>{item.user_name}</strong>
                      <time>{formatDate(item.created_at)}</time>
                    </div>
                    <p>{item.text}</p>
                  </article>
                ))}
              </div>
            ) : (
              <p className="muted empty-hint">{t.noComments}</p>
            )}
            {canAct(incident, "comment") && (
              <form className="edit-form" onSubmit={comment}>
                <label>
                  {t.addComment}
                  <textarea
                    required
                    maxLength={4000}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                  />
                </label>
                {error && (
                  <p className="error" role="alert">
                    {error}
                  </p>
                )}
                <Button disabled={pending}>
                  <Send size={16} />
                  {t.send}
                </Button>
              </form>
            )}
          </section>
        </div>
        <aside>
          <section className="panel metadata-panel">
            <h2>{t.metadata}</h2>
            <dl className="metadata">
              {[
                [t.created, formatDate(incident.created_at)],
                [t.started, formatDate(incident.started_at)],
                [t.ended, formatDate(incident.ended_at)],
                [t.duration, duration(incident.duration_sec)],
                [t.area, incident.area_name],
                [t.camera, incident.camera_name],
                [t.zone, incident.zone_name ?? t.noZone],
                [
                  t.track,
                  incident.track_id === null ? "—" : `#${incident.track_id}`,
                ],
                [t.confidence, `${Math.round(incident.confidence * 100)}%`],
                [
                  t.source,
                  t.sources[incident.source as keyof typeof t.sources] ??
                    incident.source,
                ],
                [t.assignee, incident.assignee_name ?? t.unassigned],
                [t.reaction, duration(incident.reaction_sec)],
              ].map(([key, value]) => (
                <div key={key}>
                  <dt>{key}</dt>
                  <dd>{value}</dd>
                </div>
              ))}
            </dl>
            {incident.resolution_note && (
              <div className="resolution-note">
                <strong>{t.measure}</strong>
                <p>{incident.resolution_note}</p>
              </div>
            )}
            {incident.false_positive_reason && (
              <div className="resolution-note">
                <strong>{t.reason}</strong>
                <p>{incident.false_positive_reason}</p>
              </div>
            )}
          </section>
          <section className="panel timeline-panel">
            <h2>{t.timeline}</h2>
            <ol className="timeline">
              {incident.timeline.map((event) => (
                <li key={event.id}>
                  <span className={`timeline-dot event-${event.event_type}`} />
                  <div>
                    <strong>
                      {t.events[event.event_type as keyof typeof t.events] ??
                        event.event_type}
                    </strong>
                    <small>
                      {formatDate(event.created_at)}
                      {event.user_name ? ` · ${event.user_name}` : ""}
                    </small>
                    {event.payload.channel && (
                      <p>
                        {t.channels[
                          event.payload.channel as keyof typeof t.channels
                        ] ?? event.payload.channel}
                      </p>
                    )}
                    {event.payload.user_id && (
                      <p>
                        {assignees.data?.find(
                          (a) => a.id === event.payload.user_id,
                        )?.full_name ?? t.assigned}
                      </p>
                    )}
                    {event.event_type === "assigned" &&
                      event.payload.user_id === null && <p>{t.unassigned}</p>}
                    {(event.payload.note ||
                      event.payload.reason ||
                      event.payload.text) && (
                      <p>
                        {event.payload.note ??
                          event.payload.reason ??
                          event.payload.text}
                      </p>
                    )}
                  </div>
                </li>
              ))}
            </ol>
          </section>
        </aside>
      </div>
      <div className="incident-navigation">
        <Button
          asChild={!!incident.previous_id}
          variant="outline"
          disabled={!incident.previous_id}
        >
          {incident.previous_id ? (
            <Link to={`/incidents/${incident.previous_id}`}>
              <ChevronLeft size={16} />
              {t.previousIncident}
            </Link>
          ) : (
            <span>{t.previousIncident}</span>
          )}
        </Button>
        <Button
          asChild={!!incident.next_id}
          variant="outline"
          disabled={!incident.next_id}
        >
          {incident.next_id ? (
            <Link to={`/incidents/${incident.next_id}`}>
              {t.nextIncident}
              <ChevronRight size={16} />
            </Link>
          ) : (
            <span>{t.nextIncident}</span>
          )}
        </Button>
      </div>
    </>
  );
}
