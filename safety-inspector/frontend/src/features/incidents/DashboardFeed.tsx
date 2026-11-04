import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../api";
import { Button } from "../../components/ui/button";
import { useLiveStore } from "../realtime/store";
import { IncidentActions } from "./IncidentActions";
import { Snapshot } from "./Snapshot";
import { SeverityBadge, SourceBadge, StatusBadge } from "./IncidentsPage";
import { relativeTime } from "./utils";
import type { IncidentPage } from "./types";
import ru from "../../locales/ru.json";
const t = ru.incidentUi;
export function DashboardFeed() {
  useLiveStore((s) => s.clock);
  const feed = useQuery({
    queryKey: ["incidents", "latest"],
    queryFn: () =>
      api<IncidentPage>("/api/v1/incidents?page_size=10&sort=-created_at"),
  });
  return (
    <div className="dashboard-live">
      <section className="panel feed-panel">
        <div className="panel-heading">
          <h2>{t.latest}</h2>
          <Link className="accent" to="/incidents">
            {t.allIncidents} →
          </Link>
        </div>
        {feed.isPending ? (
          <div className="skeleton">{ru.loading}</div>
        ) : feed.isError ? (
          <div>
            <p className="error">{ru.error}</p>
            <Button onClick={() => void feed.refetch()}>{ru.retry}</Button>
          </div>
        ) : !feed.data.items.length ? (
          <p className="empty-hint muted">{t.emptyHint}</p>
        ) : (
          <div className="incident-feed">
            {feed.data.items.map((incident) => (
              <div className="feed-row" key={incident.id}>
                <Link to={`/incidents/${incident.id}`}>
                  <Snapshot incident={incident} thumbnail />
                </Link>
                <div>
                  <Link to={`/incidents/${incident.id}`}>
                    <strong>{incident.type_name}</strong>
                  </Link>{" "}
                  <SourceBadge value={incident.source} />
                  <p>
                    {incident.camera_name} → {incident.zone_name ?? t.noZone}
                  </p>
                  <small>{relativeTime(incident.created_at)}</small>
                </div>
                <SeverityBadge value={incident.severity} />
                <StatusBadge value={incident.status} />
                <IncidentActions incident={incident} compact />
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}
