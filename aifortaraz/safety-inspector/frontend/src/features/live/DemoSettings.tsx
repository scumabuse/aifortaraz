import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api, readSession } from "../../api";
import { queryClient } from "../../lib/query";
import { Button } from "../../components/ui/button";
import { invalidateIncidents } from "../realtime/Realtime";
import type { DemoStatus } from "./types";
import ru from "../../locales/ru.json";
import "./live.css";

const t = ru.liveUi;
export function useDemo() {
  return useQuery({
    queryKey: ["demo"],
    queryFn: () => api<DemoStatus>("/api/v1/demo"),
    refetchInterval: 5000,
  });
}
export function DemoBadge() {
  const demo = useDemo();
  return demo.data?.enabled ? (
    <span className="badge demo">{ru.demo}</span>
  ) : null;
}
export function DemoSettings() {
  const demo = useDemo();
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  async function configure(enabled: boolean, rate: number) {
    setPending(true);
    setError("");
    try {
      await api("/api/v1/demo", {
        method: "PUT",
        body: JSON.stringify({ enabled, rate }),
      });
      await queryClient.invalidateQueries({ queryKey: ["demo"] });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  async function reset() {
    if (!confirm(t.resetConfirm)) return;
    setPending(true);
    setError("");
    try {
      await api("/api/v1/demo/reset", { method: "POST" });
      invalidateIncidents();
      void queryClient.invalidateQueries({ queryKey: ["demo"] });
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  if (demo.isPending) return <div className="skeleton">{ru.loading}</div>;
  if (demo.isError)
    return (
      <div>
        <p className="error">{ru.error}</p>
        <Button onClick={() => void demo.refetch()}>{ru.retry}</Button>
      </div>
    );
  return (
    <div className="demo-settings">
      <h2>{t.demoSettings}</h2>
      <p className="muted">{t.demoDescription}</p>
      <label className="check-label">
        <input
          type="checkbox"
          checked={demo.data.enabled}
          disabled={pending || demo.data.seeding}
          onChange={(e) => void configure(e.target.checked, demo.data.rate)}
        />
        {t.enableDemo}
      </label>
      <label>
        {t.intensity}
        <select
          value={demo.data.rate}
          disabled={pending || demo.data.seeding}
          onChange={(e) =>
            void configure(demo.data.enabled, Number(e.target.value))
          }
        >
          {[0.1, 0.25, 0.5, 1, 2, 3, 5, 10].map((rate) => (
            <option key={rate} value={rate}>
              ×{rate}
            </option>
          ))}
        </select>
      </label>
      <p className="muted">{t.intensityHint}</p>
      {demo.data.seeding && (
        <p role="status" className="notice">
          {t.seeding}
        </p>
      )}
      {(error || demo.data.error) && (
        <p role="alert" className="error">
          {error || demo.data.error}
        </p>
      )}
      {readSession()?.user.role === "admin" && (
        <Button
          variant="outline"
          disabled={pending || demo.data.seeding || !demo.data.enabled}
          onClick={() => void reset()}
        >
          {pending ? ru.loading : t.resetDemo}
        </Button>
      )}
    </div>
  );
}
