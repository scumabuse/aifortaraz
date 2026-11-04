import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../api";
import { Button } from "../../components/ui/button";
import { invalidateIncidents } from "../realtime/Realtime";
import { canAct } from "./utils";
import type { Assignee, IncidentRecord } from "./types";
import ru from "../../locales/ru.json";

const t = ru.incidentUi;
type Action =
  "acknowledge" | "assign" | "resolve" | "false-positive" | "reopen";
export function IncidentActions({
  incident,
  onChanged,
  compact = false,
}: {
  incident: IncidentRecord;
  onChanged?: () => void;
  compact?: boolean;
}) {
  const [dialog, setDialog] = useState<Action | null>(null);
  const [text, setText] = useState("");
  const [assignee, setAssignee] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const options = useQuery({
    queryKey: ["assignees"],
    queryFn: () => api<Assignee[]>("/api/v1/incident-assignees"),
    enabled: dialog === "assign",
  });
  async function execute(action: Action) {
    setPending(true);
    setError("");
    const payload =
      action === "resolve"
        ? { note: text }
        : action === "false-positive"
          ? { reason: text }
          : action === "assign"
            ? { user_id: assignee || null }
            : undefined;
    try {
      await api(`/api/v1/incidents/${incident.id}/${action}`, {
        method: "POST",
        ...(payload ? { body: JSON.stringify(payload) } : {}),
      });
      setDialog(null);
      setText("");
      invalidateIncidents();
      onChanged?.();
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  function show(action: Action) {
    setDialog(action);
    setText("");
    setAssignee(incident.assignee_id ?? "");
    setError("");
  }
  return (
    <>
      <div className="action-row incident-actions">
        {incident.status === "new" && canAct(incident, "acknowledge") && (
          <Button
            disabled={pending}
            onClick={() => void execute("acknowledge")}
          >
            {t.acknowledge}
          </Button>
        )}
        {!compact && (
          <>
            {canAct(incident, "assign") && (
              <Button
                variant="outline"
                disabled={pending}
                onClick={() => show("assign")}
              >
                {t.assign}
              </Button>
            )}
            {incident.status === "acknowledged" &&
              canAct(incident, "resolve") && (
                <Button disabled={pending} onClick={() => show("resolve")}>
                  {t.resolve}
                </Button>
              )}
            {incident.status !== "false_positive" &&
              canAct(incident, "false_positive") && (
                <Button
                  variant="outline"
                  disabled={pending}
                  onClick={() => show("false-positive")}
                >
                  {t.falsePositive}
                </Button>
              )}
            {["resolved", "false_positive"].includes(incident.status) &&
              canAct(incident, "reopen") && (
                <Button
                  variant="outline"
                  disabled={pending}
                  onClick={() => void execute("reopen")}
                >
                  {t.reopen}
                </Button>
              )}
          </>
        )}
      </div>
      {error && !dialog && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      {dialog && (
        <div className="modal-backdrop">
          <section
            className="modal panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby={`action-${incident.id}`}
          >
            <h2 id={`action-${incident.id}`}>
              {dialog === "assign"
                ? t.assign
                : dialog === "resolve"
                  ? t.resolve
                  : t.falsePositive}
            </h2>
            <form
              className="edit-form"
              onSubmit={(e) => {
                e.preventDefault();
                void execute(dialog);
              }}
            >
              {dialog === "assign" ? (
                <label>
                  {t.assignee}
                  <select
                    value={assignee}
                    onChange={(e) => setAssignee(e.target.value)}
                    disabled={options.isPending || options.isError}
                  >
                    <option value="">{t.unassigned}</option>
                    {options.data
                      ?.filter(
                        (u) =>
                          u.role !== "master" || u.area_id === incident.area_id,
                      )
                      .map((u) => (
                        <option value={u.id} key={u.id}>
                          {u.full_name}
                        </option>
                      ))}
                  </select>
                </label>
              ) : (
                <label>
                  {dialog === "resolve" ? t.measure : t.reason}
                  <textarea
                    autoFocus
                    required
                    minLength={1}
                    maxLength={4000}
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                  />
                </label>
              )}
              {options.isError && dialog === "assign" && (
                <p className="error">
                  {ru.error}
                  <Button
                    type="button"
                    variant="outline"
                    onClick={() => void options.refetch()}
                  >
                    {ru.retry}
                  </Button>
                </p>
              )}
              {error && (
                <p className="error" role="alert">
                  {error}
                </p>
              )}
              <div className="action-row">
                <Button
                  disabled={
                    pending ||
                    (dialog === "assign" &&
                      (options.isPending || options.isError))
                  }
                >
                  {pending ? ru.loading : t.confirm}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  disabled={pending}
                  onClick={() => setDialog(null)}
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
