import { useEffect, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import {
  Stage,
  Layer,
  Group,
  Rect,
  Line,
  Circle,
  Text,
  Image as KonvaImage,
} from "react-konva";
import Konva from "konva";
import {
  Download,
  Eye,
  Plus,
  Redo2,
  Save,
  Trash2,
  Undo2,
  ZoomOut,
} from "lucide-react";
import { api, type DictionaryItem } from "../../api";
import { Button } from "../../components/ui/button";
import { canEdit } from "../cameras/Cameras";
import { isPpeDetectable } from "../cameras/ppe";
import {
  blankRules,
  type CameraRecord,
  type Config,
  type Point,
  type Rules,
  type ZoneRecord,
  type ZoneType,
} from "../cameras/types";
import ru from "../../locales/ru.json";

const t = ru.zoneUi;
interface DocumentState {
  zones: ZoneRecord[];
  draft: Point[];
}
const clamp = (n: number) => Math.max(0, Math.min(1, n));
function serializeZones(zones: ZoneRecord[]) {
  return zones.map((z, i) => ({
    id: z.id,
    name: z.name,
    color: z.color,
    zone_type: z.zone_type,
    polygon: z.polygon,
    rules: z.rules,
    enabled: z.enabled,
    sort_order: i,
  }));
}
export function ZoneEditor() {
  const { id } = useParams();
  const [search] = useSearchParams();
  const [camera, setCamera] = useState<CameraRecord | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [ppe, setPpe] = useState<DictionaryItem[]>([]);
  const [responsibles, setResponsibles] = useState<
    { id: string; full_name: string }[]
  >([]);
  const [history, setHistory] = useState<DocumentState[]>([
    { zones: [], draft: [] },
  ]);
  const [cursor, setCursor] = useState(0);
  const [selected, setSelected] = useState<string | null>(null);
  const [vertex, setVertex] = useState<number | null>(null);
  const [drawing, setDrawing] = useState(false);
  const [aiView, setAiView] = useState(false);
  const [baseline, setBaseline] = useState("[]");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const [notice, setNotice] = useState("");
  const [attempt, setAttempt] = useState(0);
  const [width, setWidth] = useState(800);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [view, setView] = useState({ scale: 1, x: 0, y: 0 });
  const container = useRef<HTMLDivElement>(null);
  const group = useRef<Konva.Group>(null);
  const doc = history[cursor];
  const zone = doc.zones.find((z) => z.id === selected);
  const editable = canEdit();
  const height =
    width * (image ? image.naturalHeight / image.naturalWidth : 9 / 16);
  const dirty =
    JSON.stringify(serializeZones(doc.zones)) !== baseline ||
    doc.draft.length > 0;
  function commit(next: DocumentState) {
    const trimmed = history.slice(0, cursor + 1);
    trimmed.push(structuredClone(next));
    if (trimmed.length > 100) trimmed.shift();
    setHistory(trimmed);
    setCursor(trimmed.length - 1);
    setNotice("");
  }
  function patchZone(patch: Partial<ZoneRecord>) {
    if (!zone || !editable) return;
    commit({
      ...doc,
      zones: doc.zones.map((z) => (z.id === zone.id ? { ...z, ...patch } : z)),
    });
  }
  function patchRule<K extends keyof Rules>(key: K, value: Rules[K]) {
    if (zone) patchZone({ rules: { ...zone.rules, [key]: value } });
  }
  function undo() {
    const next = Math.max(0, cursor - 1);
    setCursor(next);
    setDrawing(history[next].draft.length > 0);
    setVertex(null);
  }
  function redo() {
    const next = Math.min(history.length - 1, cursor + 1);
    setCursor(next);
    setDrawing(history[next].draft.length > 0);
    setVertex(null);
  }
  useEffect(() => {
    let active = true;
    setLoading(true);
    setError("");
    Promise.all([
      api<CameraRecord>(`/api/v1/cameras/${id}`),
      api<ZoneRecord[]>(`/api/v1/cameras/${id}/zones`),
      api<Config>(`/api/v1/cameras/${id}/config`),
      api<DictionaryItem[]>("/api/v1/settings/ppe-types"),
      editable
        ? api<{ id: string; full_name: string }[]>("/api/v1/zone-responsibles")
        : Promise.resolve([]),
    ])
      .then(([c, z, cfg, p, r]) => {
        if (!active) return;
        setCamera(c);
        setConfig(cfg);
        setPpe(p);
        setResponsibles(r);
        setHistory([{ zones: z, draft: [] }]);
        setCursor(0);
        setBaseline(JSON.stringify(serializeZones(z)));
        setSelected(search.get("zone") ?? z[0]?.id ?? null);
        setDrawing(false);
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
  }, [id, attempt, editable, search]);
  useEffect(() => {
    const status = (event: Event) => {
      const next = (event as CustomEvent).detail as Partial<CameraRecord> & {
        camera_id: string;
      };
      if (next.camera_id === id && "supported_ppe" in next)
        setCamera((current) =>
          current ? { ...current, supported_ppe: next.supported_ppe } : current,
        );
    };
    window.addEventListener("camera-status", status);
    return () => window.removeEventListener("camera-status", status);
  }, [id]);
  useEffect(() => {
    if (!container.current) return;
    const observer = new ResizeObserver((entries) =>
      setWidth(Math.max(200, entries[0].contentRect.width)),
    );
    observer.observe(container.current);
    return () => observer.disconnect();
  }, [loading]);
  useEffect(() => {
    setImage(null);
    if (!camera?.preview_url) return;
    let active = true;
    const img = new window.Image();
    img.onload = () => {
      if (active) setImage(img);
    };
    img.onerror = () => {
      if (active) setImage(null);
    };
    img.src = camera.preview_url;
    return () => {
      active = false;
    };
  }, [camera?.preview_url]);
  useEffect(() => {
    if (!dirty) return;
    function leave(e: BeforeUnloadEvent) {
      e.preventDefault();
      e.returnValue = "";
    }
    window.addEventListener("beforeunload", leave);
    return () => window.removeEventListener("beforeunload", leave);
  }, [dirty]);
  function finish() {
    if (doc.draft.length < 3) return;
    const next: ZoneRecord = {
      id: crypto.randomUUID(),
      name: `${t.defaultName} ${doc.zones.length + 1}`,
      color: "#65d9b2",
      zone_type: "ppe",
      polygon: doc.draft,
      rules: blankRules(),
      enabled: true,
      sort_order: doc.zones.length,
    };
    commit({ zones: [...doc.zones, next], draft: [] });
    setSelected(next.id);
    setDrawing(false);
    setVertex(null);
  }
  function addPoint() {
    if (!drawing || !editable) return;
    const pointer = group.current?.getRelativePointerPosition();
    if (!pointer) return;
    if (
      pointer.x < 0 ||
      pointer.y < 0 ||
      pointer.x > width ||
      pointer.y > height
    )
      return;
    const point: Point = [clamp(pointer.x / width), clamp(pointer.y / height)];
    if (
      doc.draft.length &&
      Math.hypot(
        (point[0] - doc.draft[0][0]) * width,
        (point[1] - doc.draft[0][1]) * height,
      ) *
        view.scale <
        12
    ) {
      finish();
      return;
    }
    commit({ ...doc, draft: [...doc.draft, point] });
  }
  function removeVertex() {
    if (!zone || vertex === null || zone.polygon.length <= 3) return;
    patchZone({ polygon: zone.polygon.filter((_, i) => i !== vertex) });
    setVertex(null);
  }
  async function save() {
    setPending(true);
    setError("");
    try {
      await api(`/api/v1/cameras/${id}/zones`, {
        method: "PUT",
        body: JSON.stringify(serializeZones(doc.zones)),
      });
      const cfg = await api<Config>(`/api/v1/cameras/${id}/config`);
      setConfig(cfg);
      setBaseline(JSON.stringify(serializeZones(doc.zones)));
      setNotice(`${t.saved} · ${t.version} ${cfg.version}`);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setPending(false);
    }
  }
  function cancel() {
    if (dirty && !confirm(t.cancelConfirm)) return;
    setAttempt((x) => x + 1);
    setView({ scale: 1, x: 0, y: 0 });
  }
  function exportConfig() {
    if (!config) return;
    const output: Config = {
      ...config,
      zones: doc.zones
        .filter((z) => z.enabled)
        .map((z) => {
          const {
            regulation_ref: _regulation,
            responsible_user_id: _responsible,
            ...rules
          } = z.rules;
          void _regulation;
          void _responsible;
          return {
            id: z.id,
            name: z.name,
            type: z.zone_type,
            polygon: z.polygon,
            rules: {
              ...rules,
              confirm_frames:
                rules.confirm_frames ?? config.thresholds.confirm_frames,
              cooldown_sec:
                rules.cooldown_sec ?? config.thresholds.cooldown_sec,
            },
          };
        }),
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(output, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = `camera-${id}-config.json`;
    link.click();
    URL.revokeObjectURL(url);
  }
  if (loading)
    return (
      <div className="panel skeleton" role="status">
        {ru.loading}
      </div>
    );
  if (!camera)
    return (
      <div className="panel">
        <p className="error" role="alert">
          {error}
        </p>
        <Button onClick={() => setAttempt((x) => x + 1)}>{ru.retry}</Button>
      </div>
    );
  return (
    <>
      <div className="page-heading">
        <div>
          <Link
            className="accent"
            to={`/cameras/${id}`}
            onClick={(e) => {
              if (dirty && !confirm(t.cancelConfirm)) e.preventDefault();
            }}
          >
            ← {camera.name}
          </Link>
          <h1>{ru.zones}</h1>
          <p>
            {t.subtitle} · {t.version} {config?.version}
            {dirty ? ` · ${t.unsaved}` : ""}
          </p>
        </div>
        <div className="action-row">
          <Button
            variant="outline"
            onClick={exportConfig}
            disabled={!!doc.draft.length}
          >
            <Download size={17} />
            {t.export}
          </Button>
          {editable && (
            <Button
              disabled={pending || drawing || doc.draft.length > 0 || !dirty}
              onClick={() => void save()}
            >
              <Save size={17} />
              {pending ? ru.loading : t.save}
            </Button>
          )}
        </div>
      </div>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
      {notice && (
        <p className="notice" role="status">
          {notice}
        </p>
      )}
      <div className="zone-editor">
        <aside className="panel zone-list">
          <div className="panel-heading">
            <h2>{t.list}</h2>
            <span className="tag">{doc.zones.length}</span>
          </div>
          {editable && (
            <Button
              className="w-full"
              variant="outline"
              disabled={drawing || pending}
              onClick={() => {
                setDrawing(true);
                setSelected(null);
                setVertex(null);
              }}
            >
              <Plus size={16} />
              {t.draw}
            </Button>
          )}
          <div className="zone-items">
            {doc.zones.map((z) => (
              <button
                className={z.id === selected ? "selected" : ""}
                key={z.id}
                onClick={() => {
                  setSelected(z.id);
                  setVertex(null);
                }}
              >
                <span className="zone-dot" style={{ background: z.color }} />
                <div>
                  {z.name}
                  <small>
                    {t.types[z.zone_type]} ·{" "}
                    {z.enabled ? t.enabled : t.disabled}
                  </small>
                </div>
              </button>
            ))}
          </div>
          {!doc.zones.length && <p className="muted">{t.empty}</p>}
          {!editable && <p className="muted">{t.readOnly}</p>}
        </aside>
        <div className="zone-center">
          <div className="canvas-toolbar">
            <Button
              variant="ghost"
              disabled={!cursor || pending}
              onClick={undo}
              aria-label={t.undo}
              title={t.undo}
            >
              <Undo2 size={17} />
            </Button>
            <Button
              variant="ghost"
              disabled={cursor === history.length - 1 || pending}
              onClick={redo}
              aria-label={t.redo}
              title={t.redo}
            >
              <Redo2 size={17} />
            </Button>
            <span className="muted">{Math.round(view.scale * 100)}%</span>
            <Button
              variant="ghost"
              onClick={() => setView({ scale: 1, x: 0, y: 0 })}
              title={t.resetZoom}
              aria-label={t.resetZoom}
            >
              <ZoomOut size={17} />
            </Button>
            <Button
              variant={aiView ? "default" : "outline"}
              onClick={() => setAiView((v) => !v)}
            >
              <Eye size={17} />
              {t.aiView}
            </Button>
          </div>
          <div
            className={`canvas-frame ${drawing ? "drawing" : ""}`}
            ref={container}
            tabIndex={0}
            role="application"
            aria-label={t.canvasLabel}
            onKeyDown={(e) => {
              if (e.ctrlKey || e.metaKey) {
                if (e.key === "z") {
                  e.preventDefault();
                  if (e.shiftKey) redo();
                  else undo();
                }
                if (e.key === "y") {
                  e.preventDefault();
                  redo();
                }
              } else if (e.key === "Delete") {
                e.preventDefault();
                removeVertex();
              } else if (e.key === "Enter" && drawing) {
                finish();
              } else if (e.key === "Escape" && drawing) {
                commit({ ...doc, draft: [] });
                setDrawing(false);
              }
            }}
          >
            <Stage
              width={width}
              height={height}
              onClick={addPoint}
              onTap={addPoint}
              onWheel={(e) => {
                e.evt.preventDefault();
                const pointer = e.target.getStage()?.getPointerPosition();
                if (!pointer) return;
                const scale = Math.max(
                  1,
                  Math.min(5, view.scale * (e.evt.deltaY > 0 ? 0.9 : 1.1)),
                );
                setView({
                  scale,
                  x: pointer.x - ((pointer.x - view.x) / view.scale) * scale,
                  y: pointer.y - ((pointer.y - view.y) / view.scale) * scale,
                });
              }}
            >
              <Layer>
                <Group
                  ref={group}
                  x={view.x}
                  y={view.y}
                  scaleX={view.scale}
                  scaleY={view.scale}
                >
                  <Rect width={width} height={height} fill="#202b34" />
                  {image ? (
                    <KonvaImage image={image} width={width} height={height} />
                  ) : (
                    <>
                      {Array.from({ length: 21 }, (_, i) => (
                        <Line
                          key={`x${i}`}
                          points={[
                            (i * width) / 20,
                            0,
                            (i * width) / 20,
                            height,
                          ]}
                          stroke="#31404c"
                          strokeWidth={0.6}
                        />
                      ))}
                      {Array.from({ length: 13 }, (_, i) => (
                        <Line
                          key={`y${i}`}
                          points={[
                            0,
                            (i * height) / 12,
                            width,
                            (i * height) / 12,
                          ]}
                          stroke="#31404c"
                          strokeWidth={0.6}
                        />
                      ))}
                      <Text
                        x={18}
                        y={18}
                        text={t.noFrame}
                        fill="#8fa0ab"
                        fontSize={14}
                      />
                    </>
                  )}
                  {doc.zones.map((z) => (
                    <Group key={z.id} opacity={z.enabled ? 1 : 0.35}>
                      <Line
                        points={z.polygon.flatMap(([x, y]) => [
                          x * width,
                          y * height,
                        ])}
                        closed
                        fill={`${z.color}24`}
                        stroke={z.color}
                        strokeWidth={(z.id === selected ? 3 : 1.5) / view.scale}
                        onClick={(e) => {
                          if (!drawing) {
                            e.cancelBubble = true;
                            setSelected(z.id);
                            setVertex(null);
                          }
                        }}
                        onTap={(e) => {
                          if (!drawing) {
                            e.cancelBubble = true;
                            setSelected(z.id);
                            setVertex(null);
                          }
                        }}
                      />
                      <Text
                        x={z.polygon[0][0] * width + 8}
                        y={z.polygon[0][1] * height + 8}
                        text={
                          aiView
                            ? `${z.name}\n${t.types[z.zone_type]} · ${z.rules.required_ppe.map((code) => ppe.find((p) => p.code === code)?.name_ru ?? code).join(", ")}\n${t.frames}: ${z.rules.confirm_frames ?? config?.thresholds.confirm_frames} · ${t.cooldown}: ${z.rules.cooldown_sec ?? config?.thresholds.cooldown_sec}`
                            : z.name
                        }
                        fill={z.color}
                        fontSize={12 / view.scale}
                        lineHeight={1.5}
                        listening={false}
                      />
                      {z.id === selected &&
                        !drawing &&
                        editable &&
                        !aiView &&
                        z.polygon.map(([x, y], i) => (
                          <Circle
                            key={i}
                            x={x * width}
                            y={y * height}
                            radius={6 / view.scale}
                            fill={vertex === i ? "#ffffff" : z.color}
                            stroke="#10151b"
                            strokeWidth={2 / view.scale}
                            draggable={!pending}
                            onClick={(e) => {
                              e.cancelBubble = true;
                              setVertex(i);
                            }}
                            onTap={(e) => {
                              e.cancelBubble = true;
                              setVertex(i);
                            }}
                            onContextMenu={(e) => {
                              e.evt.preventDefault();
                              e.cancelBubble = true;
                              if (z.polygon.length > 3) {
                                patchZone({
                                  polygon: z.polygon.filter(
                                    (_, index) => index !== i,
                                  ),
                                });
                                setVertex(null);
                              }
                            }}
                            dragBoundFunc={(position) => ({
                              x: Math.max(
                                view.x,
                                Math.min(
                                  view.x + width * view.scale,
                                  position.x,
                                ),
                              ),
                              y: Math.max(
                                view.y,
                                Math.min(
                                  view.y + height * view.scale,
                                  position.y,
                                ),
                              ),
                            })}
                            onDragStart={() => setVertex(i)}
                            onDragEnd={(e) => {
                              const polygon = z.polygon.map((p, index) =>
                                index === i
                                  ? ([
                                      clamp(e.target.x() / width),
                                      clamp(e.target.y() / height),
                                    ] as Point)
                                  : p,
                              );
                              patchZone({ polygon });
                            }}
                          />
                        ))}
                    </Group>
                  ))}
                  {doc.draft.length > 0 && (
                    <>
                      <Line
                        points={doc.draft.flatMap(([x, y]) => [
                          x * width,
                          y * height,
                        ])}
                        stroke="#eab54d"
                        strokeWidth={2 / view.scale}
                        dash={[5 / view.scale, 4 / view.scale]}
                      />
                      {doc.draft.map(([x, y], i) => (
                        <Circle
                          key={i}
                          x={x * width}
                          y={y * height}
                          radius={(i === 0 ? 8 : 4) / view.scale}
                          fill={i === 0 ? "#eab54d" : "#fff"}
                          onClick={(e) => {
                            if (i === 0) {
                              e.cancelBubble = true;
                              finish();
                            }
                          }}
                          onTap={(e) => {
                            if (i === 0) {
                              e.cancelBubble = true;
                              finish();
                            }
                          }}
                        />
                      ))}
                    </>
                  )}
                </Group>
              </Layer>
            </Stage>
          </div>
          <div className="canvas-help">
            <p>{drawing ? t.drawHint : t.editHint}</p>
            <div className="action-row">
              {drawing && (
                <>
                  <Button
                    variant="outline"
                    disabled={doc.draft.length < 3}
                    onClick={finish}
                  >
                    {t.finish}
                  </Button>
                  <Button
                    variant="ghost"
                    onClick={() => {
                      commit({ ...doc, draft: [] });
                      setDrawing(false);
                    }}
                  >
                    {t.cancelDraw}
                  </Button>
                </>
              )}
              {editable && !drawing && (
                <Button
                  variant="outline"
                  disabled={
                    vertex === null || !zone || zone.polygon.length <= 3
                  }
                  onClick={removeVertex}
                >
                  {t.removeVertex}
                </Button>
              )}
            </div>
          </div>
          {editable && (
            <div className="action-row">
              <Button variant="outline" onClick={cancel} disabled={pending}>
                {t.cancel}
              </Button>
            </div>
          )}
        </div>
        <aside className="panel rule-panel">
          {zone ? (
            <>
              <h2>{t.rules}</h2>
              <fieldset className="edit-form" disabled={!editable || pending}>
                <label>
                  {t.name}
                  <input
                    value={zone.name}
                    maxLength={200}
                    onChange={(e) => patchZone({ name: e.target.value })}
                  />
                </label>
                <div className="form-grid">
                  <label>
                    {t.type}
                    <select
                      value={zone.zone_type}
                      onChange={(e) =>
                        patchZone({ zone_type: e.target.value as ZoneType })
                      }
                    >
                      {Object.entries(t.types).map(([code, name]) => (
                        <option key={code} value={code}>
                          {name}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label>
                    {t.color}
                    <input
                      type="color"
                      value={zone.color}
                      onChange={(e) => patchZone({ color: e.target.value })}
                    />
                  </label>
                </div>
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={zone.enabled}
                    onChange={(e) => patchZone({ enabled: e.target.checked })}
                  />
                  {t.enabled}
                </label>
                <h3>{t.requiredPpe}</h3>
                <div className="ppe-checks">
                  {ppe.map((p) => {
                    const detectable = isPpeDetectable(
                      p.code!,
                      camera.supported_ppe,
                    );
                    return (
                      <label
                        key={p.code}
                        className={
                          detectable
                            ? "check-label"
                            : "check-label ppe-undetectable"
                        }
                      >
                        <input
                          type="checkbox"
                          checked={zone.rules.required_ppe.includes(p.code!)}
                          onChange={(e) =>
                            patchRule(
                              "required_ppe",
                              e.target.checked
                                ? [...zone.rules.required_ppe, p.code!]
                                : zone.rules.required_ppe.filter(
                                    (code) => code !== p.code,
                                  ),
                            )
                          }
                        />
                        <span>
                          {p.name_ru}
                          {!detectable && <small>{t.notDetected}</small>}
                        </span>
                      </label>
                    );
                  })}
                </div>
                <label className="check-label">
                  <input
                    type="checkbox"
                    checked={zone.rules.entry_forbidden}
                    onChange={(e) =>
                      patchRule("entry_forbidden", e.target.checked)
                    }
                  />
                  {t.entryForbidden}
                </label>
                {(
                  [
                    "max_persons",
                    "min_vehicle_distance_m",
                    "confirm_frames",
                    "cooldown_sec",
                  ] as const
                ).map((key) => (
                  <label key={key}>
                    {t.ruleLabels[key]}
                    <input
                      type="number"
                      min={
                        key === "cooldown_sec"
                          ? 0
                          : key === "min_vehicle_distance_m"
                            ? 0.1
                            : 1
                      }
                      max={
                        key === "cooldown_sec"
                          ? 86400
                          : key === "min_vehicle_distance_m"
                            ? 1000
                            : 10000
                      }
                      step={key === "min_vehicle_distance_m" ? 0.1 : 1}
                      placeholder={
                        key === "confirm_frames"
                          ? `${t.defaultValue}: ${config?.thresholds.confirm_frames}`
                          : key === "cooldown_sec"
                            ? `${t.defaultValue}: ${config?.thresholds.cooldown_sec}`
                            : t.notSet
                      }
                      value={zone.rules[key] ?? ""}
                      onChange={(e) =>
                        patchRule(
                          key,
                          e.target.value === "" ? null : Number(e.target.value),
                        )
                      }
                    />
                  </label>
                ))}
                <label>
                  {t.severity}
                  <select
                    value={zone.rules.severity_override ?? ""}
                    onChange={(e) =>
                      patchRule(
                        "severity_override",
                        (e.target.value as Rules["severity_override"]) || null,
                      )
                    }
                  >
                    <option value="">{t.defaultSeverity}</option>
                    {Object.entries(t.severities).map(([code, name]) => (
                      <option key={code} value={code}>
                        {name}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  {t.regulation}
                  <input
                    value={zone.rules.regulation_ref ?? ""}
                    maxLength={500}
                    onChange={(e) =>
                      patchRule("regulation_ref", e.target.value || null)
                    }
                  />
                </label>
                <label>
                  {t.responsible}
                  <select
                    value={zone.rules.responsible_user_id ?? ""}
                    onChange={(e) =>
                      patchRule("responsible_user_id", e.target.value || null)
                    }
                  >
                    <option value="">{t.notSet}</option>
                    {responsibles.map((u) => (
                      <option value={u.id} key={u.id}>
                        {u.full_name}
                      </option>
                    ))}
                  </select>
                </label>
                <h3>{t.schedule}</h3>
                <p className="muted">{t.scheduleHint}</p>
                {zone.rules.schedule?.map((entry, index) => (
                  <div className="schedule-entry" key={index}>
                    <div className="day-picker">
                      {t.days.map((day, d) => (
                        <label key={day}>
                          <input
                            type="checkbox"
                            checked={entry.days.includes(d)}
                            onChange={(e) =>
                              patchRule(
                                "schedule",
                                zone.rules.schedule!.map((s, i) =>
                                  i === index
                                    ? {
                                        ...s,
                                        days: e.target.checked
                                          ? [...s.days, d].sort()
                                          : s.days.filter((v) => v !== d),
                                      }
                                    : s,
                                ),
                              )
                            }
                          />
                          {day}
                        </label>
                      ))}
                    </div>
                    <div className="form-grid">
                      {(["from", "to"] as const).map((key) => (
                        <label key={key}>
                          {key === "from" ? t.from : t.to}
                          <input
                            type="time"
                            value={entry[key]}
                            onChange={(e) =>
                              patchRule(
                                "schedule",
                                zone.rules.schedule!.map((s, i) =>
                                  i === index
                                    ? { ...s, [key]: e.target.value }
                                    : s,
                                ),
                              )
                            }
                          />
                        </label>
                      ))}
                    </div>
                    <Button
                      variant="ghost"
                      onClick={() =>
                        patchRule(
                          "schedule",
                          zone.rules.schedule!.filter((_, i) => i !== index)
                            .length
                            ? zone.rules.schedule!.filter((_, i) => i !== index)
                            : null,
                        )
                      }
                    >
                      {t.removeSchedule}
                    </Button>
                  </div>
                ))}
                <Button
                  variant="outline"
                  onClick={() =>
                    patchRule("schedule", [
                      ...(zone.rules.schedule ?? []),
                      { days: [0, 1, 2, 3, 4], from: "08:00", to: "20:00" },
                    ])
                  }
                >
                  {t.addSchedule}
                </Button>
                <details>
                  <summary>{t.coordinates}</summary>
                  <ol className="coordinates">
                    {zone.polygon.map(([x, y], i) => (
                      <li key={i}>
                        <button
                          onClick={() => setVertex(i)}
                          className={vertex === i ? "accent" : ""}
                        >
                          {i + 1}: {x.toFixed(4)}, {y.toFixed(4)}
                        </button>
                      </li>
                    ))}
                  </ol>
                </details>
                {editable && (
                  <Button
                    variant="outline"
                    onClick={() => {
                      if (confirm(t.deleteConfirm)) {
                        commit({
                          ...doc,
                          zones: doc.zones.filter((z) => z.id !== zone.id),
                        });
                        setSelected(null);
                      }
                    }}
                  >
                    <Trash2 size={16} />
                    {t.removeZone}
                  </Button>
                )}
              </fieldset>
            </>
          ) : (
            <div className="muted">{drawing ? t.drawHint : t.selectHint}</div>
          )}
        </aside>
      </div>
    </>
  );
}
