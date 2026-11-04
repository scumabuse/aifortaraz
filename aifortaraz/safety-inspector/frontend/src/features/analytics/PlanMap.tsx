import { useRef, useState, type PointerEvent } from "react";
import { Link } from "react-router-dom";
import { Camera } from "lucide-react";
import type { Marker, PlanArea, Plans } from "./data";
import { relativeTime } from "../incidents/utils";

type Drag = {
  id: string;
  pointer: number;
  x: number;
  y: number;
  offsetX: number;
  offsetY: number;
  moved: boolean;
};
type Position = { x: number; y: number };
const statusNames = {
  online: "Онлайн",
  offline: "Офлайн",
  degraded: "Нестабильно",
};

export function incidentLink(cameraId: string, data: Plans) {
  return `/incidents?${new URLSearchParams({ camera_id: cameraId, from: data.start, to: data.end })}`;
}

export function PlanMap({
  area,
  data,
  max,
  live,
  mini = false,
  move,
}: {
  area: PlanArea;
  data: Plans;
  max: number;
  live?: boolean;
  mini?: boolean;
  move?: (id: string, position: Position) => void;
}) {
  const canvas = useRef<HTMLDivElement>(null);
  const drag = useRef<Drag | null>(null);
  const [position, setPosition] = useState<(Position & { id: string }) | null>(
    null,
  );
  function start(
    event: PointerEvent<HTMLButtonElement>,
    camera: Marker,
    x: number,
    y: number,
  ) {
    if (!move || !canvas.current || event.button !== 0) return;
    const rect = canvas.current.getBoundingClientRect();
    drag.current = {
      id: camera.id,
      pointer: event.pointerId,
      x,
      y,
      offsetX: event.clientX - rect.left - x * rect.width,
      offsetY: event.clientY - rect.top - y * rect.height,
      moved: false,
    };
    event.currentTarget.setPointerCapture(event.pointerId);
  }
  function update(event: PointerEvent<HTMLButtonElement>) {
    const current = drag.current;
    if (!current || current.pointer !== event.pointerId || !canvas.current)
      return;
    const rect = canvas.current.getBoundingClientRect();
    const clamp = (value: number) => Math.max(0, Math.min(1, value));
    const x = clamp((event.clientX - rect.left - current.offsetX) / rect.width);
    const y = clamp((event.clientY - rect.top - current.offsetY) / rect.height);
    current.x = x;
    current.y = y;
    current.moved = true;
    setPosition({ id: current.id, x, y });
  }
  function stop(event: PointerEvent<HTMLButtonElement>, save: boolean) {
    const current = drag.current;
    if (!current || current.pointer !== event.pointerId) return;
    if (save && current.moved)
      move?.(current.id, { x: current.x, y: current.y });
    drag.current = null;
    setPosition(null);
  }

  return (
    <div
      className={`plan-map ${mini ? "plan-mini" : ""}`}
      ref={canvas}
      style={{
        aspectRatio: area.image
          ? `${area.image.width} / ${area.image.height}`
          : "16 / 9",
      }}
    >
      {area.image && (
        <img
          src={area.image.url}
          alt={`План: ${area.name}`}
          draggable={false}
        />
      )}
      {area.cameras.map((camera) => {
        const x = position?.id === camera.id ? position.x : camera.x;
        const y = position?.id === camera.id ? position.y : camera.y;
        if (x == null || y == null) return null;
        const power = Math.sqrt(camera.count / Math.max(1, max));
        const label = `${camera.name} · ${camera.count} инц. · ${statusNames[camera.status]} · ${camera.latest ? relativeTime(camera.latest) : "без событий"}`;
        const recent =
          !!camera.latest && Date.now() - Date.parse(camera.latest) < 3600_000;
        const className = `plan-pin ${live ? camera.status : ""} ${live && recent ? "recent" : ""}`;
        return (
          <div
            className="plan-point"
            key={camera.id}
            style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
          >
            {camera.count > 0 && (
              <div
                className="plan-glow"
                aria-hidden
                style={{
                  width: `${32 + 112 * power}px`,
                  opacity: 0.34 + 0.62 * power,
                }}
              />
            )}
            {mini ? (
              <span className="plan-dot" title={label} />
            ) : (
              <>
                {move ? (
                  <button
                    type="button"
                    className={className}
                    aria-label={`Переместить ${camera.name}`}
                    title={label}
                    onPointerDown={(e) => start(e, camera, x, y)}
                    onPointerMove={update}
                    onPointerUp={(e) => stop(e, true)}
                    onPointerCancel={(e) => stop(e, false)}
                    onLostPointerCapture={(e) => stop(e, false)}
                    onKeyDown={(e) => {
                      const step = e.shiftKey ? 0.05 : 0.01;
                      const offsets: Record<string, [number, number]> = {
                        ArrowLeft: [-step, 0],
                        ArrowRight: [step, 0],
                        ArrowUp: [0, -step],
                        ArrowDown: [0, step],
                      };
                      const delta = offsets[e.key];
                      if (!delta) return;
                      e.preventDefault();
                      move(camera.id, {
                        x: Math.max(0, Math.min(1, x + delta[0])),
                        y: Math.max(0, Math.min(1, y + delta[1])),
                      });
                    }}
                  >
                    <Camera size={16} />
                    {camera.count}
                  </button>
                ) : (
                  <Link
                    className={className}
                    to={incidentLink(camera.id, data)}
                    title={label}
                    aria-label={label}
                  >
                    <Camera size={16} />
                    {camera.count}
                  </Link>
                )}
                <div
                  className="plan-zones"
                  aria-label={camera.zones.map((zone) => zone.name).join(", ")}
                >
                  {camera.zones.map((zone) => (
                    <i
                      key={zone.id}
                      style={{ background: zone.color }}
                      title={zone.name}
                    />
                  ))}
                </div>
              </>
            )}
          </div>
        );
      })}
    </div>
  );
}
