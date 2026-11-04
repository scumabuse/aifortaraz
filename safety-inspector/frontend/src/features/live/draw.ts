import type { ZoneRecord } from "../cameras/types";
import { ppeName } from "../incidents/utils";
import ru from "../../locales/ru.json";
import { containRect, detectionOpacity, projectBox } from "./geometry";
import type { DetectionFrame, DetectionObject, Overlays } from "./types";

const t = ru.liveUi;
export function fallbackObjects(
  cameraId: string,
  timestamp: number,
): DetectionObject[] {
  const seed = [...cameraId].reduce((sum, c) => sum + c.charCodeAt(0), 0) % 17;
  return [17, 24, 31].map((track, index) => ({
    track_id: track,
    label: "person",
    bbox: [
      0.12 + (Math.sin(timestamp * 0.14 + index * 1.7 + seed) + 1) * 0.32,
      0.36 + 0.07 * Math.cos(timestamp * 0.12 + index),
      0.085,
      0.32,
    ],
    conf: 0.94,
    ppe: { helmet: true, vest: true },
    zones: [],
    state: "ok",
  }));
}
function drawScene(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  cameraId: string,
  objects: DetectionObject[],
) {
  ctx.fillStyle = "#101a22";
  ctx.fillRect(0, 0, width, height);
  const rect = containRect(1280, 720, width, height);
  ctx.save();
  ctx.translate(rect.x, rect.y);
  ctx.scale(rect.width / 1280, rect.height / 720);
  ctx.fillStyle = "#1c2c37";
  ctx.fillRect(0, 0, 1280, 720);
  ctx.strokeStyle = "#2c4351";
  ctx.lineWidth = 1;
  for (let x = 0; x < 1280; x += 80) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 720);
    ctx.stroke();
  }
  for (let y = 0; y < 720; y += 80) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(1280, y);
    ctx.stroke();
  }
  ctx.strokeStyle = "#476454";
  ctx.lineWidth = 3;
  ctx.setLineDash([18, 12]);
  ctx.strokeRect(45, 365, 1190, 280);
  ctx.setLineDash([]);
  for (const [x, y, w, h] of [
    [65, 80, 290, 220],
    [930, 80, 270, 220],
  ]) {
    ctx.fillStyle = "#2e424e";
    ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = "#506c78";
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = "#1b2b35";
    ctx.fillRect(x + 25, y + 25, w - 50, h - 50);
    ctx.fillStyle = "#59786b";
    for (let n = 0; n < 3; n++) ctx.fillRect(x + 40 + n * 60, y + 45, 35, 12);
  }
  ctx.fillStyle = "#597c70";
  ctx.font = "20px Segoe UI, sans-serif";
  ctx.fillText(`${t.syntheticScene} · ${cameraId.slice(0, 8)}`, 65, 45);
  for (const object of objects) {
    const [nx, ny, nw, nh] = object.bbox;
    const x = nx * 1280,
      y = ny * 720,
      w = nw * 1280,
      h = nh * 720;
    if (object.label === "forklift") {
      ctx.fillStyle = "#b08f42";
      ctx.fillRect(x, y + h * 0.25, w * 0.8, h * 0.5);
      ctx.fillStyle = "#233843";
      ctx.fillRect(x + w * 0.15, y, w * 0.4, h * 0.45);
      ctx.strokeStyle = "#baa56b";
      ctx.lineWidth = 8;
      ctx.beginPath();
      ctx.moveTo(x + w * 0.85, y + h * 0.1);
      ctx.lineTo(x + w * 0.85, y + h * 0.85);
      ctx.lineTo(x + w, y + h * 0.85);
      ctx.stroke();
      ctx.fillStyle = "#0e1920";
      for (const cx of [x + w * 0.15, x + w * 0.65]) {
        ctx.beginPath();
        ctx.arc(cx, y + h * 0.83, h * 0.13, 0, Math.PI * 2);
        ctx.fill();
      }
    } else {
      ctx.fillStyle = object.ppe?.helmet === false ? "#a1b1b9" : "#d2b666";
      ctx.beginPath();
      ctx.ellipse(
        x + w * 0.5,
        y + h * 0.13,
        w * 0.26,
        h * 0.13,
        0,
        0,
        Math.PI * 2,
      );
      ctx.fill();
      ctx.fillStyle = object.ppe?.vest === false ? "#728b98" : "#5c9984";
      ctx.beginPath();
      ctx.roundRect(x + w * 0.1, y + h * 0.3, w * 0.8, h * 0.46, 12);
      ctx.fill();
      ctx.strokeStyle = "#90a5ae";
      ctx.lineWidth = 11;
      ctx.beginPath();
      ctx.moveTo(x + w * 0.32, y + h * 0.77);
      ctx.lineTo(x + w * 0.28, y + h);
      ctx.moveTo(x + w * 0.68, y + h * 0.77);
      ctx.lineTo(x + w * 0.72, y + h);
      ctx.stroke();
    }
  }
  ctx.restore();
}
export function drawOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  frameWidth: number,
  frameHeight: number,
  frame: DetectionFrame | undefined,
  zones: ZoneRecord[],
  overlays: Overlays,
  placeholder: boolean,
  cameraId: string,
  now: number,
) {
  ctx.clearRect(0, 0, width, height);
  const age = frame ? now - frame.receivedAt : Infinity;
  const fresh = age < 3000;
  const objects =
    frame && fresh
      ? frame.objects
      : fallbackObjects(cameraId, Date.now() / 1000);
  if (placeholder) drawScene(ctx, width, height, cameraId, objects);
  const rect = containRect(
    placeholder ? 1280 : frameWidth,
    placeholder ? 720 : frameHeight,
    width,
    height,
  );
  ctx.save();
  ctx.beginPath();
  ctx.rect(rect.x, rect.y, rect.width, rect.height);
  ctx.clip();
  if (overlays.zones)
    for (const zone of zones.filter((z) => z.enabled)) {
      ctx.beginPath();
      zone.polygon.forEach(([x, y], i) => {
        const px = rect.x + x * rect.width,
          py = rect.y + y * rect.height;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.fillStyle = `${zone.color}20`;
      ctx.fill();
      ctx.strokeStyle = zone.color;
      ctx.lineWidth = 1.5;
      ctx.stroke();
      ctx.font = "13px Segoe UI, sans-serif";
      ctx.fillStyle = zone.color;
      ctx.fillText(
        zone.name,
        rect.x + zone.polygon[0][0] * rect.width + 5,
        rect.y + zone.polygon[0][1] * rect.height + 16,
      );
    }
  if (frame && fresh) {
    ctx.globalAlpha = detectionOpacity(age);
    for (const object of frame.objects) {
      if (overlays.violationsOnly && (!object.state || object.state === "ok"))
        continue;
      const box = projectBox(object.bbox, rect);
      const vehicle = ["forklift", "vehicle", "truck"].includes(object.label);
      const color = vehicle
        ? "#73b9ff"
        : object.state === "danger"
          ? "#ff6673"
          : object.state === "violation"
            ? "#f3a15c"
            : "#6ee0b7";
      if (object.state === "danger")
        ctx.globalAlpha =
          detectionOpacity(age) * (0.6 + 0.4 * Math.abs(Math.sin(now / 190)));
      else ctx.globalAlpha = detectionOpacity(age);
      if (overlays.boxes) {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(box.x, box.y, box.width, box.height);
      }
      const parts: string[] = [];
      if (
        overlays.trackIds &&
        object.track_id !== undefined &&
        object.track_id !== null
      )
        parts.push(`#${object.track_id}`);
      if (overlays.labels) {
        parts.push(
          t.objectTypes[object.label as keyof typeof t.objectTypes] ??
            t.objectTypes.other,
        );
        for (const [code, present] of Object.entries(object.ppe ?? {}).slice(
          0,
          4,
        ))
          parts.push(`${ppeName(code)} ${present ? "✓" : "✗"}`);
      }
      if (parts.length) {
        ctx.font = "13px Segoe UI, sans-serif";
        let text = parts.join(" · ");
        const max = Math.min(rect.width - 8, 360);
        while (ctx.measureText(text).width > max && text.length > 3)
          text = text.slice(0, -2);
        const textWidth = ctx.measureText(text).width;
        const left = Math.max(
          rect.x + 2,
          Math.min(box.x, rect.x + rect.width - textWidth - 10),
        );
        const top = Math.max(rect.y + 2, box.y - 24);
        ctx.fillStyle = "#0d1820e8";
        ctx.fillRect(left, top, textWidth + 10, 22);
        ctx.fillStyle = color;
        ctx.fillText(text, left + 5, top + 16);
      }
    }
  }
  ctx.restore();
}
