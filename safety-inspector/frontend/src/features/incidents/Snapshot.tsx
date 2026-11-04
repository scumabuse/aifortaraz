import { useEffect, useState } from "react";
import { ImageOff } from "lucide-react";
import { boxCaption } from "../live/labels";
import type { IncidentRecord } from "./types";
import ru from "../../locales/ru.json";

export function Snapshot({
  incident,
  thumbnail = false,
}: {
  incident: IncidentRecord;
  thumbnail?: boolean;
}) {
  const [failed, setFailed] = useState(false);
  const [ratio, setRatio] = useState(16 / 9);
  const url = thumbnail ? incident.thumbnail_url : incident.snapshot_url;
  const viewWidth = Number.isFinite(ratio) && ratio > 0 ? ratio : 1;
  useEffect(() => {
    setFailed(false);
    setRatio(
      incident.details.frame_size
        ? incident.details.frame_size[0] / incident.details.frame_size[1]
        : 16 / 9,
    );
  }, [url, incident.details.frame_size]);
  return (
    <div
      className={`incident-snapshot ${thumbnail ? "thumbnail" : ""}`}
      style={{ aspectRatio: thumbnail ? "16/9" : ratio }}
    >
      {url && !failed ? (
        <>
          <img
            src={url}
            alt={`${incident.type_name} · ${incident.camera_name}`}
            loading={thumbnail ? "lazy" : "eager"}
            onLoad={(e) => {
              if (!thumbnail)
                setRatio(
                  e.currentTarget.naturalWidth / e.currentTarget.naturalHeight,
                );
            }}
            onError={() => setFailed(true)}
          />
          {!thumbnail && (
            <svg
              viewBox={`0 0 ${viewWidth} 1`}
              preserveAspectRatio="none"
              aria-label={ru.incidentUi.boxes}
            >
              {incident.details.boxes?.map((box, index) => {
                const alignEnd = box.bbox[0] > 0.75;
                return (
                  <g key={index}>
                    <rect
                      x={box.bbox[0] * viewWidth}
                      y={box.bbox[1]}
                      width={box.bbox[2] * viewWidth}
                      height={box.bbox[3]}
                      fill="none"
                      stroke={
                        incident.severity === "critical" ? "#ff5c69" : "#eab54d"
                      }
                      strokeWidth="2"
                      vectorEffect="non-scaling-stroke"
                    />
                    <text
                      x={
                        alignEnd
                          ? (box.bbox[0] + box.bbox[2]) * viewWidth - 0.005
                          : box.bbox[0] * viewWidth + 0.005
                      }
                      y={Math.max(0.025, box.bbox[1] - 0.01)}
                      textAnchor={alignEnd ? "end" : "start"}
                      fill="#ffffff"
                      stroke="#0d1820"
                      strokeWidth="0.005"
                      paintOrder="stroke"
                      fontSize="0.025"
                    >
                      {boxCaption(box, ru.liveUi.objectTypes)}
                    </text>
                  </g>
                );
              })}
            </svg>
          )}
        </>
      ) : (
        <div className="snapshot-empty">
          <ImageOff size={thumbnail ? 20 : 40} />
          {!thumbnail && <p>{ru.incidentUi.noSnapshot}</p>}
        </div>
      )}
    </div>
  );
}
