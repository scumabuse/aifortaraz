import { useEffect, useState } from "react";
import { ImageOff } from "lucide-react";
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
              viewBox="0 0 1 1"
              preserveAspectRatio="none"
              aria-label={ru.incidentUi.boxes}
            >
              {incident.details.boxes?.map((box, index) => (
                <g key={index}>
                  <rect
                    x={box.bbox[0]}
                    y={box.bbox[1]}
                    width={box.bbox[2]}
                    height={box.bbox[3]}
                    fill="none"
                    stroke={
                      incident.severity === "critical" ? "#ff5c69" : "#eab54d"
                    }
                    strokeWidth="2"
                    vectorEffect="non-scaling-stroke"
                  />
                  <text
                    x={box.bbox[0] + 0.005}
                    y={Math.max(0.025, box.bbox[1] - 0.01)}
                    fill="#ffffff"
                    fontSize="0.025"
                  >
                    {box.track_id !== null && box.track_id !== undefined
                      ? `#${box.track_id}`
                      : ""}
                  </text>
                </g>
              ))}
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
