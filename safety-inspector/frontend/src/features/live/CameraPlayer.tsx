import { useEffect, useRef, useState } from "react";
import Hls from "hls.js/light";
import { Camera, RotateCw, Video } from "lucide-react";
import { Link } from "react-router-dom";
import { api, readSession } from "../../api";
import { Button } from "../../components/ui/button";
import type { CameraRecord, ZoneRecord } from "../cameras/types";
import { useLiveStore } from "../realtime/store";
import { drawOverlay } from "./draw";
import { isAiLive, isAiOffline } from "./status";
import type { DetectionFrame, Overlays } from "./types";
import ru from "../../locales/ru.json";

const t = ru.liveUi;
function mediaUrl(camera: CameraRecord, attempt: number) {
  const input =
    camera.stream_type === "rtsp"
      ? (camera.preview_url ?? "")
      : camera.stream_url;
  if (!input) return "";
  let url: URL;
  try {
    url = new URL(input, location.origin);
  } catch {
    return input;
  }
  if (
    url.origin === location.origin &&
    /^\/api\/v1\/stream\/[^/]+\.mjpeg$/.test(url.pathname)
  ) {
    const token = readSession()?.access_token;
    if (token) url.searchParams.set("token", token);
  }
  if (
    camera.stream_type === "file" &&
    url.origin === location.origin &&
    attempt
  )
    url.searchParams.set("_retry", String(attempt));
  return url.href;
}
export function CameraPlayer({
  camera,
  zones,
  overlays,
  demoActive,
}: {
  camera: CameraRecord;
  zones: ZoneRecord[];
  overlays: Overlays;
  demoActive: boolean;
}) {
  const wrapper = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const image = useRef<HTMLImageElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const alive = useRef(true);
  const generation = useRef(0);
  const dimensions = useRef<[number, number]>([1280, 720]);
  const props = useRef({
    zones,
    overlays,
    demoActive,
    camera,
    status: "loading",
  });
  const [status, setStatus] = useState("loading");
  const [attempt, setAttempt] = useState(0);
  const [now, setNow] = useState(Date.now());
  const [permissionError, setPermissionError] = useState(false);
  const source = mediaUrl(camera, attempt);
  props.current = { zones, overlays, demoActive, camera, status };
  const placeholder = demoActive && status !== "ready";
  const offline = isAiOffline(camera, now);
  const aiLive = isAiLive(camera, now);
  useEffect(() => {
    alive.current = true;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => {
      alive.current = false;
      clearInterval(timer);
      stream.current?.getTracks().forEach((track) => track.stop());
    };
  }, []);
  useEffect(() => {
    setPermissionError(false);
    dimensions.current = [1280, 720];
    if (camera.stream_type === "webcam") {
      setStatus("waiting");
      return;
    }
    setStatus("loading");
    const current = video.current;
    let hls: Hls | null = null;
    let active = true;
    const fail = () => {
      if (active) setStatus("failed");
    };
    if (camera.stream_type === "file" || camera.stream_type === "hls") {
      if (current && source) {
        if (
          camera.stream_type === "hls" &&
          !current.canPlayType("application/vnd.apple.mpegurl")
        ) {
          if (Hls.isSupported()) {
            hls = new Hls({
              enableWorker: true,
              lowLatencyMode: true,
              maxBufferLength: 10,
            });
            hls.loadSource(source);
            hls.attachMedia(current);
            hls.on(Hls.Events.ERROR, (_event, data) => {
              if (data.fatal) fail();
            });
          } else fail();
        } else current.src = source;
        void current.play().catch(() => {
          /* Muted playback is retried by media events. */
        });
      } else fail();
    }
    if (!source) fail();
    const timeout = setTimeout(() => {
      if (
        active &&
        !(current && current.readyState >= 2) &&
        !(image.current && image.current.naturalWidth > 0)
      )
        setStatus("failed");
    }, 10000);
    return () => {
      active = false;
      if (timeout) clearTimeout(timeout);
      hls?.destroy();
      if (current) {
        current.pause();
        current.removeAttribute("src");
        current.load();
      }
    };
  }, [camera.stream_type, source, attempt]);
  useEffect(() => {
    if (status !== "failed" || camera.stream_type === "webcam") return;
    const timeout = setTimeout(() => {
      const reconnect = async () => {
        if (source.includes("/api/v1/stream/")) {
          try {
            await api("/api/v1/auth/me");
          } catch {
            /* The global session handler owns authentication errors. */
          }
        }
        if (alive.current) setAttempt((value) => value + 1);
      };
      void reconnect();
    }, 5000);
    return () => clearTimeout(timeout);
  }, [status, camera.stream_type, source, attempt]);
  useEffect(() => {
    if (camera.stream_type !== "mjpeg" || !source.includes("/api/v1/stream/"))
      return;
    let active = true;
    const check = async () => {
      try {
        const state = await api<{ relay_frame_age_sec: number | null }>(
          `/api/v1/cameras/${camera.id}/status`,
        );
        if (
          active &&
          (state.relay_frame_age_sec === null || state.relay_frame_age_sec > 5)
        )
          setStatus("failed");
      } catch {
        if (active) setStatus("failed");
      }
    };
    const timer = setInterval(() => void check(), 2000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [camera.id, camera.stream_type, source]);
  useEffect(() => {
    const parent = wrapper.current,
      element = canvas.current;
    if (!parent || !element) return;
    let width = parent.clientWidth,
      height = parent.clientHeight;
    const resize = () => {
      width = parent.clientWidth;
      height = parent.clientHeight;
      const dpr = Math.min(devicePixelRatio || 1, 2);
      element.width = Math.round(width * dpr);
      element.height = Math.round(height * dpr);
      element.style.width = `${width}px`;
      element.style.height = `${height}px`;
      element.getContext("2d")?.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    const observer = new ResizeObserver(resize);
    observer.observe(parent);
    resize();
    let request = 0,
      last = 0;
    const render = (time: number) => {
      request = requestAnimationFrame(render);
      if (time - last < 1000 / 15 || !width || !height) return;
      last = time;
      const state = props.current;
      const ctx = element.getContext("2d");
      if (!ctx) return;
      const frame = useLiveStore.getState().detections[state.camera.id] as
        DetectionFrame | undefined;
      drawOverlay(
        ctx,
        width,
        height,
        ...dimensions.current,
        frame,
        state.zones,
        state.overlays,
        state.demoActive && state.status !== "ready",
        state.camera.id,
        time,
      );
    };
    request = requestAnimationFrame(render);
    return () => {
      observer.disconnect();
      cancelAnimationFrame(request);
    };
  }, []);
  async function enableWebcam() {
    const currentGeneration = ++generation.current;
    setStatus("loading");
    setPermissionError(false);
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error("Unavailable");
      const media = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (!alive.current || currentGeneration !== generation.current) {
        media.getTracks().forEach((track) => track.stop());
        return;
      }
      stream.current?.getTracks().forEach((track) => track.stop());
      stream.current = media;
      if (video.current) {
        video.current.srcObject = media;
        await video.current.play();
      }
      media.getVideoTracks().forEach((track) => {
        track.onended = () => {
          if (alive.current) setStatus("failed");
        };
      });
    } catch {
      if (alive.current) {
        stream.current?.getTracks().forEach((track) => track.stop());
        stream.current = null;
        setPermissionError(true);
        setStatus("failed");
      }
    }
  }
  function ready() {
    if (video.current?.videoWidth)
      dimensions.current = [
        video.current.videoWidth,
        video.current.videoHeight,
      ];
    if (image.current?.naturalWidth)
      dimensions.current = [
        image.current.naturalWidth,
        image.current.naturalHeight,
      ];
    setStatus("ready");
  }
  return (
    <article className="camera-player">
      <div className="player-heading">
        <Link to={`/live/${camera.id}`}>
          <Camera size={16} />
          {camera.name}
        </Link>
        <span className={offline ? "player-offline" : "player-online"}>
          {offline ? t.offline : t.online} ·{" "}
          {offline || camera.fps === null ? "—" : Math.round(camera.fps)}{" "}
          {t.fps}
        </span>
      </div>
      <div className="player-frame" ref={wrapper}>
        {["file", "hls", "webcam"].includes(camera.stream_type) && (
          <video
            ref={video}
            autoPlay
            muted
            playsInline
            loop={camera.stream_type === "file"}
            onLoadedData={ready}
            onPlaying={ready}
            onError={() => setStatus("failed")}
            className={status === "ready" ? "" : "media-hidden"}
            aria-hidden={status !== "ready"}
          />
        )}
        {["mjpeg", "rtsp"].includes(camera.stream_type) && source && (
          <img
            key={`${source}-${attempt}`}
            ref={image}
            src={source}
            alt={camera.name}
            onLoad={ready}
            onError={() => setStatus("failed")}
            className={status === "ready" ? "" : "media-hidden"}
            aria-hidden={status !== "ready"}
          />
        )}
        <canvas
          ref={canvas}
          className="camera-overlay"
          aria-label={t.overlayCanvas}
        />
        <div className="player-badges">
          {aiLive && <span className="player-ai">{t.aiBadge}</span>}
          {demoActive && !aiLive && (
            <span className="player-demo">{t.demo}</span>
          )}
          {offline && (
            <span className="player-ai-offline">
              {camera.ai_enabled ? t.aiOffline : t.aiDisabled}
            </span>
          )}
        </div>
        {!placeholder && status !== "ready" && (
          <div className="no-signal">
            <Video size={30} />
            <strong>
              {status === "loading"
                ? t.connecting
                : camera.stream_type === "webcam" && status === "waiting"
                  ? t.webcamWaiting
                  : t.noSignal}
            </strong>
            {camera.stream_type !== "webcam" && <small>{t.retryHint}</small>}
          </div>
        )}
        {camera.stream_type === "webcam" && status !== "ready" && (
          <div className="webcam-action">
            <Button
              onClick={() => void enableWebcam()}
              disabled={status === "loading"}
            >
              <Video size={17} />
              {t.enableWebcam}
            </Button>
            {permissionError && <p role="alert">{t.webcamError}</p>}
          </div>
        )}
        {status === "failed" &&
          camera.stream_type !== "webcam" &&
          !placeholder && (
            <button
              className="retry-stream icon-button"
              title={t.retry}
              aria-label={t.retry}
              onClick={() => setAttempt((v) => v + 1)}
            >
              <RotateCw size={17} />
            </button>
          )}
        {camera.stream_type === "rtsp" && (
          <p className="rtsp-hint">{t.rtspHint}</p>
        )}
      </div>
      <div className="player-footer">
        <span>{camera.area_name}</span>
        <span>{t.streamTypes[camera.stream_type]}</span>
      </div>
    </article>
  );
}
