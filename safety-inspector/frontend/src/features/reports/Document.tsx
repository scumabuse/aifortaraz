import { useEffect, useRef, useState } from "react";
import { apiBlob } from "../../api";
import { Button } from "../../components/ui/button";
import "./reports.css";

export async function download(path: string, name: string) {
  const url = URL.createObjectURL(await apiBlob(path));
  const link = document.createElement("a");
  link.href = url;
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function PrintDocument({
  html,
  title,
}: {
  html: string;
  title: string;
}) {
  const frame = useRef<HTMLIFrameElement>(null);
  const [loaded, setLoaded] = useState(false);
  return (
    <section className="panel report-document">
      <div className="panel-heading">
        <h2>{title}</h2>
        <Button
          disabled={!loaded}
          onClick={() => frame.current?.contentWindow?.print()}
        >
          Печать / PDF
        </Button>
      </div>
      <iframe
        ref={frame}
        title={title}
        sandbox="allow-same-origin allow-modals"
        srcDoc={html}
        onLoad={() => setLoaded(true)}
      />
    </section>
  );
}

export function useDocument(id: string | undefined) {
  const [document, setDocument] = useState<{
    id: string;
    html?: string;
    error?: string;
  }>();
  const [retry, setRetry] = useState(0);
  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    apiBlob(`/api/v1/reports/${id}/document`, { signal: controller.signal })
      .then((blob) => blob.text())
      .then((html) => {
        if (!controller.signal.aborted) setDocument({ id, html });
      })
      .catch((error: Error) => {
        if (!controller.signal.aborted)
          setDocument({ id, error: error.message });
      });
    return () => controller.abort();
  }, [id, retry]);
  return {
    document: document?.id === id ? document : undefined,
    retry: () => setRetry((value) => value + 1),
  };
}
