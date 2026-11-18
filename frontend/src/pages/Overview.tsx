import { RefreshCw } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";
import { useApiStatusContext } from "@/contexts/ApiStatusContext";

function Readout({ label, tone, children }: { label: string; tone: "ok" | "warn" | "bad" | "idle"; children: ReactNode }) {
  const dot = { ok: "bg-ok", warn: "bg-warn", bad: "bg-bad", idle: "bg-muted" }[tone];
  return (
    <div className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-4 sm:grid-cols-[8.5rem_minmax(0,1fr)] border-b border-line py-3.5 text-sm last:border-b-0">
      <dt className="text-muted">{label}</dt>
      <dd className="flex items-start gap-2.5">
        <span className={`mt-1.5 h-2 w-2 shrink-0 rounded-full ${dot}`} aria-hidden />
        <span className="min-w-0 break-words">{children}</span>
      </dd>
    </div>
  );
}

function PipelineStatus() {
  const { status, refresh } = useApiStatusContext();

  let api: ReactNode = "Checking…";
  let apiTone: "ok" | "bad" | "idle" = "idle";
  let model: ReactNode = "Checking…";
  let modelTone: "ok" | "warn" | "idle" = "idle";
  let evaluation: ReactNode = "Checking…";

  if (status.state === "offline") {
    api = status.message;
    apiTone = "bad";
    model = "Unknown while the API is unreachable.";
    evaluation = "Unknown while the API is unreachable.";
  } else if (status.state === "online") {
    const { health, ready } = status;
    api = `Online · v${health.version} · ${health.environment}`;
    apiTone = "ok";
    if (ready.ready) {
      model = `Loaded · v${ready.model_version ?? "unknown"} · ${ready.device ?? "unknown device"}`;
      modelTone = "ok";
      evaluation = "Provided by the API once evaluation artifacts are published.";
    } else {
      model = `Not loaded. ${ready.reason ?? ""}`.trim();
      modelTone = "warn";
      evaluation = "Unavailable until a trained model is loaded.";
    }
  }

  return (
    <section aria-labelledby="status-heading" className="min-w-0">
      <div className="flex min-h-8 items-center justify-between">
        <h2 id="status-heading" className="text-base font-semibold">
          Pipeline status
        </h2>
        <button
          type="button"
          onClick={refresh}
          className="inline-flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1.5 text-xs text-muted transition-colors hover:text-ink"
        >
          <RefreshCw size={13} aria-hidden /> Check again
        </button>
      </div>
      <dl className="mt-3 border-t border-line">
        <Readout label="API" tone={apiTone}>{api}</Readout>
        <Readout label="Model" tone={modelTone}>{model}</Readout>
        <Readout label="Evaluation" tone="idle">{evaluation}</Readout>
      </dl>
    </section>
  );
}

const DATASET_FACTS: Array<[string, string]> = [
  ["Source", "38-Cloud, Mohajerani & Saeedi (Simon Fraser University)"],
  ["Satellite", "Landsat 8"],
  ["Scenes", "38"],
  ["Patches", "384 × 384 px · 8,400 for training · 9,201 for testing"],
  ["Model input", "Red (band 4), Green (3), Blue (2), Near infrared (5)"],
  ["Labels", "Manual pixel-level cloud masks; thin cloud and haze count as cloud"],
  ["Licence", "Apache 2.0, as published in the dataset repository"],
];

export default function Overview() {
  return (
    <div className="mx-auto max-w-5xl">
      <header className="max-w-2xl">
        <h1 className="text-4xl font-semibold tracking-tight sm:text-5xl">StratoSense</h1>
        <p className="mt-3 text-xl text-accent">Satellite Cloud Intelligence, Pixel by Pixel.</p>
        <p className="mt-5 leading-relaxed text-muted">
          StratoSense segments clouds in four-band Landsat 8 imagery with a U-Net trained on the 38-Cloud dataset,
          then reports cloud coverage over the valid pixels of the scene.
        </p>
        <p className="mt-3 leading-relaxed text-muted">
          This is the foundation build. The analysis workspace becomes available once the trained model is loaded
          by the API; the status panel below shows where things stand.
        </p>
        <Link
          to="/analyze"
          className="mt-7 inline-flex items-center rounded-md bg-accent px-4 py-2.5 text-sm font-medium text-accent-ink transition-opacity hover:opacity-90"
        >
          Open the analysis workspace
        </Link>
      </header>

      <div className="mt-14 grid gap-12 lg:grid-cols-2">
        <PipelineStatus />
        <section aria-labelledby="dataset-heading" className="min-w-0">
          <div className="flex min-h-8 items-center">
            <h2 id="dataset-heading" className="text-base font-semibold">
              Training dataset
            </h2>
          </div>
          <dl className="mt-3 border-t border-line">
            {DATASET_FACTS.map(([term, detail]) => (
              <div key={term} className="grid grid-cols-[5.5rem_minmax(0,1fr)] gap-x-4 sm:grid-cols-[6rem_minmax(0,1fr)] border-b border-line py-3.5 text-sm last:border-b-0">
                <dt className="text-muted">{term}</dt>
                <dd className="min-w-0 break-words">{detail}</dd>
              </div>
            ))}
          </dl>
        </section>
      </div>
    </div>
  );
}
