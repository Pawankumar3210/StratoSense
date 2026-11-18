import { render } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { vi } from "vitest";
import App from "@/App";
import { ApiStatusProvider } from "@/contexts/ApiStatusContext";
import { ThemeProvider } from "@/contexts/ThemeContext";

export const HEALTH = { status: "ok", service: "stratosense-api", version: "0.1.0", environment: "test" };
export const NOT_READY = {
  ready: false,
  model_loaded: false,
  model_version: null,
  device: null,
  reason: "No model checkpoint found at 'artifacts/checkpoints/stratosense_unet.pth'.",
};

/** Stub fetch with per-path responses: `{ "/health": [200, body], "/ready": [503, body] }`. */
export function stubApi(routes: Record<string, [number, unknown]> | "network-error") {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    if (routes === "network-error") throw new TypeError("Failed to fetch");
    const path = String(input).replace(/^.*\/api\/v1/, "");
    const hit = routes[path];
    if (!hit) return new Response("{}", { status: 404 });
    return new Response(JSON.stringify(hit[1]), { status: hit[0] });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

export function renderApp(path = "/") {
  return render(
    <ThemeProvider>
      <ApiStatusProvider>
        <MemoryRouter initialEntries={[path]}>
          <App />
        </MemoryRouter>
      </ApiStatusProvider>
    </ThemeProvider>,
  );
}
