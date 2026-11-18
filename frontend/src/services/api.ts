import type { HealthResponse, ReadyResponse } from "@/types/api";

/** Raised for network failures, misconfiguration and unexpected HTTP responses. */
export class ApiError extends Error {
  readonly status?: number;
  constructor(message: string, status?: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Resolve the backend base URL. Production builds must set VITE_API_BASE_URL; the localhost
 * fallback exists only in development builds and is removed from production bundles.
 */
export function getApiBaseUrl(): string {
  const configured = import.meta.env.VITE_API_BASE_URL?.trim();
  if (configured) return configured.replace(/\/+$/, "");
  if (import.meta.env.DEV) return "http://localhost:8000";
  throw new ApiError("VITE_API_BASE_URL is not configured for this build.");
}

async function request(path: string, init?: RequestInit): Promise<Response> {
  const url = `${getApiBaseUrl()}/api/v1${path}`;
  try {
    return await fetch(url, init);
  } catch {
    throw new ApiError("Could not reach the StratoSense API.");
  }
}

async function readJson<T>(response: Response): Promise<T> {
  try {
    return (await response.json()) as T;
  } catch {
    throw new ApiError("The API returned an unreadable response.", response.status);
  }
}

export async function getHealth(): Promise<HealthResponse> {
  const response = await request("/health");
  if (!response.ok) throw new ApiError(`Health check failed (${response.status}).`, response.status);
  return readJson<HealthResponse>(response);
}

/** `/ready` answers 503 with a JSON body when the model is not loaded; that body is data, not an error. */
export async function getReady(): Promise<ReadyResponse> {
  const response = await request("/ready");
  if (response.ok || response.status === 503) return readJson<ReadyResponse>(response);
  throw new ApiError(`Readiness check failed (${response.status}).`, response.status);
}
