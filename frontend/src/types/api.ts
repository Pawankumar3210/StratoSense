/** Mirrors backend/app/schemas/response.py. Keep in sync. */

export interface HealthResponse {
  status: string;
  service: string;
  version: string;
  environment: string;
}

export interface ReadyResponse {
  ready: boolean;
  model_loaded: boolean;
  model_version: string | null;
  device: string | null;
  reason: string | null;
}
