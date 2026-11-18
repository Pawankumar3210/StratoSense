import { useCallback, useEffect, useState } from "react";
import { ApiError, getHealth, getReady } from "@/services/api";
import type { HealthResponse, ReadyResponse } from "@/types/api";

export type ApiStatus =
  | { state: "checking" }
  | { state: "offline"; message: string }
  | { state: "online"; health: HealthResponse; ready: ReadyResponse };

/** Checks backend liveness and model readiness on mount and on demand. */
export function useApiStatus(): { status: ApiStatus; refresh: () => void } {
  const [status, setStatus] = useState<ApiStatus>({ state: "checking" });
  const [tick, setTick] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setStatus({ state: "checking" });
    (async () => {
      try {
        const health = await getHealth();
        const ready = await getReady();
        if (!cancelled) setStatus({ state: "online", health, ready });
      } catch (error) {
        const message = error instanceof ApiError ? error.message : "Unexpected error.";
        if (!cancelled) setStatus({ state: "offline", message });
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [tick]);

  const refresh = useCallback(() => setTick((t) => t + 1), []);
  return { status, refresh };
}
