import { createContext, useContext } from "react";
import type { ReactNode } from "react";
import { useApiStatus } from "@/hooks/useApiStatus";
import type { ApiStatus } from "@/hooks/useApiStatus";

interface Value {
  status: ApiStatus;
  refresh: () => void;
}

const ApiStatusContext = createContext<Value | null>(null);

/** One shared backend status check for the whole app (top bar and overview read the same result). */
export function ApiStatusProvider({ children }: { children: ReactNode }) {
  const value = useApiStatus();
  return <ApiStatusContext.Provider value={value}>{children}</ApiStatusContext.Provider>;
}

export function useApiStatusContext(): Value {
  const ctx = useContext(ApiStatusContext);
  if (!ctx) throw new Error("useApiStatusContext must be used inside <ApiStatusProvider>");
  return ctx;
}
