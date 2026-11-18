import { describe, expect, it, vi } from "vitest";
import { ApiError, getApiBaseUrl, getHealth, getReady } from "@/services/api";
import { HEALTH, NOT_READY, stubApi } from "./utils";

describe("api client", () => {
  it("parses the body of a 503 readiness response instead of throwing", async () => {
    stubApi({ "/ready": [503, NOT_READY] });
    await expect(getReady()).resolves.toEqual(NOT_READY);
  });

  it("throws ApiError for unexpected statuses and network failures", async () => {
    stubApi({ "/health": [500, {}] });
    await expect(getHealth()).rejects.toMatchObject({ name: "ApiError", status: 500 });
    stubApi("network-error");
    await expect(getHealth()).rejects.toBeInstanceOf(ApiError);
  });

  it("throws ApiError on a non-JSON body", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("<html>", { status: 200 })));
    await expect(getHealth()).rejects.toThrow(/unreadable/);
  });

  it("builds URLs from VITE_API_BASE_URL and strips trailing slashes", async () => {
    vi.stubEnv("VITE_API_BASE_URL", "https://api.example.app//");
    expect(getApiBaseUrl()).toBe("https://api.example.app");
    const fetchMock = stubApi({ "/health": [200, HEALTH] });
    await getHealth();
    expect(fetchMock).toHaveBeenCalledWith("https://api.example.app/api/v1/health", undefined);
  });

  it("refuses to guess a backend URL outside development", () => {
    vi.stubEnv("VITE_API_BASE_URL", "");
    vi.stubEnv("DEV", false);
    expect(() => getApiBaseUrl()).toThrow(/not configured/);
  });
});
