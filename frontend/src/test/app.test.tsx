import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { THEME_STORAGE_KEY } from "@/contexts/ThemeContext";
import { HEALTH, NOT_READY, renderApp, stubApi } from "./utils";

describe("routing", () => {
  it("renders the overview with the product identity", async () => {
    stubApi({ "/health": [200, HEALTH], "/ready": [503, NOT_READY] });
    renderApp("/");
    expect(screen.getByRole("heading", { name: "StratoSense", level: 1 })).toBeInTheDocument();
    expect(screen.getByText("Satellite Cloud Intelligence, Pixel by Pixel.")).toBeInTheDocument();
  });

  it("exposes all seven sections in the primary navigation", () => {
    stubApi({ "/health": [200, HEALTH], "/ready": [503, NOT_READY] });
    renderApp("/");
    const nav = screen.getByRole("navigation", { name: "Primary" });
    for (const label of ["Overview", "Analyze", "Sample Library", "Model Performance", "Mission & Data", "Methodology", "About"]) {
      expect(nav).toHaveTextContent(label);
    }
  });

  it("navigates to an unbuilt module and says plainly that it is not available", async () => {
    stubApi({ "/health": [200, HEALTH], "/ready": [503, NOT_READY] });
    renderApp("/");
    await userEvent.click(screen.getByRole("link", { name: /Analyze/ }));
    expect(screen.getByRole("heading", { name: "Analyze", level: 1 })).toBeInTheDocument();
    expect(screen.getByText("Not available in this build")).toBeInTheDocument();
    expect(screen.getByText(/trained U-Net checkpoint/i)).toBeInTheDocument();
  });

  it("shows a not-found page for unknown routes", () => {
    stubApi({ "/health": [200, HEALTH], "/ready": [503, NOT_READY] });
    renderApp("/does-not-exist");
    expect(screen.getByRole("heading", { name: "Page not found" })).toBeInTheDocument();
  });
});

describe("theme", () => {
  it("defaults to dark, toggles to light and persists the choice", async () => {
    stubApi({ "/health": [200, HEALTH], "/ready": [503, NOT_READY] });
    renderApp("/");
    expect(document.documentElement).toHaveAttribute("data-theme", "dark");
    await userEvent.click(screen.getByRole("button", { name: "Switch to light theme" }));
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
    expect(window.localStorage.getItem(THEME_STORAGE_KEY)).toBe("light");
    expect(screen.getByRole("button", { name: "Switch to dark theme" })).toBeInTheDocument();
  });

  it("restores a saved theme on load", () => {
    window.localStorage.setItem(THEME_STORAGE_KEY, "light");
    stubApi({ "/health": [200, HEALTH], "/ready": [503, NOT_READY] });
    renderApp("/");
    expect(document.documentElement).toHaveAttribute("data-theme", "light");
  });
});

describe("pipeline status (overview)", () => {
  it("reports an unloaded model honestly, with the backend's reason", async () => {
    stubApi({ "/health": [200, HEALTH], "/ready": [503, NOT_READY] });
    renderApp("/");
    await waitFor(() => expect(screen.getByText(/Online · v0.1.0 · test/)).toBeInTheDocument());
    expect(screen.getByText(/Not loaded\./)).toBeInTheDocument();
    expect(screen.getByText(/No model checkpoint found/)).toBeInTheDocument();
    expect(screen.getByText("Unavailable until a trained model is loaded.")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("API online");
  });

  it("shows the API as offline when it cannot be reached", async () => {
    stubApi("network-error");
    renderApp("/");
    await waitFor(() => expect(screen.getByRole("status")).toHaveTextContent("API offline"));
    expect(screen.getByText("Could not reach the StratoSense API.")).toBeInTheDocument();
  });

  it("shows a loaded model when the backend reports ready", async () => {
    stubApi({
      "/health": [200, HEALTH],
      "/ready": [200, { ready: true, model_loaded: true, model_version: "1.0.0", device: "cpu", reason: null }],
    });
    renderApp("/");
    await waitFor(() => expect(screen.getByText(/Loaded · v1.0.0 · cpu/)).toBeInTheDocument());
  });

  it("re-checks the API when asked", async () => {
    const fetchMock = stubApi({ "/health": [200, HEALTH], "/ready": [503, NOT_READY] });
    renderApp("/");
    await waitFor(() => expect(screen.getByText(/Online/)).toBeInTheDocument());
    const before = fetchMock.mock.calls.length;
    await userEvent.click(screen.getByRole("button", { name: /Check again/ }));
    await waitFor(() => expect(fetchMock.mock.calls.length).toBeGreaterThan(before));
  });
});
