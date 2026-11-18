import { Menu, Moon, Sun } from "lucide-react";
import { useApiStatusContext } from "@/contexts/ApiStatusContext";
import { useTheme } from "@/contexts/ThemeContext";

function ApiPill() {
  const { status } = useApiStatusContext();
  const map = {
    checking: { dot: "bg-muted", text: "Checking API" },
    offline: { dot: "bg-bad", text: "API offline" },
    online: { dot: "bg-ok", text: "API online" },
  } as const;
  const { dot, text } = map[status.state];
  return (
    <span
      className="inline-flex items-center gap-2 rounded-full border border-line px-3 py-1 text-xs text-muted"
      role="status"
    >
      <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden />
      {text}
    </span>
  );
}

export function Topbar({ onMenu }: { onMenu: () => void }) {
  const { theme, toggleTheme } = useTheme();
  const next = theme === "dark" ? "light" : "dark";
  return (
    <header className="flex h-16 shrink-0 items-center justify-between border-b border-line bg-raised px-4 lg:px-8">
      <button
        type="button"
        aria-label="Open navigation"
        className="rounded p-2 text-muted hover:text-ink lg:hidden"
        onClick={onMenu}
      >
        <Menu size={20} />
      </button>
      <div className="hidden lg:block" />
      <div className="flex items-center gap-3">
        <ApiPill />
        <button
          type="button"
          onClick={toggleTheme}
          aria-label={`Switch to ${next} theme`}
          className="rounded-md border border-line p-2 text-muted transition-colors hover:text-ink"
        >
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
      </div>
    </header>
  );
}
