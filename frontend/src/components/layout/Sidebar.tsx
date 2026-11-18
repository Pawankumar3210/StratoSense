import { X } from "lucide-react";
import { NavLink } from "react-router-dom";
import { Logo } from "@/components/common/Logo";
import { NAV_ITEMS } from "./navigation";

export function Sidebar({ open, onClose }: { open: boolean; onClose: () => void }) {
  return (
    <>
      {open && (
        <button
          type="button"
          aria-label="Close navigation"
          className="fixed inset-0 z-30 bg-black/50 lg:hidden"
          onClick={onClose}
        />
      )}
      <aside
        className={`${
          open ? "fixed inset-y-0 left-0 z-40 flex" : "hidden"
        } w-72 shrink-0 flex-col border-r border-line bg-raised lg:static lg:flex`}
      >
        <div className="flex h-16 items-center justify-between px-5">
          <Logo />
          <button
            type="button"
            aria-label="Close navigation"
            className="rounded p-1.5 text-muted hover:text-ink lg:hidden"
            onClick={onClose}
          >
            <X size={18} />
          </button>
        </div>
        <nav aria-label="Primary" className="flex-1 overflow-y-auto px-3 pb-6">
          <ul className="space-y-0.5">
            {NAV_ITEMS.map(({ id, path, label, icon: Icon, status }) => (
              <li key={id}>
                <NavLink
                  to={path}
                  end={path === "/"}
                  className={({ isActive }) =>
                    `flex items-center gap-3 rounded-md px-3 py-2 text-sm transition-colors ${
                      isActive ? "bg-active text-ink" : "text-muted hover:bg-active/60 hover:text-ink"
                    }`
                  }
                >
                  {({ isActive }) => (
                    <>
                      <Icon size={17} className={isActive ? "text-accent" : undefined} aria-hidden />
                      <span className="flex-1">{label}</span>
                      {status === "planned" && (
                        <span className="rounded border border-line px-1.5 py-px text-[11px] text-muted">
                          Planned
                        </span>
                      )}
                    </>
                  )}
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
      </aside>
    </>
  );
}
