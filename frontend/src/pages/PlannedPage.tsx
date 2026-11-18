import { Navigate } from "react-router-dom";
import { NAV_ITEMS } from "@/components/layout/navigation";

/** Honest placeholder for a module that is not built yet: says what it will do and what it needs. */
export default function PlannedPage({ id }: { id: string }) {
  const item = NAV_ITEMS.find((n) => n.id === id);
  if (!item) return <Navigate to="/" replace />;
  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="text-3xl font-semibold tracking-tight">{item.label}</h1>
      <p className="mt-4 leading-relaxed text-muted">{item.summary}</p>
      <section className="mt-10" aria-labelledby="planned-heading">
        <h2 id="planned-heading" className="text-base font-semibold">
          Not available in this build
        </h2>
        <p className="mt-2 text-sm text-muted">This module needs:</p>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm">
          {item.needs.map((need) => (
            <li key={need}>{need}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
