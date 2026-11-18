import { Link } from "react-router-dom";

export default function NotFound() {
  return (
    <div className="mx-auto max-w-2xl">
      <h1 className="text-3xl font-semibold tracking-tight">Page not found</h1>
      <p className="mt-4 text-muted">That address doesn’t match any StratoSense page.</p>
      <Link to="/" className="mt-6 inline-block text-accent underline underline-offset-4">
        Go to the overview
      </Link>
    </div>
  );
}
