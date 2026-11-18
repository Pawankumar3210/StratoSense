import { Route, Routes } from "react-router-dom";
import { AppShell } from "@/components/layout/AppShell";
import { NAV_ITEMS } from "@/components/layout/navigation";
import NotFound from "@/pages/NotFound";
import Overview from "@/pages/Overview";
import PlannedPage from "@/pages/PlannedPage";

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<Overview />} />
        {NAV_ITEMS.filter((n) => n.status === "planned").map((n) => (
          <Route key={n.id} path={n.path} element={<PlannedPage id={n.id} />} />
        ))}
        <Route path="*" element={<NotFound />} />
      </Route>
    </Routes>
  );
}
