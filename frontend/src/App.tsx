import { lazy, Suspense } from "react";
import { NavLink, Route, Routes } from "react-router-dom";
import { Loading } from "./components/common";

// Route-level code splitting: recharts (the bulk of the bundle) only loads with the overview.
const Dashboard = lazy(() => import("./pages/Dashboard").then((m) => ({ default: m.Dashboard })));
const LotsPage = lazy(() => import("./pages/LotsPage").then((m) => ({ default: m.LotsPage })));
const LotDetailPage = lazy(() =>
  import("./pages/LotDetailPage").then((m) => ({ default: m.LotDetailPage })),
);
const WaferDetailPage = lazy(() =>
  import("./pages/WaferDetailPage").then((m) => ({ default: m.WaferDetailPage })),
);
const ClassifyPage = lazy(() =>
  import("./pages/ClassifyPage").then((m) => ({ default: m.ClassifyPage })),
);

export function App() {
  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="brand__logo" aria-hidden />
          Wafer-map Defect Analytics
        </div>
        <nav>
          <NavLink to="/" end>
            Overview
          </NavLink>
          <NavLink to="/lots">Lots</NavLink>
          <NavLink to="/classify">Classify</NavLink>
          <a href="/docs" target="_blank" rel="noreferrer">
            API docs
          </a>
        </nav>
      </header>
      <main>
        <Suspense fallback={<Loading />}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/lots" element={<LotsPage />} />
            <Route path="/lots/:lotName" element={<LotDetailPage />} />
            <Route path="/wafers/:waferId" element={<WaferDetailPage />} />
            <Route path="/classify" element={<ClassifyPage />} />
            <Route path="*" element={<p>Page not found.</p>} />
          </Routes>
        </Suspense>
      </main>
      <footer className="footer">
        Data: WM-811K (Wu, Jang &amp; Chen, IEEE TSM 2015). Portfolio project, not affiliated with
        any manufacturer.
      </footer>
    </div>
  );
}
