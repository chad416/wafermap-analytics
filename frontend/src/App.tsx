import { NavLink, Route, Routes } from "react-router-dom";
import { ClassifyPage } from "./pages/ClassifyPage";
import { Dashboard } from "./pages/Dashboard";
import { LotDetailPage } from "./pages/LotDetailPage";
import { LotsPage } from "./pages/LotsPage";
import { WaferDetailPage } from "./pages/WaferDetailPage";

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
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/lots" element={<LotsPage />} />
          <Route path="/lots/:lotName" element={<LotDetailPage />} />
          <Route path="/wafers/:waferId" element={<WaferDetailPage />} />
          <Route path="/classify" element={<ClassifyPage />} />
          <Route path="*" element={<p>Page not found.</p>} />
        </Routes>
      </main>
      <footer className="footer">
        Data: WM-811K (Wu, Jang &amp; Chen, IEEE TSM 2015). Portfolio project, not affiliated with
        any manufacturer.
      </footer>
    </div>
  );
}
