import { useEffect, useState } from "react";
import { MODEL_NAMES, useFetch } from "./api.js";
import Benchmark from "./pages/Benchmark.jsx";
import Investigate from "./pages/Investigate.jsx";
import Methodology from "./pages/Methodology.jsx";
import Overview from "./pages/Overview.jsx";
import Queue from "./pages/Queue.jsx";

const PAGES = [
  ["overview", "Overview"],
  ["queue", "Risk queue"],
  ["investigate", "Investigate"],
  ["benchmark", "Benchmark lab"],
  ["methodology", "Methodology"],
];

function parseHash() {
  const [page = "overview", id = null] = window.location.hash.replace(/^#\/?/, "").split("/");
  return { page: PAGES.some(([key]) => key === page) ? page : "overview", id };
}

export default function App() {
  const [route, setRoute] = useState(parseHash());
  const health = useFetch("/health");

  useEffect(() => {
    const onChange = () => setRoute(parseHash());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
  }, []);

  const ready = health.data?.status === "ok";
  return (
    <div className="shell">
      <header className="topbar">
        <a className="brand" href="#/overview">TraceIQ</a>
        <nav aria-label="Main">
          {PAGES.map(([key, label]) => (
            <a key={key} href={`#/${key}`} className={route.page === key ? "active" : ""}>{label}</a>
          ))}
        </nav>
        <div className="model-chip" title={health.data?.run}>
          {ready ? `Model: ${MODEL_NAMES[health.data.model] ?? health.data.model}` : "Model: not loaded"}
        </div>
      </header>
      <main>
        {health.loading && <p className="state">Connecting to the local API…</p>}
        {health.error && (
          <div className="state error">
            The local API is not reachable. Start it with <code>uvicorn src.api.main:app --port 8000</code>.
          </div>
        )}
        {health.data && !ready && (
          <div className="state error">
            No index found. Run <code>python -m src.api.build_index</code>, then reload.
          </div>
        )}
        {ready && route.page === "overview" && <Overview />}
        {ready && route.page === "queue" && <Queue />}
        {ready && route.page === "investigate" && <Investigate id={route.id} />}
        {ready && route.page === "benchmark" && <Benchmark />}
        {ready && route.page === "methodology" && <Methodology />}
      </main>
      <footer>
        Research prototype on historical, anonymized Bitcoin data. Scores are model estimates for analyst
        prioritization and are not findings of crime.
      </footer>
    </div>
  );
}
