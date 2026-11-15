import { useMemo, useState } from "react";
import { fmt, useFetch } from "../api.js";
import { ErrorState, LabelBadge, Loading, TierBadge } from "../components/common.jsx";

const PAGE_SIZE = 25;

function Select({ label, value, onChange, options }) {
  return (
    <label className="filter">
      {label}
      <select value={value} onChange={(event) => onChange(event.target.value)}>
        {options.map(([key, text]) => <option key={key} value={key}>{text}</option>)}
      </select>
    </label>
  );
}

export default function Queue() {
  const [filters, setFilters] = useState({ split: "test", tier: "", label: "", time_step: "", explained: "" });
  const [page, setPage] = useState(1);
  const update = (key) => (value) => {
    setFilters((previous) => ({ ...previous, [key]: value }));
    setPage(1);
  };
  const path = useMemo(() => {
    const query = new URLSearchParams({ sort: "risk", page, page_size: PAGE_SIZE });
    Object.entries(filters).forEach(([key, value]) => value && query.set(key, value));
    return `/transactions?${query}`;
  }, [filters, page]);
  const { data, error, loading } = useFetch(path);
  const pages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;
  const steps = [["", "All steps"], ...Array.from({ length: 49 }, (_, i) => [String(i + 1), `Step ${i + 1}`])];

  return (
    <>
      <h1>Risk queue</h1>
      <p className="lede">Transactions ranked by model score. Model-estimated risk is separate from the dataset label.</p>
      <div className="filters">
        <Select label="Period" value={filters.split} onChange={update("split")}
          options={[["test", "Test (steps 40-49)"], ["val", "Validation"], ["train", "Train"], ["", "All"]]} />
        <Select label="Risk tier" value={filters.tier} onChange={update("tier")}
          options={[["", "All tiers"], ["high", "High"], ["medium", "Medium"], ["low", "Low"]]} />
        <Select label="Dataset label" value={filters.label} onChange={update("label")}
          options={[["", "All labels"], ["illicit", "Illicit"], ["licit", "Licit"], ["unknown", "Unknown"]]} />
        <Select label="Time step" value={filters.time_step} onChange={update("time_step")} options={steps} />
        <Select label="Explanation" value={filters.explained} onChange={update("explained")}
          options={[["", "Any"], ["true", "Available"], ["false", "Not available"]]} />
      </div>
      {error && <ErrorState error={error} />}
      {loading && !data && <Loading what="transactions" />}
      {data && data.total === 0 && <p className="state">No transactions match these filters.</p>}
      {data && data.total > 0 && (
        <>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th className="num">Rank</th><th>Transaction</th><th className="num">Step</th>
                  <th className="num">Model score</th><th>Risk tier</th><th>Dataset label</th><th>Explanation</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((item, index) => (
                  <tr key={item.transaction_id} onClick={() => { window.location.hash = `#/investigate/${item.transaction_id}`; }}>
                    <td className="num">{(page - 1) * PAGE_SIZE + index + 1}</td>
                    <td><a href={`#/investigate/${item.transaction_id}`}>{item.transaction_id}</a></td>
                    <td className="num">{item.time_step}</td>
                    <td className="num">{fmt.score(item.model_score)}</td>
                    <td><TierBadge tier={item.risk_tier} /></td>
                    <td><LabelBadge label={item.dataset_label} /></td>
                    <td>{item.explanation_available ? "Available" : "Not computed"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pager">
            <button disabled={page <= 1} onClick={() => setPage(page - 1)}>Previous</button>
            <span>Page {fmt.count(page)} of {fmt.count(pages)} · {fmt.count(data.total)} transactions</span>
            <button disabled={page >= pages} onClick={() => setPage(page + 1)}>Next</button>
          </div>
        </>
      )}
    </>
  );
}
