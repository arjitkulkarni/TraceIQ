import { useCallback, useState } from "react";
import { MODEL_NAMES, fmt, useFetch } from "../api.js";
import GraphView from "../components/GraphView.jsx";
import { ErrorState, LabelBadge, Loading, Section, TierBadge } from "../components/common.jsx";

function Legend() {
  return (
    <div className="legend">
      <span><i className="key shape-triangle" /> Dataset: illicit</span>
      <span><i className="key shape-square" /> Dataset: licit</span>
      <span><i className="key shape-circle" /> Dataset: unknown</span>
      <span><i className="key border-high" /> Border: high tier</span>
      <span><i className="key border-medium" /> Border: medium tier</span>
      <span>Arrow: real fund flow, source to destination</span>
      <span>Hover a node for details, click to open it</span>
      <span>Thick edge: larger measured logit drop</span>
    </div>
  );
}

function Explanation({ id }) {
  const { data, error, loading } = useFetch(`/transactions/${id}/explanation`);
  if (loading) return <Loading what="explanation" />;
  if (error) return <ErrorState error={error} />;
  if (!data.available) {
    return (
      <Section title="Model evidence">
        <p className="state">No explanation was precomputed for this transaction. Pick one marked Available in the risk queue.</p>
      </Section>
    );
  }
  return (
    <Section title="Model evidence"
      note={`${data.method} applied to the ${data.explained_model} model (${MODEL_NAMES[data.model_name] ?? data.model_name}), averaged over ${data.stability.repeats} runs. Measured logit drop is the real change in the model output when that one edge is deleted.`}>
      <div className="two-col">
        <div>
          <h3>Edges with the largest measured effect</h3>
          <div className="table-wrap"><table>
            <thead><tr><th>Source</th><th>Destination</th><th className="num">Importance</th><th className="num">Measured logit drop</th></tr></thead>
            <tbody>
              {data.edges.slice(0, 8).map((edge) => (
                <tr key={`${edge.source}-${edge.target}`}>
                  <td><a href={`#/investigate/${edge.source}`}>{edge.source}</a></td>
                  <td><a href={`#/investigate/${edge.target}`}>{edge.target}</a></td>
                  <td className="num">{edge.importance.toFixed(3)} ± {edge.importance_std.toFixed(3)}</td>
                  <td className="num">{edge.occlusion_logit_drop.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table></div>
        </div>
        <div>
          <h3>Most influential features</h3>
          <div className="table-wrap"><table>
            <thead><tr><th>Column</th><th>Group</th><th className="num">Importance</th></tr></thead>
            <tbody>
              {data.features.slice(0, 8).map((feature) => (
                <tr key={feature.feature}>
                  <td>{feature.feature}</td><td>{feature.group}</td>
                  <td className="num">{feature.importance.toFixed(3)} ± {feature.importance_std.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table></div>
        </div>
      </div>
      <dl className="facts compact">
        <div><dt>Logit with all edges</dt><dd>{data.fidelity.logit_original.toFixed(2)}</dd></div>
        <div><dt>Logit without top 5 edges</dt><dd>{data.fidelity.logit_without_top_edges.toFixed(2)}</dd></div>
        <div><dt>Logit with only top 5 edges</dt><dd>{data.fidelity.logit_only_top_edges.toFixed(2)}</dd></div>
        <div><dt>Logit without any edges</dt><dd>{data.fidelity.logit_without_any_edges.toFixed(2)}</dd></div>
        <div><dt>Edge agreement across runs</dt><dd>{fmt.score(data.stability.edge_top5_jaccard)}</dd></div>
        <div><dt>Feature agreement across runs</dt><dd>{fmt.score(data.stability.feature_top5_jaccard)}</dd></div>
        <div><dt>Importance vs measured drop (rank)</dt><dd>{data.occlusion_spearman == null ? "Too few edges" : data.occlusion_spearman.toFixed(2)}</dd></div>
      </dl>
      <ul className="limits">{data.limitations.map((text) => <li key={text}>{text}</li>)}</ul>
      <p className="disclaimer">{data.disclaimer}</p>
    </Section>
  );
}

function Detail({ id }) {
  const [hops, setHops] = useState(1);
  const detail = useFetch(`/transactions/${id}`);
  const graph = useFetch(`/transactions/${id}/graph?hops=${hops}`);
  const select = useCallback((next) => { window.location.hash = `#/investigate/${next}`; }, []);
  if (detail.loading && !detail.data) return <Loading what="transaction" />;
  if (detail.error) return <ErrorState error={detail.error} />;
  const tx = detail.data;
  return (
    <>
      <h1>Transaction {tx.transaction_id}</h1>
      <div className="investigate">
        <Section title="Directed neighborhood"
          note={`Real dataset edges in time step ${tx.time_step}. Edges never cross time steps in this dataset.`}>
          <div className="toolbar">
            {[1, 2].map((value) => (
              <button key={value} className={hops === value ? "on" : ""} onClick={() => setHops(value)}>{value} hop{value > 1 ? "s" : ""}</button>
            ))}
            {graph.data && (
              <span className="note">
                {graph.data.nodes.length} nodes, {graph.data.edges.length} edges
                {graph.data.truncated ? ` (limited to ${graph.data.max_nodes} nodes)` : ""}
              </span>
            )}
          </div>
          {graph.error && <ErrorState error={graph.error} />}
          {graph.loading && !graph.data && <Loading what="neighborhood" />}
          {graph.data && <GraphView graph={graph.data} onSelect={select} />}
          <Legend />
        </Section>
        <Section title="Details">
          <dl className="facts">
            <div><dt>Model score</dt><dd>{fmt.score(tx.model_score)}</dd></div>
            <div><dt>Model-estimated risk</dt><dd><TierBadge tier={tx.risk_tier} /></dd></div>
            <div><dt>Dataset label</dt><dd><LabelBadge label={tx.dataset_label} /></dd></div>
            <div><dt>Time step</dt><dd>{tx.time_step} ({tx.split})</dd></div>
            <div><dt>Alert threshold</dt><dd>{fmt.score(tx.alert_threshold)}</dd></div>
            <div><dt>Score type</dt><dd>{tx.score_type.replace("_", " ")}</dd></div>
            <div><dt>Incoming edges</dt><dd>{tx.in_degree}</dd></div>
            <div><dt>Outgoing edges</dt><dd>{tx.out_degree}</dd></div>
            <div><dt>XGBoost score</dt><dd>{fmt.score(tx.tabular_comparison_score)}</dd></div>
            <div><dt>Anomaly percentile</dt><dd>{fmt.percent(tx.anomaly_percentile)}</dd></div>
            <div><dt>Model</dt><dd>{MODEL_NAMES[tx.model_name] ?? tx.model_name}</dd></div>
          </dl>
          <p className="note">The score is not a probability of crime. Anomaly percentile means unusual compared with training-period transactions, not illicit.</p>
        </Section>
      </div>
      <Explanation id={id} />
    </>
  );
}

export default function Investigate({ id }) {
  const top = useFetch(id ? null : "/transactions?split=test&explained=true&page_size=1");
  if (id) return <Detail key={id} id={id} />;
  if (top.loading) return <Loading what="the top transaction" />;
  if (top.error) return <ErrorState error={top.error} />;
  const first = top.data?.items?.[0];
  if (!first) return <p className="state">No transaction selected. Pick one from the risk queue.</p>;
  return <Detail key={first.transaction_id} id={first.transaction_id} />;
}
