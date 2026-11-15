import { fmt, useFetch } from "../api.js";
import { ErrorState, Loading, Section } from "../components/common.jsx";

export default function Methodology() {
  const { data, error, loading } = useFetch("/metrics");
  if (loading) return <Loading what="methodology" />;
  if (error) return <ErrorState error={error} />;
  const parts = data.split.partitions;
  const leak = data.leakage_checks;
  const iso = data.isolation_forest;
  return (
    <>
      <h1>Methodology</h1>
      <Section title="Data">
        <p>
          The Elliptic Bitcoin dataset has 203,769 transactions and 234,355 directed fund-flow edges across 49 time
          steps. A node is one transaction, not an account or person. Edges exist only within a time step, so no
          money flow is observed between steps.
        </p>
      </Section>
      <Section title="Chronological split">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Period</th><th>Steps</th><th className="num">Transactions</th><th className="num">Illicit</th><th className="num">Licit</th><th className="num">Illicit share of known</th></tr></thead>
            <tbody>
              {Object.entries(parts).map(([name, part]) => (
                <tr key={name}>
                  <td>{name}</td><td>{part.steps.join("-")}</td><td className="num">{fmt.count(part.nodes)}</td>
                  <td className="num">{fmt.count(part.illicit)}</td><td className="num">{fmt.count(part.licit)}</td>
                  <td className="num">{fmt.percent(part.illicit_share_of_known)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <Section title="Leakage controls">
        <ul className="limits">
          <li>Scaling is fitted on {fmt.count(data.split.processor_fit_rows)} labeled training rows only.</li>
          <li>
            The publishers standardized the features across all 49 steps before release. This cannot be undone and is
            a disclosed limitation.
          </li>
          <li>
            Time step is never a model input. {data.time_proxies.names.length} columns ({data.time_proxies.names.join(", ")}) act
            as hidden time steps, so they were removed by a rule measured on training rows.
          </li>
          <li>
            With shuffled training labels, validation PR-AUC falls to{" "}
            {leak.shuffled_label_val_pr_auc.map((v) => v.toFixed(3)).join(", ")}, near the base rate of{" "}
            {leak.validation_base_rate.toFixed(3)}. This shows no label leakage in the pipeline.
          </li>
        </ul>
      </Section>
      <Section title="Unknown labels">
        <p>
          Unknown transactions pass messages in graph models but never contribute to loss or metrics, and they are
          never treated as safe.
        </p>
      </Section>
      <Section title="Training">
        <p>
          Each neural model trains for exactly {data.training.epochs} epochs with no early stopping, and keeps the
          weights from its best validation epoch. EvolveGCN updates its weights about every five time steps, so it
          gets 6 updates per epoch where the static neural models get 30. The same epoch count therefore gives
          temporal models less training. Tree and linear models do not train in epochs.
        </p>
      </Section>
      <Section title="What makes a model temporal here">
        <p>
          EvolveGCN processes the steps in order and evolves its graph-convolution weights with a recurrent cell.
          That state belongs to the model. It does not follow any transaction, wallet or customer between steps.
        </p>
      </Section>
      <Section title="Anomaly overlay">
        <p>
          Isolation Forest was fitted without labels on training-period transactions. Its test PR-AUC is{" "}
          {iso.test_pr_auc.toFixed(3)} against a base rate of {iso.test_base_rate.toFixed(3)}. It is shown as a
          percentile and is never merged into the risk score.
        </p>
      </Section>
      <Section title="Limitations">
        <ul className="limits">
          <li>This is illicit-transaction classification on historical Bitcoin data, not bank or card fraud detection.</li>
          <li>Labels are incomplete, and unknown transactions are not random negatives.</li>
          <li>Features are anonymized, so column names carry no business meaning.</li>
          <li>Scores are uncalibrated and are not probabilities of crime.</li>
          <li>Results depend on the seed, the split and the threshold.</li>
          <li>A benchmark result does not show real-world or regulatory suitability.</li>
          <li>This tool supports analyst prioritization. It must not accuse, freeze or punish anyone.</li>
        </ul>
      </Section>
    </>
  );
}
