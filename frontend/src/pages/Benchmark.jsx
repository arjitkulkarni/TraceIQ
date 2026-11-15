import { useState } from "react";
import { MODEL_NAMES, fmt, useFetch } from "../api.js";
import { DotRange, EpochLines, StepLines } from "../components/charts.jsx";
import { ErrorState, Loading, Section } from "../components/common.jsx";

const PAIR_NAMES = {
  graphsage_minus_xgboost: "GraphSAGE vs XGBoost",
  graphsage_minus_random_forest: "GraphSAGE vs Random Forest",
  graphsage_minus_mlp: "GraphSAGE vs MLP (adding edges)",
  evolvegcn_o_minus_static_gcn: "EvolveGCN-O vs Static GCN (adding temporal state)",
  evolvegcn_h_minus_static_gcn: "EvolveGCN-H vs Static GCN (adding temporal state)",
  evolvegcn_o_minus_graphsage: "EvolveGCN-O vs GraphSAGE",
  evolvegcn_h_minus_graphsage: "EvolveGCN-H vs GraphSAGE",
};

const NEURAL_MODELS = ["mlp", "graphsage", "static_gcn", "evolvegcn_o", "evolvegcn_h"];
const EPOCH_METRICS = {
  val_accuracy: "Accuracy",
  val_f1: "F1",
  val_pr_auc: "PR-AUC",
  train_loss: "Training loss",
};

const n = (value, digits = 3) => Number(value).toFixed(digits);

/** The accuracy axis starts just below the weakest result, so close models stay distinct. */
function accuracyAxis(rows, floor) {
  const low = Math.floor(Math.min(floor, ...rows.map((row) => row.mean - row.std)) * 50) / 50;
  const step = (1 - low) / 4;
  return { domain: [low, 1], ticks: [0, 1, 2, 3, 4].map((i) => Number((low + i * step).toFixed(3))) };
}

function TrainingCurves({ data, baseRate }) {
  const [metric, setMetric] = useState("val_accuracy");
  const [algorithm, setAlgorithm] = useState("all");
  const curves = data.epoch_curves;
  const epochs = Array.from({ length: data.training.epochs }, (_, i) => i + 1);
  const toValues = (rows) => Object.fromEntries(rows.map((row) => [Number(row.epoch), Number(row[metric])]));
  const series = algorithm === "all"
    ? NEURAL_MODELS.map((model) => {
      const { feature_set: features, edge_mode: edges } = data.selection.families[model];
      return {
        id: `model-${model}`,
        name: MODEL_NAMES[model],
        values: toValues(curves.filter((row) => row.model === model && row.feature_set === features && row.edge_mode === edges)),
      };
    })
    : [...new Set(curves.filter((row) => row.model === algorithm).map((row) => `${row.feature_set}|${row.edge_mode}`))]
      .sort()
      .map((variant, index) => {
        const [features, edges] = variant.split("|");
        return {
          id: `series-${index + 1}`,
          name: edges === "none" ? features : `${features}, ${edges}`,
          values: toValues(curves.filter((row) => row.model === algorithm && row.feature_set === features && row.edge_mode === edges)),
        };
      });
  const isLoss = metric === "train_loss";
  const digits = isLoss ? 2 : 3;
  const marks = [...new Set([1, 5, 10, 15, epochs.length])].filter((epoch) => epoch <= epochs.length);
  const bestOf = (values) => epochs.reduce(
    (best, epoch) => ((isLoss ? values[epoch] < values[best] : values[epoch] > values[best]) ? epoch : best), 1,
  );
  const reference = metric === "val_accuracy"
    ? { value: 1 - baseRate, label: `Always guessing licit: ${n(1 - baseRate)}` }
    : undefined;
  return (
    <>
      <div className="chart-controls">
        <label>Metric
          <select value={metric} onChange={(event) => setMetric(event.target.value)}>
            {Object.entries(EPOCH_METRICS).map(([key, name]) => <option key={key} value={key}>{name}</option>)}
          </select>
        </label>
        <label>Show
          <select value={algorithm} onChange={(event) => setAlgorithm(event.target.value)}>
            <option value="all">All neural models (best variant of each)</option>
            {NEURAL_MODELS.map((model) => <option key={model} value={model}>{MODEL_NAMES[model]}: every variant</option>)}
          </select>
        </label>
      </div>
      <EpochLines series={series} epochs={epochs} label={EPOCH_METRICS[metric]} reference={reference}
        unit={!isLoss} format={(value) => n(value, digits)} />
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>{algorithm === "all" ? "Model" : "Variant"}</th>
              {marks.map((epoch) => <th key={epoch} className="num">Epoch {epoch}</th>)}
              <th className="num">Best</th><th className="num">At epoch</th>
            </tr>
          </thead>
          <tbody>
            {[...series].sort((a, b) => (b.values[epochs.length] - a.values[epochs.length]) * (isLoss ? -1 : 1)).map((s) => {
              const best = bestOf(s.values);
              return (
                <tr key={s.id}>
                  <td>{s.name}</td>
                  {marks.map((epoch) => <td key={epoch} className="num">{n(s.values[epoch], digits)}</td>)}
                  <td className="num">{n(s.values[best], digits)}</td><td className="num">{best}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

export default function Benchmark() {
  const { data, error, loading } = useFetch("/metrics");
  if (loading) return <Loading what="benchmark results" />;
  if (error) return <ErrorState error={error} />;

  const rows = [...data.comparison].sort((a, b) => b.pr_auc_mean - a.pr_auc_mean);
  const accuracyRows = [...data.comparison]
    .sort((a, b) => b.accuracy_mean - a.accuracy_mean)
    .map((row) => ({ name: row.model, mean: Number(row.accuracy_mean), std: Number(row.accuracy_std) }));
  const testBaseRate = data.confusion[0].illicit_nodes / data.confusion[0].known_nodes;
  const testFloor = 1 - testBaseRate;
  const lineFamilies = ["xgboost", "graphsage", "evolvegcn_o"];
  const xs = Array.from({ length: 10 }, (_, i) => 40 + i);
  const series = lineFamilies.map((family) => {
    const values = {};
    xs.forEach((step) => {
      const group = data.per_step.filter((row) => row.family === family && Number(row.time_step) === step);
      values[step] = group.reduce((sum, row) => sum + Number(row.recall), 0) / group.length;
    });
    return { name: MODEL_NAMES[family], values };
  });
  const periods = {};
  data.periods.forEach((row) => { (periods[row.model] ??= {})[row.period] = row; });

  return (
    <>
      <h1>Benchmark lab</h1>
      <p className="lede">
        All models share one chronological split and identical metric code. Each neural model trained for
        exactly {data.training.epochs} epochs. Tree and linear models have no epochs, so they are shown as
        reference points. Variants and thresholds were frozen on validation before test labels were read.
        Test period: {fmt.count(data.confusion[0].known_nodes)} known transactions,{" "}
        {data.confusion[0].illicit_nodes} illicit.
      </p>
      <Section title="Test accuracy by model"
        note={`Share of known transactions classified correctly, as mean and standard deviation across seeds. The axis is zoomed in. Only ${fmt.percent(testBaseRate)} of test transactions are illicit, so always guessing licit scores ${n(testFloor)} while catching nothing. Read accuracy alongside F1 and PR-AUC.`}>
        <DotRange rows={accuracyRows} label="Accuracy" {...accuracyAxis(accuracyRows, testFloor)}
          reference={{ value: testFloor, label: "Always guessing licit" }} />
      </Section>
      <Section title={`Training over ${data.training.epochs} epochs`}
        note="Validation results after each epoch, averaged across seeds. Each epoch uses its own best-F1 threshold. Test results are not used here, so model choices stay validation-only.">
        <TrainingCurves data={data} baseRate={data.leakage_checks.validation_base_rate} />
      </Section>
      <Section title="Test PR-AUC by model" note="Mean and standard deviation across seeds.">
        <DotRange rows={rows.map((row) => ({ name: row.model, mean: Number(row.pr_auc_mean), std: Number(row.pr_auc_std) }))} />
      </Section>
      <Section title="Test results at the frozen threshold"
        note="The test period has 636 illicit transactions, so Recall@100 cannot exceed 0.157. A model at 0.157 has only illicit transactions in its top 100.">
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Model</th><th>Features</th><th>Edges</th><th className="num">Best epoch</th><th className="num">Accuracy</th>
                <th className="num">Precision</th><th className="num">Recall</th>
                <th className="num">F1</th><th className="num">PR-AUC</th><th className="num">Recall@100</th><th className="num">Alerts</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.family}>
                  <td>{row.model}</td><td>{row.feature_set}</td><td>{row.edge_mode}</td>
                  <td className="num">{row.best_epoch_mean ? n(row.best_epoch_mean, 1) : "No epochs"}</td>
                  <td className="num">{n(row.accuracy_mean)} ± {n(row.accuracy_std)}</td>
                  <td className="num">{n(row.precision_mean)}</td><td className="num">{n(row.recall_mean)}</td>
                  <td className="num">{n(row.f1_mean)}</td>
                  <td className="num">{n(row.pr_auc_mean)} ± {n(row.pr_auc_std)}</td>
                  <td className="num">{n(row.recall_at_100_mean)}</td><td className="num">{n(row.alerts_mean, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <Section title="Recall per test step" note="Known illicit counts fall sharply from step 43, so late steps are noisy. Step 46 has only 2 illicit transactions.">
        <StepLines series={series} xs={xs} />
      </Section>
      <Section title="Before and after step 43" note="PR-AUC within each part of the test period.">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Model</th><th className="num">Steps 40-42</th><th className="num">Steps 43-49</th></tr></thead>
            <tbody>
              {Object.entries(periods).map(([model, row]) => (
                <tr key={model}>
                  <td>{model}</td>
                  <td className="num">{n(row.steps_40_42.pr_auc)}</td>
                  <td className="num">{n(row.steps_43_49.pr_auc)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      <Section title="Is the difference real?" note="Paired bootstrap over test transactions. An interval that includes zero means no clear difference.">
        <div className="table-wrap">
          <table>
            <thead><tr><th>Comparison</th><th className="num">PR-AUC difference</th><th className="num">95% interval</th><th>Result</th></tr></thead>
            <tbody>
              {Object.entries(data.significance).map(([key, row]) => (
                <tr key={key}>
                  <td>{PAIR_NAMES[key] ?? key}</td>
                  <td className="num">{n(row.difference)}</td>
                  <td className="num">{n(row.ci95_low)} to {n(row.ci95_high)}</td>
                  <td>{row.ci_excludes_zero ? (row.difference > 0 ? "First model higher" : "Second model higher") : "No clear difference"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
      {data.explanation_summary && (
        <Section title="How far to trust the explanations"
          note={`Measured on ${data.explanation_summary.explained} held-out transactions explained with the deployed model.`}>
          <dl className="facts compact">
            <div><dt>Rank agreement, importance vs measured effect (median)</dt><dd>{n(data.explanation_summary.occlusion_spearman_median, 2)}</dd></div>
            <div><dt>Top-5 edge agreement across runs</dt><dd>{n(data.explanation_summary.mean_edge_top5_jaccard, 2)}</dd></div>
            <div><dt>Top-5 feature agreement across runs</dt><dd>{n(data.explanation_summary.mean_feature_top5_jaccard, 2)}</dd></div>
            <div><dt>Mean logit drop without any edges</dt><dd>{n(data.explanation_summary.mean_logit_drop_without_any_edges, 2)}</dd></div>
          </dl>
          <p className="note">Low rank agreement means GNNExplainer importance should not be read as a measured effect, so the Investigate page ranks edges by measured logit drop.</p>
        </Section>
      )}
      <Section title="All variants (ablations)" note="Every model, feature set and edge mode, so you can compare variants of one algorithm. These are auxiliary to the frozen comparison above.">
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Model</th><th>Features</th><th>Edges</th><th className="num">Validation PR-AUC</th><th className="num">Test accuracy</th><th className="num">Test PR-AUC</th><th className="num">Test F1</th></tr>
            </thead>
            <tbody>
              {data.ablations.map((row) => (
                <tr key={`${row.model}-${row.feature_set}-${row.edge_mode}`}>
                  <td>{MODEL_NAMES[row.model]}</td><td>{row.feature_set}</td><td>{row.edge_mode}</td>
                  <td className="num">{n(row.val_pr_auc)}</td>
                  <td className="num">{n(row.accuracy_mean)} ± {n(row.accuracy_std)}</td>
                  <td className="num">{n(row.pr_auc_mean)} ± {n(row.pr_auc_std)}</td>
                  <td className="num">{n(row.f1_mean)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </>
  );
}
