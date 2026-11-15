import { MODEL_NAMES, fmt, useFetch } from "../api.js";
import { StepBars } from "../components/charts.jsx";
import { ErrorState, Loading, Section, Stat } from "../components/common.jsx";

export default function Overview() {
  const overview = useFetch("/overview");
  const metrics = useFetch("/metrics");
  if (overview.loading || metrics.loading) return <Loading what="overview" />;
  if (overview.error || metrics.error) return <ErrorState error={overview.error ?? metrics.error} />;
  const { data } = overview;
  const known = (data.by_label.illicit ?? 0) + (data.by_label.licit ?? 0);
  const flagged = data.test_period_by_tier.high ?? 0;
  const split = metrics.data.split.split;
  return (
    <>
      <h1>Overview</h1>
      <p className="lede">
        Every transaction in the Elliptic dataset, scored offline by {MODEL_NAMES[data.serving.model_name]}.
        Flags use the threshold frozen on validation data.
      </p>
      <div className="stats">
        <Stat label="Transactions scored" value={fmt.count(data.transactions_scored)} detail="49 time steps" />
        <Stat label="Flagged in test period" value={fmt.count(flagged)}
          detail={`${fmt.percent(flagged / data.test_period_transactions)} of ${fmt.count(data.test_period_transactions)}`} />
        <Stat label="Known labels" value={fmt.count(known)}
          detail={`${fmt.count(data.by_label.unknown)} transactions have no label`} />
        <Stat label="Alert threshold" value={fmt.score(data.serving.threshold)} detail="Uncalibrated model score" />
      </div>
      <Section title="Transactions flagged per time step" note="High-tier model scores, including steps used to train the model.">
        <StepBars steps={data.steps} field="flagged" label="Flagged" split={split} />
      </Section>
      <Section title="Known illicit transactions per time step" note="Dataset labels, shown for context. These are not model output.">
        <StepBars steps={data.steps} field="known_illicit" label="Known illicit" split={split} />
      </Section>
      <p className="actions"><a className="button" href="#/queue">Open the risk queue</a></p>
    </>
  );
}
