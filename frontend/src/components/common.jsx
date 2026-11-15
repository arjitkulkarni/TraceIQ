const TIER_MARK = { high: "▲", medium: "◆", low: "●" };
const LABEL_MARK = { illicit: "▲", licit: "■", unknown: "●" };

export function TierBadge({ tier }) {
  return (
    <span className={`badge tier-${tier}`}>
      <span aria-hidden="true">{TIER_MARK[tier]}</span> {tier}
    </span>
  );
}

export function LabelBadge({ label }) {
  return (
    <span className={`badge label-${label}`}>
      <span aria-hidden="true">{LABEL_MARK[label]}</span> {label}
    </span>
  );
}

export function Loading({ what = "data" }) {
  return <p className="state">Loading {what}…</p>;
}

export function ErrorState({ error }) {
  return <div className="state error">{error?.message ?? "Something went wrong."}</div>;
}

export function Stat({ label, value, detail }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {detail && <div className="stat-detail">{detail}</div>}
    </div>
  );
}

export function Section({ title, note, children }) {
  return (
    <section className="panel">
      <h2>{title}</h2>
      {note && <p className="note">{note}</p>}
      {children}
    </section>
  );
}
