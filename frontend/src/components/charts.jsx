import { useState } from "react";

const W = 760;

function Tooltip({ tip }) {
  if (!tip) return null;
  return (
    <div className="tooltip" style={{ left: `${(tip.x / W) * 100}%`, top: tip.y }}>
      {tip.lines.map((line) => <div key={line}>{line}</div>)}
    </div>
  );
}

/** Bars by time step, with hover. Bands mark the chronological split. */
export function StepBars({ steps, field, label, split, height = 180 }) {
  const [tip, setTip] = useState(null);
  const left = 44;
  const bottom = 24;
  const top = 18;
  const width = (W - left - 8) / 49;
  const max = Math.max(...steps.map((step) => step[field]), 1);
  const y = (value) => top + (height - top - bottom) * (1 - value / max);
  const x = (step) => left + (step - 1) * width;
  const bands = [
    ["train", split.train_steps],
    ["validation", split.val_steps],
    ["test", split.test_steps],
  ];
  return (
    <div className="chart" onMouseLeave={() => setTip(null)}>
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`${label} per time step`}>
        {bands.map(([name, [low, high]], index) => (
          <g key={name}>
            <rect x={x(low)} y={top} width={(high - low + 1) * width} height={height - top - bottom}
              className={`band band-${index}`} />
            <text x={x(low) + ((high - low + 1) * width) / 2} y={12} textAnchor="middle" className="axis-label">{name}</text>
          </g>
        ))}
        {[0, 0.5, 1].map((share) => (
          <g key={share}>
            <line x1={left} x2={W - 8} y1={y(max * share)} y2={y(max * share)} className="grid" />
            <text x={left - 6} y={y(max * share) + 4} textAnchor="end" className="axis-label">{Math.round(max * share)}</text>
          </g>
        ))}
        {steps.map((step) => (
          <g key={step.time_step}
            onMouseEnter={() => setTip({ x: x(step.time_step), y: 8, lines: [`Step ${step.time_step}`, `${label}: ${step[field].toLocaleString("en-US")}`] })}>
            <rect x={x(step.time_step)} y={top} width={width} height={height - top - bottom} fill="transparent" />
            <rect x={x(step.time_step) + 1} y={y(step[field])} width={Math.max(width - 2, 1)}
              height={height - bottom - y(step[field])} rx="2" className="mark" />
          </g>
        ))}
        {[1, 10, 20, 30, 40, 49].map((step) => (
          <text key={step} x={x(step) + width / 2} y={height - 8} textAnchor="middle" className="axis-label">{step}</text>
        ))}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

/**
 * Dot plot of mean and spread, sorted best first. One row per model.
 * Pass domain and ticks to zoom in, and reference for a labeled comparison line.
 */
export function DotRange({ rows, height, label = "PR-AUC", domain, ticks, reference }) {
  const [tip, setTip] = useState(null);
  const left = 170;
  const rowHeight = 30;
  const top = reference ? 20 : 0;
  const total = height ?? rows.length * rowHeight + 34 + top;
  const low = domain?.[0] ?? 0;
  const max = domain?.[1] ?? Math.min(1, Math.max(...rows.map((row) => row.mean + row.std)) * 1.08);
  const x = (value) => left + (W - left - 60) * ((value - low) / (max - low));
  const axis = ticks ?? [0, 0.25, 0.5, 0.75, 1].filter((v) => v <= max);
  return (
    <div className="chart" onMouseLeave={() => setTip(null)}>
      <svg viewBox={`0 0 ${W} ${total}`} role="img" aria-label={`Test ${label} by model`}>
        {axis.map((value) => (
          <g key={value}>
            <line x1={x(value)} x2={x(value)} y1={4 + top} y2={total - 22} className="grid" />
            <text x={x(value)} y={total - 8} textAnchor="middle" className="axis-label">{value}</text>
          </g>
        ))}
        {reference && (
          <g>
            <line x1={x(reference.value)} x2={x(reference.value)} y1={14} y2={total - 22} className="reference" />
            <text x={x(reference.value)} y={10} textAnchor="middle" className="axis-label">{reference.label}</text>
          </g>
        )}
        {rows.map((row, index) => {
          const cy = 20 + top + index * rowHeight;
          return (
            <g key={row.name}
              onMouseEnter={() => setTip({ x: x(row.mean), y: cy + 10, lines: [row.name, `${label} ${row.mean.toFixed(3)} ± ${row.std.toFixed(3)}`] })}>
              <rect x={0} y={cy - rowHeight / 2} width={W} height={rowHeight} fill="transparent" />
              <text x={left - 12} y={cy + 4} textAnchor="end" className="row-label">{row.name}</text>
              <line x1={x(Math.max(row.mean - row.std, low))} x2={x(Math.min(row.mean + row.std, max))} y1={cy} y2={cy} className="range" />
              <circle cx={x(row.mean)} cy={cy} r="5" className="mark ring" />
              <text x={x(Math.min(row.mean + row.std, max)) + 8} y={cy + 4} className="axis-label">{row.mean.toFixed(3)}</text>
            </g>
          );
        })}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

/** Round axis ticks around the data. Rates stay within 0 to 1. */
function niceTicks(values, unit) {
  const rough = (Math.max(...values) - Math.min(...values) || 0.01) / 4;
  const power = 10 ** Math.floor(Math.log10(rough));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((s) => s >= rough);
  let low = Math.floor(Math.min(...values) / step) * step;
  let high = Math.ceil(Math.max(...values) / step) * step;
  if (unit) {
    low = Math.max(low, 0);
    high = Math.min(high, 1);
  }
  const count = Math.round((high - low) / step);
  return Array.from({ length: count + 1 }, (_, i) => Number((low + i * step).toFixed(6)));
}

/**
 * One line per epoch for each series. A series keeps its color when others are hidden,
 * so `id` picks the color, never its position in the list.
 */
export function EpochLines({ series, epochs, format, label, reference, unit = true, height = 280 }) {
  const [tip, setTip] = useState(null);
  const [hover, setHover] = useState(null);
  const left = 52;
  const right = 118;
  const top = 12;
  const bottom = 40;
  const values = series.flatMap((s) => epochs.map((epoch) => s.values[epoch]));
  const ticks = niceTicks(reference ? [...values, reference.value] : values, unit);
  const low = ticks[0];
  const high = ticks[ticks.length - 1];
  const x = (epoch) => left + (W - left - right) * ((epoch - epochs[0]) / (epochs[epochs.length - 1] - epochs[0]));
  const y = (value) => top + (height - top - bottom) * (1 - (value - low) / (high - low));
  const ends = placeLabels(series.map((s) => ({ id: s.id, name: s.name, y: y(s.values[epochs[epochs.length - 1]]) })));
  const showEpoch = (epoch) => {
    setHover(epoch);
    setTip({
      x: Math.min(x(epoch), W - 170),
      y: 36,
      lines: [`Epoch ${epoch}`, ...[...series].sort((a, b) => b.values[epoch] - a.values[epoch])
        .map((s) => `${s.name}: ${format(s.values[epoch])}`)],
    });
  };
  return (
    <div className="chart" onMouseLeave={() => { setTip(null); setHover(null); }}>
      <div className="legend">
        {series.map((s) => <span key={s.id}><i className={`swatch ${s.id}`} /> {s.name}</span>)}
      </div>
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`${label} per epoch`}>
        {ticks.map((value) => (
          <g key={value}>
            <line x1={left} x2={W - right} y1={y(value)} y2={y(value)} className="grid" />
            <text x={left - 6} y={y(value) + 4} textAnchor="end" className="axis-label">{value}</text>
          </g>
        ))}
        {reference && (
          <g>
            <line x1={left} x2={W - right} y1={y(reference.value)} y2={y(reference.value)} className="reference" />
            <text x={W - right - 6} y={y(reference.value) + 14} textAnchor="end" className="axis-label">{reference.label}</text>
          </g>
        )}
        {series.map((s) => (
          <polyline key={s.id} points={epochs.map((epoch) => `${x(epoch)},${y(s.values[epoch])}`).join(" ")}
            className={`line ${s.id}`} />
        ))}
        {ends.map((end) => (
          <text key={end.id} x={W - right + 8} y={end.labelY + 4} className="row-label end-label">{end.name}</text>
        ))}
        {hover && <line x1={x(hover)} x2={x(hover)} y1={top} y2={height - bottom} className="crosshair" />}
        {hover && series.map((s) => (
          <circle key={s.id} cx={x(hover)} cy={y(s.values[hover])} r="4" className={`dot ${s.id}`} />
        ))}
        {epochs.map((epoch) => (
          <g key={epoch} onMouseEnter={() => showEpoch(epoch)}>
            <rect x={x(epoch) - 16} y={0} width={32} height={height - 26} fill="transparent" />
            <text x={x(epoch)} y={height - 22} textAnchor="middle" className="axis-label">{epoch}</text>
          </g>
        ))}
        <text x={left + (W - left - right) / 2} y={height - 5} textAnchor="middle" className="axis-label">Epoch</text>
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}

/** Space end-of-line labels at least one line apart, keeping their order. */
function placeLabels(items, gap = 16) {
  const sorted = [...items].sort((a, b) => a.y - b.y);
  sorted.forEach((item, index) => {
    item.labelY = index === 0 ? item.y : Math.max(item.y, sorted[index - 1].labelY + gap);
  });
  return sorted;
}

/** Lines of mean recall per test step, one line per model, directly labeled and in a legend. */
export function StepLines({ series, xs, height = 230 }) {
  const [tip, setTip] = useState(null);
  const left = 40;
  const right = 16;
  const top = 10;
  const bottom = 26;
  const x = (value) => left + (W - left - right) * ((value - xs[0]) / (xs[xs.length - 1] - xs[0]));
  const y = (value) => top + (height - top - bottom) * (1 - value);
  const hover = (step) => setTip({
    x: x(step), y: 8,
    lines: [`Step ${step}`, ...series.map((s) => `${s.name}: ${s.values[step].toFixed(3)}`)],
  });
  return (
    <div className="chart" onMouseLeave={() => setTip(null)}>
      <div className="legend">
        {series.map((s, index) => (
          <span key={s.name}><i className={`swatch series-${index + 1}`} /> {s.name}</span>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label="Recall per test step">
        {[0, 0.5, 1].map((value) => (
          <g key={value}>
            <line x1={left} x2={W - right} y1={y(value)} y2={y(value)} className="grid" />
            <text x={left - 6} y={y(value) + 4} textAnchor="end" className="axis-label">{value}</text>
          </g>
        ))}
        {series.map((s, index) => (
          <g key={s.name}>
            <polyline points={xs.map((step) => `${x(step)},${y(s.values[step])}`).join(" ")}
              className={`line series-${index + 1}`} />
          </g>
        ))}
        {xs.map((step) => (
          <g key={step} onMouseEnter={() => hover(step)}>
            <rect x={x(step) - 16} y={0} width={32} height={height - bottom} fill="transparent" />
            <text x={x(step)} y={height - 8} textAnchor="middle" className="axis-label">{step}</text>
          </g>
        ))}
      </svg>
      <Tooltip tip={tip} />
    </div>
  );
}
