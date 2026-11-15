# TraceIQ EDA summary

Generated figures are in `artifacts/figures/` and the numbers are in `artifacts/reports/data_quality.json` and `eda.json`. Regenerate them with `python -m src.data.quality` and `python -m src.eval.eda`.

## Data quality

The source files are clean, and nothing needed repair.

| Check | Result |
|---|---|
| Rows, columns | 203,769 transactions, 167 columns (ID, time step, 165 features) |
| Duplicate transaction IDs | 0 |
| Missing or non-finite feature values | 0 |
| Nodes without a class, or classes without a node | 0 |
| Edges | 234,355 |
| Edge endpoints missing from the node table | 0 |
| Self-loops, duplicate edges, reciprocal edges | 0 |
| Edges crossing a time step | 0 |

## What does one record represent?

One Bitcoin transaction in one of 49 time steps, with 165 anonymized numeric features. Edges link transactions by fund flow.

## What percentage of nodes have ground-truth labels?

22.9%. That is 4,545 illicit and 42,019 licit, leaving 157,205 unknown.

## How do positive labels vary over time?

Illicit transactions are bursty, and prevalence drops in the test period.

| Period | Illicit share of known labels |
|---|---|
| Train (1-30) | 11.0% |
| Validation (31-39) | 11.3% |
| Test (40-49) | 5.7% |
| Test, steps 40-42 | 10.4% |
| Test, steps 43-49 | 2.5% |

From step 43 the known illicit count collapses. Steps 43 to 49 together contain 169 illicit transactions, and step 46 has only 2. Weber et al. (2019) attribute this period to a dark market shutdown. Late-test results therefore rest on few positives and should be read with care.

## How sparse is the graph?

Very sparse. Mean in-degree and out-degree are both 1.15 and the medians are 1. Degree is heavy-tailed, with a maximum of 284 in and 472 out. There are no isolated nodes, and each time step forms exactly one weakly connected component.

## Does it fit on 8 GB of VRAM?

Yes. The largest step has 7,880 nodes. Training per snapshot peaks at about 220 MB of GPU memory, so full-batch training within a snapshot is safe.

## Feature drift

Feature distributions shift between train and test. The median Kolmogorov-Smirnov statistic across columns is 0.08, and 65 columns exceed 0.10.

The largest drift values (near 1.0) are not behavior. Five aggregated columns rise almost linearly with the time step and act as hidden time-step inputs (`eda_feature_drift.png`). They were removed before training. See `docs/design_note.md`.

## Limitations that may bias conclusions

- Labels are incomplete, and unknown transactions are not a random sample.
- The publishers standardized features across all time steps, including future ones.
- Features are anonymized, so individual columns cannot be interpreted.
- 72 columns are neighbor aggregates, so "tabular" models with all features already use graph context.
- There are no cross-step edges, so no multi-period money flow can be observed.
- The sharp late-test change makes results sensitive to where the test period falls.
