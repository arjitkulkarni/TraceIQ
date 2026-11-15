# TraceIQ design note (Stage A)

This note freezes the definitions used by every experiment. `config/experiments.yaml` holds the same values in machine-readable form.

## Task

Binary node classification on the Elliptic Bitcoin transaction graph. For each transaction, estimate whether it is illicit, using its features, its directed neighbors, and in the temporal models, the order of time steps.

- **Node:** one Bitcoin transaction. Not an account, wallet or person.
- **Edge:** directed fund flow from a source transaction to a destination transaction.
- **Snapshot:** all transactions in one of 49 time steps. No edge crosses steps.
- **Positive class:** illicit (`class = 1`, encoded 1).
- **Negative class:** licit (`class = 2`, encoded 0).
- **Unknown:** `class = unknown`, encoded -1. Excluded from loss and metrics. Never treated as licit.

## Evaluation protocol

| Period | Steps | Illicit | Licit | Illicit share of known |
|---|---|---|---|---|
| Train | 1-30 | 2,954 | 23,951 | 11.0% |
| Validation | 31-39 | 955 | 7,520 | 11.3% |
| Test | 40-49 | 636 | 10,548 | 5.7% |

The split was declared from label counts before any model existed, and it is the same for every model.

- **Primary metric:** PR-AUC, computed as scikit-learn average precision on known test labels.
- **Secondary metrics:** illicit-class precision, recall and F1 at the frozen threshold, Recall@K, alert count.
- **Threshold:** the cutoff that maximizes validation F1, chosen per run.
- **Model selection:** highest mean validation PR-AUC.
- **Test use:** test metrics are computed in one place only, `src/eval/compare.py`, after selections are written to `artifacts/reports/frozen_selection.json`.
- **Repeats:** five seeds for stochastic models. Logistic Regression uses a deterministic solver, so it runs once.

## Input features

Time step is never an input, because test steps lie outside the training range. Two feature sets are compared:

- **local:** 93 per-transaction columns.
- **all:** local plus aggregated one-hop neighborhood columns, minus time proxies (160 columns).

The aggregated columns already contain neighbor information. So a "tabular" model on `all` is not free of graph context, and only the `local` comparison isolates what edges add.

## Decisions made from data, before training

1. **Time-proxy removal.** Five columns (f099, f100, f102, f135, f138) carry the time step almost directly. Each has more than half of its training-period variance between steps (eta squared 0.63 to 0.98). They are excluded by that rule, which is measured on training rows without labels. One XGBoost ablation keeps them, to measure the effect.
2. **Scaling.** The publishers standardized features across all 49 steps before release, which is upstream leakage we cannot remove. We disclose it, and we still fit our own scaler on labeled training rows only, followed by a fixed clip at ±10.

## Model set

| Role | Model | Tests |
|---|---|---|
| Tabular | Logistic Regression, Random Forest, XGBoost | Baseline strength |
| No edges | MLP | Neural baseline, ablation for edges |
| Static graph | GraphSAGE, Static GCN | RQ1 |
| Temporal | EvolveGCN-O, EvolveGCN-H | RQ2 |

Static GCN is EvolveGCN with weight evolution removed, so comparing the two isolates temporal state.

## Implementation differences from the EvolveGCN paper

The EvolveGCN models are our own implementation of Pareja et al. (2020), not the reference code.

- A linear input projection maps features to the hidden size before the graph layers.
- Leaky ReLU replaces ReLU, because with plain ReLU the weight columns converged and units died during rollouts, which produced a constant output.
- Backpropagation through time is truncated every 5 snapshots to control memory.

## Boundaries

The system classifies historical Bitcoin transactions for research. It is not credit-card or bank fraud detection, it is not real-time, and it does not follow any customer across time. It supports analyst prioritization and must not be used to accuse anyone.
