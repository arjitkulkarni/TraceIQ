| model | feature_set | edge_mode | accuracy_mean | precision_mean | recall_mean | f1_mean | pr_auc_mean | pr_auc_std | recall_at_100_mean | alerts_mean |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost | all | none | 0.971 | 0.855 | 0.592 | 0.699 | 0.675 | 0.003 | 0.157 | 440.2 |
| Random Forest | all | none | 0.973 | 0.922 | 0.576 | 0.709 | 0.661 | 0.001 | 0.157 | 397.2 |
| GraphSAGE | local | symmetrized | 0.966 | 0.801 | 0.548 | 0.649 | 0.627 | 0.007 | 0.153 | 438.6 |
| MLP (no edges) | local | none | 0.965 | 0.779 | 0.543 | 0.636 | 0.566 | 0.011 | 0.139 | 450.6 |
| Static GCN | local | symmetrized | 0.936 | 0.456 | 0.547 | 0.496 | 0.511 | 0.054 | 0.142 | 771.6 |
| EvolveGCN-O | local | symmetrized | 0.933 | 0.461 | 0.538 | 0.486 | 0.465 | 0.072 | 0.13 | 794.8 |
| EvolveGCN-H | all | symmetrized | 0.933 | 0.449 | 0.458 | 0.446 | 0.459 | 0.068 | 0.139 | 693.6 |
| Logistic Regression | all | none | 0.843 | 0.203 | 0.604 | 0.304 | 0.266 | 0.0 | 0.058 | 1892.0 |
