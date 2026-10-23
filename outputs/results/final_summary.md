# Wallet illicit-activity detection: final results

Test set: 39,804 labeled wallets, 2,140 illicit (5.38%). Model selected on validation PR-AUC: **GraphSAGE**.

| Model | PR-AUC [95% CI] | ROC-AUC [95% CI] | Illicit precision | Illicit recall | Illicit F1 [95% CI] |
|---|---|---|---|---|---|
| GraphSAGE | 0.794 [0.778, 0.809] | 0.974 [0.972, 0.977] | 0.764 | 0.693 | 0.726 [0.707, 0.739] |
| GCN | 0.471 [0.448, 0.494] | 0.921 [0.916, 0.926] | 0.538 | 0.432 | 0.479 [0.458, 0.497] |
| XGBoost | 0.460 [0.435, 0.482] | 0.878 [0.869, 0.886] | 0.674 | 0.512 | 0.582 [0.562, 0.599] |

## Paired differences (bootstrap, 95% CI)

- GCN - XGBoost PR-AUC: +0.012 [-0.011, +0.035]; GCN better in 85.0% of resamples
- GraphSAGE - XGBoost PR-AUC: +0.334 [+0.316, +0.355]; GraphSAGE better in 100.0% of resamples
- GraphSAGE - GCN PR-AUC: +0.322 [+0.300, +0.344]; GraphSAGE better in 100.0% of resamples

## Alert budget (review the top 1% highest-risk wallets)

- XGBoost: finds 11.1% of illicit wallets at 59.5% precision (11.1x random)
- GCN: finds 14.7% of illicit wallets at 78.9% precision (14.7x random)
- GraphSAGE: finds 18.5% of illicit wallets at 99.5% precision (18.5x random)

Thresholds were chosen on the validation set (max illicit F1) and applied unchanged to the test set.