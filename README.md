https://fraudring-gnn.onrender.com/

# FraudRing GNN: Illicit Wallet Detection


This project compares traditional machine-learning and graph neural network (GNN) models for detecting illicit cryptocurrency wallets using wallet graph features.

The best model, **GraphSAGE**, achieved **ROC-AUC 0.974** and **PR-AUC 0.794** on the held-out test set, outperforming both GCN and an XGBoost baseline. [116]

## Overview

The project builds wallet-level graph features and evaluates three approaches:

- **XGBoost** — tabular graph-feature baseline
- **GCN** — Graph Convolutional Network
- **GraphSAGE** — GraphSAGE neural network

The dataset is highly imbalanced: the final test set contains **39,804 labeled wallets**, of which **2,140 (5.38%)** are illicit. [116]

## Key Results

| Model | ROC-AUC | PR-AUC | Illicit Precision | Illicit Recall | Illicit F1 |
|---|---:|---:|---:|---:|---:|
| **GraphSAGE** | **0.974** | **0.794** | **0.764** | **0.693** | **0.726** |
| GCN | 0.921 | 0.471 | 0.538 | 0.432 | 0.479 |
| XGBoost | 0.878 | 0.460 | 0.674 | 0.512 | 0.582 |

Metrics are reported on the test set. ROC-AUC, PR-AUC, and illicit F1 include bootstrap 95% confidence intervals; GraphSAGE was selected using validation PR-AUC, not test performance. [116]

## Why GraphSAGE Helps

GraphSAGE substantially improves illicit-wallet detection over the XGBoost baseline:

- PR-AUC improves by **0.334** (95% CI: 0.316–0.355).
- GraphSAGE performs better than XGBoost in **100%** of 500 paired bootstrap resamples.
- At the top 1% highest-risk wallets, GraphSAGE identifies **18.5%** of illicit wallets at **99.5% precision**, versus **11.1%** recall at **59.5% precision** for XGBoost. [116]

This suggests that neighborhood information captured through message passing is especially useful for identifying suspicious wallets.

## Project Structure

```text
.
├── 03_train_models.ipynb        # Baseline models: Logistic Regression, XGBoost, Random Forest
├── 04gnn.ipynb                  # Graph construction, GCN and GraphSAGE training
├── 05_results.ipynb             # Final evaluation, bootstrap tests, plots, exports
├── outputs/
│   ├── results/
│   │   ├── final_results.csv
│   │   ├── final_summary.md
│   │   ├── best_model.json
│   │   └── gnn_test_predictions.csv
│   ├── models/
│   └── plots/
└── README.md
```

## Features Used

The models use wallet-level graph features:

- In-degree
- Out-degree
- Total degree
- Degree centrality
- Number of predecessors
- Number of successors
- Number of neighbors

The labeled dataset contains **265,354 wallets** with seven graph features. [114]

## Methodology

1. Load wallet graph features and convert class labels into binary illicit/licit labels.
2. Perform a stratified train/validation/test split.
3. Train baseline models using tabular graph features.
4. Train GCN and GraphSAGE models on the wallet graph.
5. Select decision thresholds on the validation set using illicit F1.
6. Evaluate all models on the untouched test set.
7. Use 500 paired bootstrap resamples to quantify uncertainty and compare models.
8. Export final metrics, predictions, and dashboard-ready artifacts. [114][116]

## Results Artifacts

Important generated files include:

- `final_results.csv` — final model metrics and confidence intervals
- `final_paired_differences.csv` — bootstrap model comparisons
- `final_topk_alert_budget.csv` — analyst alert-budget analysis
- `final_degree_breakdown.csv` — performance by wallet degree
- `best_model.json` — selected model, threshold, and test metrics
- `gnn_test_predictions.csv` — test-set predictions for dashboard use [116]

## Requirements

```bash
pip install -r requirements.txt
```

Main dependencies:

```text
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
torch
torch-geometric
optuna
joblib
```

## How to Run

Run the notebooks in order:

```bash
jupyter notebook 03_train_models.ipynb
jupyter notebook 04gnn.ipynb
jupyter notebook 05_results.ipynb
```

`05_results.ipynb` requires outputs from `04gnn.ipynb`, including `gnn_model_comparison.csv` and `gnn_test_predictions.csv`. [116]

## Notes

- Model selection is based on validation PR-AUC to avoid test-set leakage.
- Thresholds are selected on validation data and applied unchanged to test data.
- The dataset is imbalanced, so PR-AUC, illicit precision, illicit recall, and F1 are more informative than accuracy alone. [116]

## License

This project is released under the MIT License.
