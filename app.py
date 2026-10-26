
import os
import json
from pathlib import Path

import gradio as gr
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "gnn_test_predictions.csv")

with open(BASE_DIR / "best_model.json", encoding="utf-8") as f:
    info = json.load(f)

model = info.get("best_model", "GraphSAGE")
threshold = float(info.get("threshold", 0.5))

columns = {
    "GraphSAGE": "prob_graphsage",
    "GCN": "prob_gcn",
    "XGBoost": "prob_xgboost",
}

score_col = columns[model]
address_col = next(
    c for c in ["address", "wallet_address", "wallet"] if c in df.columns
)

df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
df = df.dropna(subset=[score_col])

df["Risk Score"] = df[score_col]
df["Suggestion"] = df["Risk Score"].apply(
    lambda x: "Flagged" if x >= threshold else "Not Flagged"
)

suggestions = df[
    [address_col, "Risk Score", "Suggestion"]
].rename(columns={address_col: "Wallet Address"})

results_path = BASE_DIR / "final_results.csv"
results = (
    pd.read_csv(results_path)
    if results_path.exists()
    else pd.DataFrame({"Message": ["No model results available."]})
)

with gr.Blocks(title="FraudRing-GNN") as demo:
    gr.Markdown("# FraudRing-GNN")
    gr.Markdown("### Wallet Suggestions")

    gr.Dataframe(
        value=suggestions,
        interactive=False,
        label="Wallet Risk Predictions",
    )

    gr.Markdown("### Model Results")

    gr.Dataframe(
        value=results,
        interactive=False,
        label="Evaluation Results",
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        share=False,
    )
