
import os
import json
from pathlib import Path

import gradio as gr
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Load prediction data
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

score_col = columns.get(model, "prob_graphsage")

address_col = next(
    (c for c in ["address", "wallet_address", "wallet"]
     if c in df.columns),
    None,
)

if address_col is None:
    raise ValueError("No wallet address column found in CSV.")

if score_col not in df.columns:
    raise ValueError(f"Missing prediction column: {score_col}")

df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
df = df.dropna(subset=[score_col]).copy()

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


def check_wallet(wallet):
    wallet = (wallet or "").strip()

    if not wallet:
        return "Please enter a wallet address.", None

    match = df[
        df[address_col].astype(str).str.strip().str.lower()
        == wallet.lower()
    ]

    if match.empty:
        return (
            "Wallet not found in the available prediction dataset. "
            "This app currently checks existing predictions only.",
            None,
        )

    row = match.iloc[0]
    score = float(row["Risk Score"])
    prediction = row["Suggestion"]

    output = pd.DataFrame([{
        "Wallet Address": str(row[address_col]),
        "Risk Score": round(score, 4),
        "Risk Percentage": f"{score * 100:.2f}%",
        "Prediction": prediction,
        "Threshold": threshold,
        "Model": model,
    }])

    message = (
        f"### {prediction}\n"
        f"**Risk score:** {score:.4f} ({score * 100:.2f}%)\n\n"
        f"**Model:** {model}\n\n"
        + (
            "Suggestion: Review this wallet for potential suspicious activity."
            if score >= threshold
            else "Suggestion: No flag triggered by the current model threshold."
        )
    )

    return message, output


with gr.Blocks(title="FraudRing-GNN") as demo:
    gr.Markdown("# FraudRing-GNN")
    gr.Markdown("Wallet fraud-risk prediction and review")

    with gr.Group():
        gr.Markdown("### Check a Wallet")
        wallet_input = gr.Textbox(
            label="Wallet Address",
            placeholder="Enter a wallet address from the prediction dataset",
        )
        check_button = gr.Button("Check Wallet", variant="primary")
        prediction_output = gr.Markdown()
        wallet_result = gr.Dataframe(
            label="Prediction Result",
            interactive=False,
        )

        check_button.click(
            fn=check_wallet,
            inputs=wallet_input,
            outputs=[prediction_output, wallet_result],
        )
        wallet_input.submit(
            fn=check_wallet,
            inputs=wallet_input,
            outputs=[prediction_output, wallet_result],
        )

    gr.Markdown("### Wallet Suggestions")
    gr.Dataframe(
        value=suggestions,
        interactive=False,
        label="All Wallet Predictions",
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
