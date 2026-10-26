
import os
import json
import logging
from pathlib import Path

import gradio as gr
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "10000"))

columns = {
    "GraphSAGE": "prob_graphsage",
    "GCN": "prob_gcn",
    "XGBoost": "prob_xgboost",
}

# Load model configuration
try:
    with open(BASE_DIR / "best_model.json", encoding="utf-8") as f:
        info = json.load(f)

    model = info.get("best_model", "GraphSAGE")
    threshold = float(info.get("threshold", 0.5))

    if model not in columns:
        raise ValueError(f"Unsupported model: {model}")

    score_col = columns[model]

    # Load prediction data
    df = pd.read_csv(BASE_DIR / "gnn_test_predictions.csv")

    address_col = next(
        (
            c for c in ["address", "wallet_address", "wallet"]
            if c in df.columns
        ),
        None,
    )

    if address_col is None:
        raise ValueError("Wallet address column is missing.")

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

    data_error = None
    logger.info("Loaded %s wallet predictions using %s", len(df), model)

except Exception as exc:
    logger.exception("Failed to load prediction data")
    df = pd.DataFrame()
    suggestions = pd.DataFrame(
        columns=["Wallet Address", "Risk Score", "Suggestion"]
    )
    data_error = str(exc)
    model = "Unavailable"
    threshold = 0.5
    address_col = None

# Load evaluation results
results_path = BASE_DIR / "final_results.csv"

try:
    results = (
        pd.read_csv(results_path)
        if results_path.exists()
        else pd.DataFrame({"Message": ["No model results available."]})
    )
except Exception:
    logger.exception("Failed to load evaluation results")
    results = pd.DataFrame(
        {"Message": ["Could not load model evaluation results."]}
    )


def check_wallet(wallet):
    if data_error:
        return f"Prediction data unavailable: {data_error}", None

    wallet = (wallet or "").strip()

    if not wallet:
        return "Please enter a wallet address.", None

    match = df[
        df[address_col].astype(str).str.strip().str.lower()
        == wallet.lower()
    ]

    if match.empty:
        return (
            "Wallet not found in the existing prediction dataset. "
            "No new model inference was performed.",
            None,
        )

    row = match.iloc[0]
    score = float(row["Risk Score"])
    status = row["Suggestion"]

    result = pd.DataFrame([{
        "Wallet Address": str(row[address_col]),
        "Risk Score": round(score, 4),
        "Risk Percentage": f"{score * 100:.2f}%",
        "Prediction": status,
        "Threshold": threshold,
        "Model": model,
    }])

    message = (
        f"### Prediction: {status}\n\n"
        f"**Risk score:** {score:.4f} ({score * 100:.2f}%)\n\n"
        f"**Model:** {model}\n\n"
        + (
            "Suggestion: Review this wallet for potentially suspicious activity."
            if score >= threshold
            else "Suggestion: No flag triggered by the configured threshold."
        )
    )

    return message, result


with gr.Blocks(title="FraudRing-GNN") as demo:
    gr.Markdown("# FraudRing-GNN")
    gr.Markdown("Wallet fraud-risk prediction and review")

    if data_error:
        gr.Markdown(
            "⚠️ Prediction data could not be loaded. "
            "Check the Render application logs."
        )

    with gr.Group():
        gr.Markdown("### Check a Wallet")

        wallet_input = gr.Textbox(
            label="Wallet Address",
            placeholder="Enter a wallet address",
        )

        check_button = gr.Button("Check Wallet", variant="primary")
        prediction_output = gr.Markdown()

        wallet_result = gr.Dataframe(
            label="Prediction Result",
            interactive=False,
        )

        check_button.click(
            check_wallet,
            inputs=wallet_input,
            outputs=[prediction_output, wallet_result],
        )

        wallet_input.submit(
            check_wallet,
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
    logger.info("Starting FraudRing-GNN on port %s", PORT)

    demo.launch(
        server_name="0.0.0.0",
        server_port=PORT,
        share=False,
        show_error=True,
    )
