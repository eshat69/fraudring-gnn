
import os
import json
from pathlib import Path

import gradio as gr
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_FILE = BASE_DIR / "gnn_test_predictions.csv"
MODEL_INFO_FILE = BASE_DIR / "best_model.json"
RESULTS_FILE = BASE_DIR / "final_results.csv"


# ---------- Load data ----------
if not PREDICTIONS_FILE.exists():
    raise FileNotFoundError(f"Missing file: {PREDICTIONS_FILE}")

if not MODEL_INFO_FILE.exists():
    raise FileNotFoundError(f"Missing file: {MODEL_INFO_FILE}")

df = pd.read_csv(PREDICTIONS_FILE)
results = (
    pd.read_csv(RESULTS_FILE)
    if RESULTS_FILE.exists()
    else pd.DataFrame()
)

with open(MODEL_INFO_FILE, "r", encoding="utf-8") as f:
    model_info = json.load(f)

best_model = model_info.get("best_model", "GraphSAGE")

score_columns = {
    "GraphSAGE": "prob_graphsage",
    "GCN": "prob_gcn",
    "XGBoost": "prob_xgboost",
}

score_column = score_columns.get(best_model)

if score_column not in df.columns:
    raise ValueError(
        f"Score column '{score_column}' not found in predictions CSV."
    )

threshold = float(model_info.get("threshold", 0.5))

address_column = next(
    (
        col for col in ["address", "wallet_address", "wallet"]
        if col in df.columns
    ),
    None,
)

if address_column is None:
    raise ValueError(
        "CSV must contain address, wallet_address, or wallet column."
    )

df[address_column] = df[address_column].astype(str).str.strip()
df[score_column] = pd.to_numeric(df[score_column], errors="coerce")

df = df.dropna(subset=[score_column])
df = df.drop_duplicates(subset=[address_column], keep="first")


# ---------- Wallet lookup ----------
def lookup_wallet(address):
    address = str(address).strip()

    if not address:
        return "Enter a wallet address.", None

    match = df[df[address_column] == address]

    if match.empty:
        return "Wallet address not found.", None

    row = match.iloc[0]
    score = float(row[score_column])
    decision = "Flagged" if score >= threshold else "Not Flagged"

    details = pd.DataFrame({
        "Field": ["Wallet Address", "Model", "Risk Score",
                  "Threshold", "Decision"],
        "Value": [
            address,
            best_model,
            f"{score:.4f}",
            f"{threshold:.4f}",
            decision,
        ],
    })

    return decision, details


# ---------- Dashboard ----------
custom_css = """
.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
}
footer { display: none !important; }
"""

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="cyan",
        secondary_hue="blue",
        neutral_hue="slate",
    ),
    css=custom_css,
    title="FraudRing-GNN Dashboard",
) as demo:

    gr.Markdown(
        """
        # FraudRing-GNN
        ### Illicit Wallet Detection Dashboard
        Review saved model predictions and check wallet risk scores.
        """
    )

    with gr.Row():
        gr.Number(
            label="Wallets Analyzed",
            value=len(df),
            interactive=False,
        )
        gr.Number(
            label="Flagged Wallets",
            value=int((df[score_column] >= threshold).sum()),
            interactive=False,
        )
        gr.Textbox(
            label="Selected Model",
            value=best_model,
            interactive=False,
        )

    with gr.Tab("Wallet Lookup"):
        wallet_input = gr.Textbox(
            label="Wallet Address",
            placeholder="Paste wallet address here...",
        )

        check_button = gr.Button(
            "Check Wallet",
            variant="primary",
        )

        decision_output = gr.Textbox(label="Detection Result")
        details_output = gr.Dataframe(
            label="Wallet Details",
            interactive=False,
        )

        check_button.click(
            fn=lookup_wallet,
            inputs=wallet_input,
            outputs=[decision_output, details_output],
        )

        wallet_input.submit(
            fn=lookup_wallet,
            inputs=wallet_input,
            outputs=[decision_output, details_output],
        )

    with gr.Tab("Model Results"):
        gr.Markdown("### Saved Model Evaluation Results")

        if not results.empty:
            gr.Dataframe(
                value=results,
                interactive=False,
                wrap=True,
            )
        else:
            gr.Markdown(
                "No `final_results.csv` file found."
            )

    gr.Markdown(
        """
        ---
        *FraudRing-GNN · Research Dashboard · Results from saved predictions*
        """
    )


# ---------- Render startup ----------
if __name__ == "__main__":
    print("Starting FraudRing-GNN...", flush=True)

    port = int(os.environ.get("PORT", "10000"))

    print(f"Launching Gradio on port {port}...", flush=True)

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
