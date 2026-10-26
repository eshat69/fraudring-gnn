import json
from pathlib import Path

import gradio as gr
import pandas as pd


# Files are resolved relative to app.py
BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_PATH = BASE_DIR / "gnn_test_predictions.csv"
METADATA_PATH = BASE_DIR / "best_model.json"
RESULTS_PATH = BASE_DIR / "final_results.csv"


# Check required files
for path in [PREDICTIONS_PATH, METADATA_PATH, RESULTS_PATH]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path.name}. "
            "Place it in the same folder as app.py."
        )


# Load model information
with METADATA_PATH.open("r", encoding="utf-8") as file:
    model_info = json.load(file)

MODEL_COLUMNS = {
    "GraphSAGE": "prob_graphsage",
    "GCN": "prob_gcn",
    "XGBoost": "prob_xgboost",
}

model_name = model_info["best_model"]

if model_name not in MODEL_COLUMNS:
    raise ValueError(f"Unsupported model: {model_name}")

score_column = MODEL_COLUMNS[model_name]
threshold = float(model_info["threshold"])

if not 0 <= threshold <= 1:
    raise ValueError("The threshold must be between 0 and 1.")


# Load saved predictions and results
predictions = pd.read_csv(
    PREDICTIONS_PATH,
    dtype={"address": str}
)

results = pd.read_csv(RESULTS_PATH)

required_columns = {"address", score_column}
missing = required_columns - set(predictions.columns)

if missing:
    raise ValueError(
        f"Missing columns in prediction CSV: {sorted(missing)}. "
        f"Available columns: {predictions.columns.tolist()}"
    )

if predictions.empty:
    raise ValueError("The prediction CSV is empty.")

if predictions["address"].isna().any():
    raise ValueError("The prediction CSV contains missing addresses.")

predictions["address"] = predictions["address"].str.strip()

if predictions["address"].eq("").any():
    raise ValueError("The prediction CSV contains empty addresses.")

if predictions["address"].duplicated().any():
    raise ValueError("Duplicate wallet addresses found in prediction CSV.")

predictions[score_column] = pd.to_numeric(
    predictions[score_column],
    errors="raise"
)

if not predictions[score_column].between(0, 1).all():
    raise ValueError("Saved scores must be between 0 and 1.")

wallet_lookup = predictions.set_index("address")


# Look up an existing saved result
def lookup_wallet(address):
    address = str(address or "").strip()

    if not address:
        return "Please enter a wallet address.", "", ""

    if address not in wallet_lookup.index:
        return (
            "Wallet not found",
            "",
            "Only wallets in the saved test dataset are available. "
            "Try one of the examples below."
        )

    score = float(wallet_lookup.loc[address, score_column])

    decision = (
        "Flagged for illicit-activity review"
        if score >= threshold
        else "Not flagged at this threshold"
    )

    details = (
        f"Model: {model_name}\n"
        f"Decision threshold: {threshold:.6f}\n"
        "Source: saved test-set prediction\n"
        "This score is not proof of illicit activity."
    )

    return decision, f"{score:.6f}", details


# Examples come from the actual CSV
examples = [
    [address]
    for address in predictions["address"].head(5)
]

selection_method = str(
    model_info.get("selected_by", "Not recorded")
)


# Web interface
with gr.Blocks() as demo:
    gr.Markdown(
        "# FraudRing-GNN\n"
        "### Wallet Risk — Saved Results Explorer\n"
        "Explore previously computed model scores for Elliptic++ "
        "test wallets. This demo does not perform live blockchain "
        "analysis or generate predictions for new wallets."
    )

    gr.Markdown(
        f"Model: {model_name}  \n"
        f"Available wallets: {len(predictions):,}  \n"
        f"Decision threshold: {threshold:.6f}"
    )

    with gr.Tab("Wallet lookup"):
        wallet_input = gr.Textbox(
            label="Bitcoin wallet address",
            placeholder="Enter a wallet address or select an example"
        )

        search_button = gr.Button("Show saved result")

        decision_output = gr.Textbox(
            label="Model decision",
            interactive=False
        )

        score_output = gr.Textbox(
            label="Saved illicit-class score",
            interactive=False
        )

        details_output = gr.Textbox(
            label="Details",
            lines=5,
            interactive=False
        )

        search_button.click(
            fn=lookup_wallet,
            inputs=wallet_input,
            outputs=[
                decision_output,
                score_output,
                details_output
            ]
        )

        wallet_input.submit(
            fn=lookup_wallet,
            inputs=wallet_input,
            outputs=[
                decision_output,
                score_output,
                details_output
            ]
        )

        gr.Examples(
            examples=examples,
            inputs=wallet_input
        )

    with gr.Tab("Model results"):
        gr.Dataframe(
            value=results,
            interactive=False,
            label="Offline experimental results"
        )

        gr.Markdown(
            f"Recorded model-selection method: {selection_method}\n\n"
            "These results describe the offline experiment, "
            "not guaranteed live performance."
        )

    gr.Markdown(
        "---\n"
        "Research demonstration only. A flagged wallet is not "
        "a verified fraudster, and this app does not establish "
        "fraud-ring membership."
    )


if __name__ == "__main__":
    demo.launch()
