import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd


# -----------------------------
# File paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_PATH = BASE_DIR / "gnn_test_predictions.csv"
METADATA_PATH = BASE_DIR / "best_model.json"
RESULTS_PATH = BASE_DIR / "final_results.csv"

PLOTS_DIR = BASE_DIR / "plots"

PLOT_FILES = {
    "ROC and Precision-Recall curves": "results_roc_pr_curves.png",
    "Confusion matrices": "results_confusion_matrices.png",
    "Threshold sensitivity": "results_threshold_sensitivity.png",
    "Cumulative gain / alert budget": "results_cumulative_gain.png",
    "Illicit recall by wallet degree": "results_recall_by_degree.png",
    "Training curves": "results_training_curves.png",
    "XGBoost feature importance": "results_xgb_feature_importance.png",
}


# -----------------------------
# Required-file validation
# -----------------------------

required_files = [PREDICTIONS_PATH, METADATA_PATH, RESULTS_PATH]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path.name}. "
            "Place it in the same folder as app.py."
        )


# -----------------------------
# Load model metadata
# -----------------------------

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


# -----------------------------
# Load predictions and results
# -----------------------------

predictions = pd.read_csv(
    PREDICTIONS_PATH,
    dtype={"address": str},
)

results = pd.read_csv(RESULTS_PATH)

required_columns = {"address", score_column}
missing_columns = required_columns - set(predictions.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns in prediction CSV: {sorted(missing_columns)}. "
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
    errors="raise",
)

if not predictions[score_column].between(0, 1).all():
    raise ValueError("Saved scores must be between 0 and 1.")

wallet_lookup = predictions.set_index("address")


# -----------------------------
# Wallet lookup function
# -----------------------------

def lookup_wallet(address):
    address = str(address or "").strip()

    if not address:
        return (
            "Please enter a wallet address.",
            "",
            "",
        )

    if address not in wallet_lookup.index:
        return (
            "Wallet not found",
            "",
            "Only wallets in the saved test dataset are available. "
            "Try one of the examples below.",
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


# -----------------------------
# Examples
# -----------------------------

examples = [
    [address]
    for address in predictions["address"].head(5)
]

selection_method = str(
    model_info.get("selected_by", "Not recorded")
)


# -----------------------------
# Gradio interface
# -----------------------------

with gr.Blocks(
    title="FraudRing-GNN",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # FraudRing-GNN

        ### Wallet Risk — Saved Results Explorer

        Explore previously computed model scores for Elliptic++ test wallets.
        This demo does not perform live blockchain analysis or generate
        predictions for new wallets.
        """
    )

    gr.Markdown(
        f"""
        **Best model:** {model_name}  
        **Available wallets:** {len(predictions):,}  
        **Decision threshold:** {threshold:.6f}  
        **Model-selection method:** {selection_method}
        """
    )

    with gr.Tabs():

        # ---------------------
        # Wallet lookup
        # ---------------------

        with gr.Tab("Wallet lookup"):
            wallet_input = gr.Textbox(
                label="Bitcoin wallet address",
                placeholder="Enter a wallet address or select an example",
            )

            search_button = gr.Button("Show saved result", variant="primary")

            decision_output = gr.Textbox(
                label="Model decision",
                interactive=False,
            )

            score_output = gr.Textbox(
                label="Saved illicit-class score",
                interactive=False,
            )

            details_output = gr.Textbox(
                label="Details",
                lines=5,
                interactive=False,
            )

            search_button.click(
                fn=lookup_wallet,
                inputs=wallet_input,
                outputs=[
                    decision_output,
                    score_output,
                    details_output,
                ],
            )

            wallet_input.submit(
                fn=lookup_wallet,
                inputs=wallet_input,
                outputs=[
                    decision_output,
                    score_output,
                    details_output,
                ],
            )

            gr.Examples(
                examples=examples,
                inputs=wallet_input,
                label="Example wallets",
            )

        # ---------------------
        # Model results
        # ---------------------

        with gr.Tab("Model results"):
            gr.Markdown(
                "### Offline experimental results\n"
                "Comparison of XGBoost, GCN, and GraphSAGE on the held-out test set."
            )

            gr.Dataframe(
                value=results,
                interactive=False,
                label="Final results",
            )

            gr.Markdown(
                "These results describe the offline experiment, "
                "not guaranteed live performance."
            )

        # ---------------------
        # Plots
        # ---------------------

        with gr.Tab("Plots"):
            gr.Markdown(
                "### Evaluation plots\n"
                "These plots were generated during the offline experiment."
            )

            available_plots = []

            for title, filename in PLOT_FILES.items():
                plot_path = PLOTS_DIR / filename

                if plot_path.exists():
                    available_plots.append((title, str(plot_path)))

            if not available_plots:
                gr.Markdown(
                    "No plot images were found. Add the generated PNG files "
                    "to a folder named `plots` in the repository root."
                )
            else:
                for title, plot_path in available_plots:
                    gr.Markdown(f"#### {title}")
                    gr.Image(
                        value=plot_path,
                        show_label=False,
                        interactive=False,
                    )

    gr.Markdown(
        """
        ---

        **Research demonstration only.** A flagged wallet is not a verified
        fraudster, and this app does not establish fraud-ring membership.
        """
    )


# -----------------------------
# Run app
# -----------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
        share=False,
    )