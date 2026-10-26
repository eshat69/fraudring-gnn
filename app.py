
import html
import json
import os
from pathlib import Path

import gradio as gr
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_PATH = BASE_DIR / "gnn_test_predictions.csv"
METADATA_PATH = BASE_DIR / "best_model.json"
RESULTS_PATH = BASE_DIR / "final_results.csv"
PLOTS_DIR = BASE_DIR / "plots"

PLOT_FILES = {
    "ROC and Precision-Recall Curves": "results_roc_pr_curves.png",
    "Confusion Matrices": "results_confusion_matrices.png",
    "Threshold Sensitivity": "results_threshold_sensitivity.png",
    "Cumulative Gain / Alert Budget": "results_cumulative_gain.png",
    "Illicit Recall by Wallet Degree": "results_recall_by_degree.png",
    "Training Curves": "results_training_curves.png",
    "XGBoost Feature Importance": "results_xgb_feature_importance.png",
}

MODEL_COLUMNS = {
    "GraphSAGE": "prob_graphsage",
    "GCN": "prob_gcn",
    "XGBoost": "prob_xgboost",
}


# ============================================================
# 2. VALIDATE REQUIRED FILES
# ============================================================

required_files = [
    PREDICTIONS_PATH,
    METADATA_PATH,
    RESULTS_PATH,
]

for path in required_files:
    if not path.is_file():
        raise FileNotFoundError(
            f"Required file not found: {path.name}. "
            "Place it beside app.py in the repository."
        )


# ============================================================
# 3. LOAD MODEL METADATA
# ============================================================

with METADATA_PATH.open("r", encoding="utf-8") as file:
    model_info = json.load(file)

model_name = model_info.get("best_model")

if model_name not in MODEL_COLUMNS:
    raise ValueError(
        f"Unsupported best_model: {model_name!r}. "
        f"Expected one of {list(MODEL_COLUMNS)}."
    )

score_column = MODEL_COLUMNS[model_name]

try:
    threshold = float(model_info["threshold"])
except (KeyError, TypeError, ValueError) as exc:
    raise ValueError(
        "best_model.json must contain a numeric 'threshold'."
    ) from exc

if not 0 <= threshold <= 1:
    raise ValueError("The decision threshold must be between 0 and 1.")

selection_method = str(
    model_info.get("selected_by", "Not recorded")
)


# ============================================================
# 4. LOAD PREDICTIONS AND EVALUATION RESULTS
# ============================================================

predictions = pd.read_csv(
    PREDICTIONS_PATH,
    dtype={"address": str},
)

results = pd.read_csv(RESULTS_PATH)

required_columns = {"address", score_column}
missing_columns = required_columns - set(predictions.columns)

if missing_columns:
    raise ValueError(
        f"Missing prediction columns: {sorted(missing_columns)}. "
        f"Available columns: {predictions.columns.tolist()}"
    )

if predictions.empty:
    raise ValueError("The prediction CSV is empty.")

if predictions["address"].isna().any():
    raise ValueError("Missing wallet addresses were found.")

predictions["address"] = predictions["address"].str.strip()

if predictions["address"].eq("").any():
    raise ValueError("Empty wallet addresses were found.")

if predictions["address"].duplicated().any():
    raise ValueError("Duplicate wallet addresses were found.")

predictions[score_column] = pd.to_numeric(
    predictions[score_column],
    errors="raise",
)

if not predictions[score_column].between(0, 1).all():
    raise ValueError("Prediction scores must be between 0 and 1.")

wallet_lookup = predictions.set_index("address")

# Count of available evaluation plots.
available_plot_count = sum(
    (PLOTS_DIR / filename).is_file()
    for filename in PLOT_FILES.values()
)


# ============================================================
# 5. WALLET LOOKUP
# ============================================================

def lookup_wallet(address):
    address = str(address or "").strip()

    if not address:
        return (
            "Enter a wallet address.",
            "—",
            "No wallet address was provided.",
        )

    if address not in wallet_lookup.index:
        return (
            "Wallet not found",
            "—",
            (
                "This address does not exist in the saved test "
                "predictions. Only wallets in the offline dataset "
                "can be searched."
            ),
        )

    score = float(
        wallet_lookup.at[address, score_column]
    )

    if score >= threshold:
        decision = "Flagged for illicit-activity review"
    else:
        decision = "Not flagged at this threshold"

    details = (
        f"Selected model: {model_name}\n"
        f"Saved score: {score:.6f}\n"
        f"Decision threshold: {threshold:.6f}\n"
        f"Model selection method: {selection_method}\n"
        "Source: saved offline test-set prediction\n\n"
        "Important: This model score is not proof of fraud, "
        "criminal activity, or fraud-ring membership."
    )

    return decision, f"{score:.6f}", details


def clear_lookup():
    return "", "", "", ""


# ============================================================
# 6. EXAMPLE WALLETS
# ============================================================

examples = [
    [address]
    for address in predictions["address"].head(5).tolist()
]


# ============================================================
# 7. CUSTOM DARK DASHBOARD CSS
# ============================================================

CUSTOM_CSS = """
:root {
    color-scheme: dark;
}

body {
    background: #070B14 !important;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    background: #070B14 !important;
    color: #E5EDF8 !important;
    font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
}

footer {
    display: none !important;
}

#hero {
    background:
        radial-gradient(
            circle at 90% 15%,
            rgba(0, 217, 255, 0.12),
            transparent 35%
        ),
        linear-gradient(125deg, #101C32, #0B1220 65%, #10102A);
    border: 1px solid #20344F;
    border-radius: 20px;
    padding: 30px;
    margin-bottom: 20px;
}

#hero h1 {
    color: #F0F7FF;
    font-size: clamp(27px, 4vw, 38px);
    font-weight: 800;
    letter-spacing: -1px;
    margin: 12px 0;
}

#hero p {
    color: #9DB1CA;
    line-height: 1.8;
    max-width: 800px;
}

.eyebrow {
    color: #00D9FF;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
}

.hero-status {
    color: #91A6C0;
    font-size: 11px;
    letter-spacing: 1px;
    margin-top: 20px;
}

.metric {
    background: linear-gradient(145deg, #101B2D, #0B1321);
    border: 1px solid #20344F;
    border-radius: 15px;
    padding: 20px;
    min-height: 105px;
    height: 100%;
    box-sizing: border-box;
}

.metric-label {
    color: #91A6C0;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #F0F7FF;
    font-size: clamp(19px, 2.2vw, 27px);
    font-weight: 800;
    margin-top: 15px;
    overflow-wrap: anywhere;
}

.section-heading {
    color: #E8F2FF;
    font-size: 20px;
    font-weight: 750;
}

.muted {
    color: #91A6C0;
    font-size: 13px;
    line-height: 1.8;
}

.panel {
    background: #0D1524;
    border: 1px solid #20314A;
    border-radius: 16px;
    padding: 20px;
}

button.primary {
    background: linear-gradient(90deg, #0077FF, #00B8D9) !important;
    border: 0 !important;
    color: white !important;
    font-weight: 750 !important;
}

button.primary:hover {
    filter: brightness(1.12);
}

button.secondary {
    border: 1px solid #293D58 !important;
}

input, textarea {
    background: #080F1C !important;
    border-color: #293D58 !important;
}

.tab-nav button {
    color: #AFC3DA !important;
}

.tab-nav button.selected {
    color: #00D9FF !important;
    border-color: #00D9FF !important;
}

.risk-panel {
    background: #0A1422;
    border: 1px solid #263C57;
    border-radius: 14px;
    padding: 18px;
    margin-top: 12px;
}

#disclaimer {
    color: #8EA4BF;
    border-top: 1px solid #20314A;
    padding-top: 18px;
    margin-top: 24px;
    margin-bottom: 10px;
    font-size: 12px;
    line-height: 1.8;
}

.gradio-container .prose h2,
.gradio-container .prose h3 {
    color: #E5EDF8 !important;
}

.gradio-container .prose p,
.gradio-container .prose li {
    color: #A9BBD1;
}

.gradio-container table {
    border-color: #293D58 !important;
}

@media (max-width: 640px) {
    #hero {
        padding: 20px;
    }

    .metric {
        padding: 14px;
        min-height: 95px;
    }
}
"""


# ============================================================
# 8. BUILD THE DASHBOARD
# ============================================================

with gr.Blocks(
    title="FraudRing-GNN | Wallet Intelligence",
    theme=gr.themes.Base(),
    css=CUSTOM_CSS,
) as demo:

    # --------------------------------------------------------
    # Header
    # --------------------------------------------------------

    gr.HTML("""
    <div id="hero">
        <div class="eyebrow">
            GRAPH INTELLIGENCE / RESEARCH LAB
        </div>

        <h1>FraudRing-GNN</h1>

        <p>
            Blockchain wallet risk intelligence powered by graph learning.
            Explore saved predictions, compare machine learning models,
            and inspect model evaluation results in one workspace.
        </p>

        <div class="hero-status">
            OFFLINE INFERENCE &nbsp; • &nbsp;
            MODEL EVALUATION &nbsp; • &nbsp;
            RESEARCH DEMONSTRATION
        </div>
    </div>
    """)

    # --------------------------------------------------------
    # Summary cards
    # --------------------------------------------------------

    with gr.Row(equal_height=True):
        gr.HTML(f"""
        <div class="metric">
            <div class="metric-label">Selected Model</div>
            <div class="metric-value">{html.escape(model_name)}</div>
        </div>
        """)

        gr.HTML(f"""
        <div class="metric">
            <div class="metric-label">Saved Wallets</div>
            <div class="metric-value">{len(predictions):,}</div>
        </div>
        """)

        gr.HTML(f"""
        <div class="metric">
            <div class="metric-label">Decision Threshold</div>
            <div class="metric-value">{threshold:.4f}</div>
        </div>
        """)

        gr.HTML(f"""
        <div class="metric">
            <div class="metric-label">Evaluation Plots</div>
            <div class="metric-value">
                {available_plot_count} / {len(PLOT_FILES)}
            </div>
        </div>
        """)

    gr.Markdown("## Research workspace")

    # --------------------------------------------------------
    # Tabs
    # --------------------------------------------------------

    with gr.Tabs():

        # ====================================================
        # WALLET INTELLIGENCE
        # ====================================================

        with gr.Tab("Wallet Intelligence"):

            gr.Markdown("""
            ### Inspect a wallet prediction

            Search an exact address in the saved test dataset.
            The result uses the selected model's precomputed score.
            """)

            with gr.Group(elem_classes=["panel"]):

                wallet_input = gr.Textbox(
                    label="Bitcoin wallet address",
                    placeholder="Paste a wallet address here...",
                    lines=1,
                )

                with gr.Row():
                    search_button = gr.Button(
                        "Analyze Saved Result",
                        variant="primary",
                    )

                    clear_button = gr.Button(
                        "Clear",
                        variant="secondary",
                    )

                with gr.Row():
                    decision_output = gr.Textbox(
                        label="Model decision",
                        interactive=False,
                        lines=2,
                    )

                    score_output = gr.Textbox(
                        label="Illicit-class score",
                        interactive=False,
                    )

                details_output = gr.Textbox(
                    label="Prediction details",
                    interactive=False,
                    lines=6,
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

                clear_button.click(
                    fn=clear_lookup,
                    inputs=[],
                    outputs=[
                        wallet_input,
                        decision_output,
                        score_output,
                        details_output,
                    ],
                )

            gr.Markdown("### Example wallets")

            gr.Examples(
                examples=examples,
                inputs=wallet_input,
                label="Available test-set examples",
            )

            gr.Markdown("""
            <div class="muted">
                A high model score is not proof of criminal activity.
                Only wallets present in the saved prediction dataset can
                be searched. No live blockchain analysis is performed.
            </div>
            """)

        # ====================================================
        # MODEL PERFORMANCE
        # ====================================================

        with gr.Tab("Model Performance"):

            gr.Markdown("""
            ### Experimental model comparison

            Review the saved evaluation results for GraphSAGE, GCN,
            and XGBoost. Interpret each metric using the evaluation
            protocol documented in your research.
            """)

            if results.empty:
                gr.Markdown(
                    "No evaluation result rows are available."
                )
            else:
                gr.Dataframe(
                    value=results,
                    interactive=False,
                    label="Final Evaluation Results",
                    wrap=True,
                )

            gr.Markdown("""
            <div class="muted">
                These results describe an offline experiment.
                They do not guarantee performance on unseen wallets,
                other datasets, or future transactions.
            </div>
            """)

        # ====================================================
        # EVALUATION LAB
        # ====================================================

        with gr.Tab("Evaluation Lab"):

            gr.Markdown("""
            ### Model diagnostics

            Inspect the plots generated by your training and evaluation
            pipeline. Expand a section to view its image.
            """)

            available_plots = [
                (title, str(PLOTS_DIR / filename))
                for title, filename in PLOT_FILES.items()
                if (PLOTS_DIR / filename).is_file()
            ]

            if not available_plots:
                gr.Markdown("""
                No plots were found.

                Make sure the generated PNG files are stored inside
                the `plots/` directory in your repository.
                """)
            else:
                for title, plot_path in available_plots:
                    with gr.Accordion(title, open=False):
                        gr.Image(
                            value=plot_path,
                            label=title,
                            interactive=False,
                            show_download_button=True,
                            container=True,
                        )

        # ====================================================
        # ABOUT
        # ====================================================

        with gr.Tab("About"):

            gr.Markdown("""
            ## About FraudRing-GNN

            FraudRing-GNN is a research demonstration for exploring
            precomputed wallet-risk predictions and offline model
            evaluation results.

            ### Available features

            - Search saved test-set wallet predictions.
            - Inspect the selected model's score and threshold.
            - Review comparative model evaluation metrics.
            - Inspect ROC, precision-recall, confusion matrix,
              threshold sensitivity, and other available plots.

            ### Current limitations

            - No live blockchain connection.
            - No predictions for addresses absent from the saved dataset.
            - A flagged wallet is not a confirmed fraudster.
            - A single wallet score does not establish fraud-ring membership.
            """)

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    gr.HTML("""
    <div id="disclaimer">
        <strong>RESEARCH DEMONSTRATION ONLY</strong><br>
        Offline predictions · No live blockchain analysis ·
        Not a fraud determination system
    </div>
    """)


# ============================================================
# 9. RENDER THE APP ON RENDER.COM
# ============================================================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
    )
