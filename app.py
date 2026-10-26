import json
import os

import gradio as gr
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "outputs", "results")

RESULTS_PATH = os.path.join(RESULTS_DIR, "final_results.csv")
PREDICTIONS_PATH = os.path.join(RESULTS_DIR, "gnn_test_predictions.csv")
MODEL_INFO_PATH = os.path.join(RESULTS_DIR, "best_model.json")


def load_data():
    results = pd.read_csv(RESULTS_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    with open(MODEL_INFO_PATH, "r") as f:
        model_info = json.load(f)

    return results, predictions, model_info


results_df, predictions_df, model_info = load_data()


def search_wallet(wallet_address):
    if not wallet_address:
        return "Please enter a wallet address."

    match = predictions_df[
        predictions_df["address"].str.lower() == wallet_address.strip().lower()
    ]

    if match.empty:
        return "Wallet not found in the test-set predictions."

    row = match.iloc[0]

    return (
        f"Wallet: {row['address']}\n"
        f"True label: {'Illicit' if row['y_true'] == 1 else 'Licit'}\n"
        f"XGBoost risk score: {row['prob_xgboost']:.4f}\n"
        f"GCN risk score: {row['prob_gcn']:.4f}\n"
        f"GraphSAGE risk score: {row['prob_graphsage']:.4f}"
    )


with gr.Blocks(title="FraudRing GNN") as demo:
    gr.Markdown(
        """
        # FraudRing GNN
        ### Illicit Cryptocurrency Wallet Detection

        Compare XGBoost, GCN, and GraphSAGE risk scores for wallets in the test set.
        """
    )

    with gr.Row():
        wallet_input = gr.Textbox(
            label="Wallet address",
            placeholder="Enter a test-set wallet address",
        )
        search_button = gr.Button("Search")

    output_box = gr.Textbox(label="Result", lines=8)

    search_button.click(
        fn=search_wallet,
        inputs=wallet_input,
        outputs=output_box,
    )

    gr.Markdown("### Final model comparison")
    gr.Dataframe(value=results_df, interactive=False)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        share=False,
    )