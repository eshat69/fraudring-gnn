import gradio as gr
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops

# -------------------------
# File paths
# -------------------------
MODEL_PATH = "best_graphsage_model.pt"
METADATA_PATH = "gnn_metadata.pkl"
FEATURES_PATH = "wallet_graph_features_with_labels.csv"

device = torch.device("cpu")

# -------------------------
# Feature columns
# -------------------------
feature_cols = [
    "in_degree",
    "out_degree",
    "total_degree",
    "degree_centrality",
    "num_predecessors",
    "num_successors",
    "num_neighbors"
]

# -------------------------
# Load metadata
# -------------------------
metadata = joblib.load(METADATA_PATH)

threshold = metadata["sage_threshold"]

# -------------------------
# Load wallet features
# -------------------------
features_df = pd.read_csv(FEATURES_PATH)
features_df["address"] = features_df["address"].astype(str)

wallet_ids = features_df["address"].tolist()
wallet_to_idx = {
    wallet: idx
    for idx, wallet in enumerate(wallet_ids)
}

X = features_df[feature_cols].values.astype(np.float32)
y = features_df["label"].values.astype(np.int64)

x = torch.tensor(X, dtype=torch.float)
y_tensor = torch.tensor(y, dtype=torch.long)

# -------------------------
# GraphSAGE model
# -------------------------
class GraphSAGE(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        out_channels=2,
        dropout=0.3
    ):
        super().__init__()

        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)

        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv3(x, edge_index)
        return x


model = GraphSAGE(
    in_channels=len(feature_cols),
    hidden_channels=64,
    out_channels=2
)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.eval()

# -------------------------
# Prediction function
# -------------------------
def predict_wallet(wallet_address):
    wallet_address = str(wallet_address).strip()

    if wallet_address not in wallet_to_idx:
        return (
            "Wallet not found in the available dataset.",
            "",
            ""
        )

    idx = wallet_to_idx[wallet_address]

    wallet_features = torch.tensor(
        X[idx:idx + 1],
        dtype=torch.float
    )

    # Minimal self-loop graph for single-wallet inference
    edge_index = torch.tensor(
        [[0], [0]],
        dtype=torch.long
    )

    with torch.no_grad():
        out = model(wallet_features, edge_index)
        illicit_prob = F.softmax(out, dim=1)[0, 1].item()

    if illicit_prob >= threshold:
        prediction = "Illicit / High Risk"
    else:
        prediction = "Licit / Low Risk"

    if illicit_prob >= 0.8:
        risk_level = "High"
    elif illicit_prob >= 0.5:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return (
        prediction,
        f"{illicit_prob:.4f}",
        f"Risk level: {risk_level} | Decision threshold: {threshold:.4f}"
    )


# -------------------------
# Gradio interface
# -------------------------
demo = gr.Interface(
    fn=predict_wallet,
    inputs=gr.Textbox(
        label="Wallet Address",
        placeholder="Enter a Bitcoin wallet address"
    ),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Illicit Probability"),
        gr.Textbox(label="Risk Details")
    ],
    title="FraudRing-GNN",
    description=(
        "Real-time wallet illicit-activity risk scoring using GraphSAGE. "
        "Enter a wallet address available in the Elliptic++ dataset."
    ),
    article=(
        "Note: This is a simplified deployment demo. "
        "Full GraphSAGE inference requires the wallet's neighborhood graph "
        "and transaction history."
    ),
    examples=[
        ["13vHWR3iLsHeYwT42RnuKYNBoVPrKKZgRv"]
    ]
)

demo.launch()