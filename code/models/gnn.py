from datetime import datetime
import math
import atexit
import multiprocessing
import os
import glob
from time import perf_counter

import torch
from torch.nn import ModuleList
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from quanti_gin.shared import generate_min_global_distance_edges, np, read_data_file
from scipy.spatial import distance_matrix
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv
from tqdm import tqdm

# parameter to control whether GNN is generating undirected graphs
UNDIRECTED = False

torch.set_num_threads(8)

model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


def build_knn_graph(node_features, edge_set, k=3):
    """Builds a k-NN graph based on node features."""
    coords = node_features.cpu().numpy()  # Convert to numpy
    dist_matrix = distance_matrix(coords, coords)
    knn_edges = []
    features = []
    for i in range(len(coords)):
        neighbors = dist_matrix[i].argsort()[1 : k + 1]  # Exclude self-loop (0th index)
        for neighbor in neighbors:
            knn_edges.append((i, neighbor))
            angle = np.arccos(
                np.dot(coords[i], coords[neighbor])
                / (
                    (np.linalg.norm(coords[i]) * np.linalg.norm(coords[neighbor]))
                    + 1e-5
                )
            )

            features.append([dist_matrix[i][neighbor], angle])
    edge_index = torch.tensor(knn_edges, dtype=torch.long).t().contiguous()
    return edge_index, features


# Hybrid GNN model
class GNN(torch.nn.Module):
    def __init__(
        self,
        hidden_dim=64,
        input_dim=3,
        use_edge_features=True,
        use_graph_attention=True,
        additional_depth=False,
        gat_layers=2,
        gat_heads=1,
        gat_dropout=0.0,
        gat_residual=False,
        gatconv_only=False,
        larger_mlp=False,
    ):
        super(GNN, self).__init__()
        self.use_edge_features = use_edge_features
        self.use_graph_attention = use_graph_attention
        self.additional_depth = additional_depth
        self.gatconv_only = gatconv_only
        self.gat_dropout = gat_dropout
        actual_hidden_dim = hidden_dim

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # Message passing layers
        if not self.use_graph_attention:
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)

        if self.use_graph_attention and not self.gatconv_only:
            self.gatconv = GATConv(
                input_dim,
                hidden_dim,
                dropout=gat_dropout,
                heads=gat_heads,
                residual=gat_residual,
            )
            actual_hidden_dim = hidden_dim * gat_heads
            if self.additional_depth:
                self.gat_mlp = torch.nn.Sequential(
                    torch.nn.Linear(actual_hidden_dim, actual_hidden_dim * 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(actual_hidden_dim * 4, hidden_dim),
                )
                self.gatconv2 = GATConv(hidden_dim, hidden_dim, residual=True)
                self.gat_mlp_2 = torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim * 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_dim * gat_heads, hidden_dim),
                )

        # only use gat and no MLP in between, only one MLP at the end
        if self.use_graph_attention and self.gatconv_only:
            self.conv_stack = ModuleList(
                [
                    GATConv(
                        hidden_dim * gat_heads if i > 0 else input_dim,
                        hidden_dim,
                        heads=gat_heads,
                        dropout=gat_dropout,
                        residual=gat_residual,
                    )
                    for i in range(gat_layers)
                ]
            )

            actual_hidden_dim = hidden_dim * gat_heads
        # Fully connected layers for edge prediction,
        # takes in both edges node features
        edge_input_dim = actual_hidden_dim * 2 + (2 if use_edge_features else 0)
        if larger_mlp:
            self.edge_mlp = torch.nn.Sequential(
                torch.nn.Linear(edge_input_dim, edge_input_dim * 4),
                torch.nn.ELU(),
                torch.nn.Linear(edge_input_dim * 4, edge_input_dim * 2),
                torch.nn.ELU(),
                torch.nn.Linear(edge_input_dim * 2, 1),
            )
        else:
            self.edge_mlp = torch.nn.Sequential(
                torch.nn.Linear(edge_input_dim, edge_input_dim * 4),
                torch.nn.ELU(),
                torch.nn.Linear(edge_input_dim * 4, 1),
            )

    def forward(
        self,
        z,
        batch=None,
        pos=None,
        edges=None,
    ):

        data = prepare_prediction_datapoint(pos, edges[0])
        data.to(self.device)

        coordinates, edge_index, global_edge_index, global_edge_attr, global_edge_features, edge_attr = data
        edge_index = extract(edge_index)
        global_edge_index = extract(global_edge_index)
        global_edge_attr = extract(global_edge_attr)
        edge_attr = extract(edge_attr)
        global_edge_features = extract(global_edge_features)

        x = pos
        if self.use_graph_attention:
            if self.gatconv_only:
                x = F.dropout(x, p=self.gat_dropout, training=self.training)
                for m in self.conv_stack:
                    x = m(
                        x, edge_index=global_edge_index, edge_attr=global_edge_features
                    )
                    x = F.elu(F.dropout(x, p=self.gat_dropout, training=self.training))

            else:
                # Message passing with GATConv to allow cross attention between multiple nodes
                # this uses the global edge index for the closest nodes
                global_edge_features = global_edge_features.t()

                print(x, "\n", global_edge_index, "\n", global_edge_features)

                x = self.gatconv(
                    x, edge_index=global_edge_index, edge_attr=global_edge_features
                )
                x = F.relu(x)
                if self.additional_depth:
                    x = self.gat_mlp(x)
                    x = self.gatconv2(
                        x, edge_index=global_edge_index, edge_attr=global_edge_features
                    )
                    x = F.relu(x)
                    x = self.gat_mlp_2(x)
                    x = F.relu(x)
        else:
            x = self.conv1(x, global_edge_index)
            x = F.relu(x)
            x = self.conv2(x, global_edge_index)

        # Extract node embeddings for edges
        edge_src = x[edge_index[0]]  # Source node embeddings
        edge_dst = x[edge_index[1]]  # Target node embeddings
        edge_features = torch.cat([edge_src, edge_dst], dim=1)
        # print(edge_feartu)

        # Add edge-specific features if provided
        if self.use_edge_features and edge_attr is not None:
            edge_features = torch.cat([edge_features, edge_attr], dim=1)

        # Predict edge values
        edge_values = self.edge_mlp(edge_features).squeeze(dim=1)
        return edge_values


def extract(t):
    # if t is ("name", tensor)
    return t[1] if isinstance(t, tuple) else t

def prepare_prediction_datapoint(coordinates, edges, scalers=None):
    edges = [(a.item(), b.item()) for a, b in edges]
    coordinates = torch.tensor(coordinates, dtype=torch.float)
    edge_set = np.array(edges)
    edge_index = torch.tensor(edge_set).t().contiguous()

    # Add global edges for message passing (k-NN graph)
    # That uses to closest k edges
    global_edge_index, global_edge_features = build_knn_graph(
        coordinates, edge_set, k=6
    )
    edge_features = []
    for src, dst in edge_index.t().tolist():
        dist = torch.norm(coordinates[src] - coordinates[dst])
        angle = torch.arccos(
            torch.dot(coordinates[src], coordinates[dst])
            / (torch.norm(coordinates[src]) * torch.norm(coordinates[dst]) + 1e-5)
        )
        edge_features.append([dist, angle])

    edge_features = torch.tensor(edge_features, dtype=torch.float)  # .unsqueeze(1)

    # x = torch.tensor(
    #     [[1.0, 1.0, 1.0] for _ in range(len(coordinates))], dtype=torch.float
    # )
    data = Data(
        x=coordinates,
        edge_index=edge_index,
        global_edge_index=global_edge_index,
        global_edge_attr=torch.tensor(global_edge_features, dtype=torch.float),
        global_edge_features = torch.tensor(global_edge_features),
        edge_attr=edge_features,
    )

    return data


def process_datapoint(args):
    coordinates, edge_target = args
    edges = generate_min_global_distance_edges(coordinates)
    data = prepare_prediction_datapoint(coordinates, edges)
    data.y = torch.tensor(edge_target, dtype=torch.float)
    return data
