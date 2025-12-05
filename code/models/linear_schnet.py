import math
import multiprocessing
from datetime import datetime
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
#from quanti_gin.shared import generate_min_global_distance_edges, read_data_file
from torch.nn import Dropout, Linear, Sequential, Sigmoid
#from torch.utils.tensorboard.writer import SummaryWriter
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SchNet
from torch_geometric.nn.models.schnet import GaussianSmearing, ShiftedSoftplus
from torch_geometric.typing import OptTensor, Tensor
from tqdm import tqdm

#from vqm.training.gnn.gnn import EarlyStopping, build_knn_graph

# device = torch.device("mps")
# if device is None:
torch.set_num_threads(4)

model_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


class CutoffSchNet(SchNet):
    def forward(self, z: Tensor, pos: Tensor, batch: OptTensor = None) -> Tensor:
        r"""Forward pass.

        Args:
            z (torch.Tensor): Atomic number of each atom with shape
                :obj:`[num_atoms]`.
            pos (torch.Tensor): Coordinates of each atom with shape
                :obj:`[num_atoms, 3]`.
            batch (torch.Tensor, optional): Batch indices assigning each atom
                to a separate molecule with shape :obj:`[num_atoms]`.
                (default: :obj:`None`)
        """
        batch = torch.zeros_like(z) if batch is None else batch

        h = self.embedding(z)
        edge_index, edge_weight = self.interaction_graph(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + torch.nn.functional.dropout(
                interaction(h, edge_index, edge_weight, edge_attr),
                p=0.3,
                training=self.training,
            )

        h = self.lin1(h)
        return h


class LinearSchNet(torch.nn.Module):
    """
    Customized SchNet implementation for edge prediction tasks.

    This model adapts the SchNet architecture from PyTorch Geometric to predict
    edge properties rather than node or graph properties.
    """

    def __init__(
        self,
        hidden_dim=128,
        num_filters=128,
        num_interactions=6,
        num_gaussians=50,
        cutoff=10.0,
        max_num_neighbors=32,
        use_edge_features=True,
        larger_mlp=False,
        mlp_increase=4,
        use_gaussian_smearing=False,
        mlp_skip_connection=False,
        dropout=0.3,
    ):
        super(LinearSchNet, self).__init__()

        self.use_edge_features = use_edge_features
        self.use_gaussian_smearing = use_gaussian_smearing

        # Initialize the SchNet model from PyTorch Geometric
        self.schnet = CutoffSchNet(
            hidden_channels=hidden_dim,
            num_filters=num_filters,
            num_interactions=num_interactions,
            num_gaussians=num_gaussians,
            cutoff=cutoff,
            max_num_neighbors=max_num_neighbors,
        )

        # two features (angle and distance)
        self.edge_feature_count = 2
        if self.use_gaussian_smearing:
            self.distance_expansion = GaussianSmearing(0.0, cutoff, num_gaussians)
            # with gaussian smearing, the first edge feature gets expanded to dim num_gaussians
            self.edge_feature_count = num_gaussians + 1

        # Edge prediction MLP, CutoffSchNet outputs one number for each atom,
        edge_input_dim = hidden_dim + (
            self.edge_feature_count if use_edge_features else 0
        )

        if larger_mlp:
            self.edge_mlp = Sequential(
                Linear(edge_input_dim, edge_input_dim * mlp_increase),
                Dropout(p=dropout),
                ShiftedSoftplus(),
                Linear(
                    edge_input_dim * mlp_increase, edge_input_dim * mlp_increase // 2
                ),
                Dropout(p=dropout),
                ShiftedSoftplus(),
                Linear(edge_input_dim * mlp_increase // 2, 1),
            )
        else:
            self.edge_mlp = Sequential(
                Linear(edge_input_dim, edge_input_dim * mlp_increase),
                Dropout(p=dropout),
                Sigmoid(),
                Linear(edge_input_dim * mlp_increase, 1),
            )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def forward(
        self,
        z,
        batch=None,
        pos=None,
        edges=None,
    ):
        """
        Forward pass of the CustomSchNet model.

        Args:
            x: Node features
            edge_index: Edge indices for prediction
            edge_attr: Edge features for prediction edges
            batch: Batch indices for nodes
            pos: Node positions (used by SchNet instead of x)

        Returns:
            Edge predictions
        """

        x, coordinates, edge_index, edge_attr = prepare_prediction_datapoint(pos, edges[0])
        edge_attr.to(self.device)
        edge_index.to(self.device)

        # Get node embeddings from SchNet
        node_embeddings = self.schnet(
            z=z,
            pos=pos,
            batch=batch,
        )

        # Extract node embeddings for edges to predict
        edge_src = node_embeddings[edge_index[0]]  # Source node embeddings
        edge_dst = node_embeddings[edge_index[1]]  # Target node embeddings
        edge_features = torch.cat([edge_src, edge_dst], dim=1)

        # Add edge-specific features if provided
        if self.use_edge_features and edge_attr is not None:
            distance_feat = edge_attr[:, 0]
            # unsqueeze: batch dimension
            angle_feat = edge_attr[:, 1].unsqueeze(1)

            # apply gaussian smearing if wanted
            if self.use_gaussian_smearing:
                distance_feat = self.distance_expansion(distance_feat)

            distance_feat = distance_feat.to(self.device, non_blocking=True)
            angle_feat = angle_feat.to(self.device, non_blocking=True)
            edge_features = edge_features.to(self.device, non_blocking=True)
            distance_feat = distance_feat.unsqueeze(1)

            edge_features = torch.cat([edge_features, distance_feat, angle_feat], dim=1)

        # Predict edge values
        edge_values = self.edge_mlp(edge_features).squeeze(dim=1)
        return edge_values


def prepare_prediction_datapoint(coordinates, edges, scalers=None):
    coordinates = torch.tensor(coordinates, dtype=torch.float)
    edges = [(a.item(), b.item()) for a, b in edges]

    edge_set = np.array(edges)
    edge_index = torch.tensor(edge_set).t().contiguous()

    edge_features = []
    for src, dst in edge_index.t().tolist():
        dist = torch.norm(coordinates[src] - coordinates[dst])
        angle = torch.arccos(
            torch.dot(coordinates[src], coordinates[dst])
            / (torch.norm(coordinates[src]) * torch.norm(coordinates[dst]) + 1e-5)
        )
        edge_features.append([dist, angle])

    edge_features = torch.tensor(edge_features, dtype=torch.float)  # .unsqueeze(1)

    # make sure that every atom knows, what system size we are working with
    # if scaling to non-hydrogenic systems, it makes sense to change this to the atomic number or at least add it
    x = torch.tensor([len(coordinates) for _ in range(len(coordinates))])

    # create Data conventional to pytorch-geometric
    # For SchNet, we need to provide positions
    return x, coordinates, edge_index, edge_features

