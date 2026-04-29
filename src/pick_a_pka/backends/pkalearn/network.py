import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, ModuleList
from torch_geometric.nn import AttentionalAggregation, GATv2Conv


class PkaLearnGNN(torch.nn.Module):
    def __init__(self, feature_size, edge_dim, model_params):
        super(PkaLearnGNN, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        self.gnn_layers = model_params["model_gnn_layers"]
        self.dense_layers = model_params["model_fc_layers"]
        self.p = model_params["model_dropout_rate"]
        dense_neurons = model_params["model_dense_neurons"]
        n_heads = model_params["model_attention_heads"]

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.bn_layers = ModuleList([])
        self.fc_layers = ModuleList([])

        # GNN Layers
        self.conv1 = GATv2Conv(feature_size, embedding_size, heads=n_heads, edge_dim=edge_dim, dropout=self.p,
                               concat=True
                               )
        self.transf1 = Linear(embedding_size * n_heads, embedding_size)
        self.bn1 = BatchNorm1d(embedding_size)

        for i in range(self.gnn_layers - 1):
            self.conv_layers.append(
                GATv2Conv(embedding_size, embedding_size, heads=n_heads, edge_dim=edge_dim, dropout=self.p, concat=True)
                )
            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(BatchNorm1d(embedding_size))

        self.att = AttentionalAggregation(Linear(embedding_size, 1))

        # We add 2 to the embedding size to account for Molecule Charge and Center Charge
        self.linear1 = Linear(embedding_size + 2, dense_neurons)

        for i in range(self.dense_layers - 1):
            self.fc_layers.append(Linear(dense_neurons, int(dense_neurons / 4)))
            dense_neurons = int(dense_neurons / 4)

        self.out_layer = Linear(dense_neurons, 1)

    def forward(self, x, edge_index, edge_attr, node_index, mol_formal_charge, center_formal_charge, batch_index):
        # 1. GNN Message Passing
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        for i in range(self.gnn_layers - 1):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)

        # 2. Filter nodes to the local environment mask
        x = x[node_index]

        # Update batch_index to match the filtered nodes
        mask = torch.zeros(batch_index.numel(), dtype=torch.bool, device=batch_index.device)
        mask[node_index] = True
        batch_index_filtered = batch_index[mask]

        # 3. Global Pooling (Attention)
        # Results in [num_graphs, embedding_size]
        x = self.att(x, batch_index_filtered)

        # 4. Concatenate Formal Charges
        # Fix for "IndexError: too many indices for tensor of dimension 0"
        # Ensure charges are [num_graphs, 1] regardless of input dimension (0D or 1D)
        mol_formal_charge = mol_formal_charge.view(-1, 1)
        center_formal_charge = center_formal_charge.view(-1, 1)

        # Concat along the feature dimension (dim=1)
        x = torch.cat([x, mol_formal_charge, center_formal_charge], dim=1)

        # 5. Fully Connected Layers
        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=self.p, training=self.training)

        for i in range(self.dense_layers - 1):
            x = torch.relu(self.fc_layers[i](x))
            x = F.dropout(x, p=self.p, training=self.training)

        return self.out_layer(x)
