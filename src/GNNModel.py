import torch
from torch.nn import LogSoftmax, ReLU, Tanh, LeakyReLU, ModuleList, Dropout
from torch_geometric.nn import GCNConv, GraphConv, ChebConv

target_dim = 3

activations = {
    'relu': ReLU(),
    'tanh': Tanh(),
    'leaky': LeakyReLU()
}


class GNNModel(torch.nn.Module):
    def __init__(self, num_teams, embed_dim,runMainCode, n_conv=3, conv_dims=(32, 32, 32, 16), n_dense=5, dense_dims=(8, 8, 8, 8,8),
                 act_f='leaky', **kwargs):
        super(GNNModel, self).__init__()
        self.embed_dim = embed_dim
        self.n_conv = n_conv
        self.runMainCode=runMainCode
        self.conv_dims = conv_dims
        self.n_dense = n_dense
        self.activation = activations[act_f]
        self.num_teams = num_teams
        self.embedding = torch.nn.Embedding(num_embeddings=num_teams, embedding_dim=embed_dim)
        conv_layers = [GraphConv(self.embed_dim, self.conv_dims[0])]
        for i in range(n_conv - 1):
            conv_layers.append(GraphConv(conv_dims[i], conv_dims[i + 1]))
        self.conv_layers = ModuleList(conv_layers)

        lin_layers = []
        lin_layers.append(torch.nn.Linear(conv_dims[n_conv - 1]*2, dense_dims[0]))
        for i in range(n_dense - 2):
            lin_layers.append(torch.nn.Linear(dense_dims[i], dense_dims[i + 1]))
        lin_layers.append(torch.nn.Linear(dense_dims[n_dense - 2], target_dim))

        self.lin_layers = ModuleList(lin_layers)

        self.out = LogSoftmax(dim=1)
        self.drop = Dropout(p=0.1)


    def forward(self, data, home, away):
        edge_index, edge_weight = data.edge_index, data.edge_weight
        if hasattr(self, 'num_teams'):
            num_teams = self.num_teams
        else:
            num_teams = data.n_teams
        x = torch.tensor(list(range(num_teams)))
        if self.runMainCode:
            x = self.embedding(x).reshape(-1, self.embed_dim)
        else :
            x = data.team_features

        if len(edge_weight) > 0:
            x = self.conv_layers[0](x, edge_index, edge_weight )
        else:
            x = self.conv_layers[0](x, edge_index)
        x = self.activation(x)
        x = self.drop(x)

        for i in range(self.n_conv - 1):
            if len(edge_weight) > 0:
                    x = self.activation(self.conv_layers[i + 1](x, data.edge_index, edge_weight))
            else:
                x = self.activation(self.conv_layers[i + 1](x, data.edge_index))
            # x = self.drop(x)

        x = torch.cat([x[home], x[away]], dim=1)
        # x = torch.sub(x[home], x[away])

        for i in range(self.n_dense):
            x = self.activation(self.lin_layers[i](x))
            x = self.drop(x)

        x = self.out(x)
        return x.reshape(-1, target_dim)
