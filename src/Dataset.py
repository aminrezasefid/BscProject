from turtle import home
import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from DataTransformer import DataTransformer


class Dataset:
    def __init__(self, filename):
        self.filename = filename

    def process(self):
        dt = DataTransformer(self.filename)
        data_train, data_val, data_test, data_test_final, teams_enc = dt.prepare_data(val_sep=0.8,test_sep=0.9)
        n_teams = len(teams_enc['teams'].values)
        win_lose_network = np.zeros((dt.n_teams, 2, dt.n_teams))
        x = torch.ones(dt.n_teams).reshape(-1, 1)
        edge_time = np.empty((dt.n_teams, dt.n_teams))
        edge_time[:] = None
        node_time = np.zeros(dt.n_teams)
        won = data_test[data_test['FTR'] == "H"].shape[0]/data_test.shape[0]
        lost = data_test[data_test['FTR'] == "A"].shape[0]/data_test.shape[0]
        draw = data_test[data_test['FTR'] == "D"].shape[0]/data_test.shape[0]
        data = Data(
            edge_index=torch.tensor([]).reshape(2,-1).long(),
            edge_weight=torch.tensor([]),
            matches=data_train,
            n_teams=n_teams,
            win_lose_network=win_lose_network,
            node_time=node_time,
            node_weight=None,
            edge_time=edge_time,
            data_val=data_val,
            data_test=data_test,
            data_test_final=data_test_final,
            curr_time=0,
            N=dt.N,
            baseline=max(won, lost, draw),
            train_loss=[],
            train_accuracy=[],
            val_loss=[],
            val_accuracy=[],
            teams_enc=teams_enc
        )
        return data
