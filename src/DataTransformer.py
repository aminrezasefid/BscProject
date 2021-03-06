import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch

names = ['Div', 'Date', 'time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR',
         'HTHG', 'HTAG','HTR','Referee','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']

class DataTransformer:
    def __init__(self, filename: str):
        self.label_encoder = LabelEncoder()
        self.filename = filename
        self.data = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.n_teams = None
        self.embedding = None
        self.read_data()
        self.data = self.clean_data(self.data)
        self.N = self.data.shape[0]
        self.fit_encoder()
        # self.prepare_data()
    
        
    def read_data(self):
        """Read the data from csv with correct data types."""
        self.data = pd.read_csv(self.filename, header=None, names=names,skiprows=[0],
                                dtype=dict(zip(names, [str] * 5 + [int] * 2 + [str] + [int]*2+[str]*2+[int]*12)))

    def clean_data(self,data, convert_to_numpy=False, allow_draw=True) :
        """Add a column to transform result of the match into int, """
        # result ot int
        conditions = [
            (data['FTR'] == 'H'),
            (data['FTR'] == 'D'),
            (data['FTR'] == 'A')]
        choices = [2,1,0]
        data['lwd'] = np.select(conditions, choices)
        if names[-1] != 'lwd':
            names.append('lwd')
        # ignore the draw results
        if not allow_draw:
            data = self.data[self.data['FTR'] != 'D']
        if convert_to_numpy:
            data = self.data.to_numpy()
        return data

    def fit_encoder(self):
        data = self.data
        data = data.to_numpy()
        teams = np.unique(data[:, [3, 4]])
        self.n_teams = len(teams)
        print(self.n_teams)
        X = data[:, [3, 4]]
        X = X.flatten()
        self.label_encoder.fit(X)

    def encode_teams(self, data):
        data = data.to_numpy()
        teams = np.unique(data[:, [3, 4]])
        # self.n_teams = len(teams)
        X = data[:, [3, 4]]

        X = X.flatten()
        X = self.label_encoder.transform(X)
        teams_encoded = self.label_encoder.transform(teams)
        teams_encoded = pd.DataFrame({'teams': teams, 'label_encoding': teams_encoded})

        data[:, [3, 4]] = np.reshape(X, (-1, 2))
        return data, teams_encoded


    def prepare_data(self, data=None, split_to_test=True, save_to_self=False,val_sep=0.8,test_sep=0.9):
        if data is None:
            data = self.data

        data, teams_encoded = self.encode_teams(data)

        if split_to_test:
            separator_val = int(data.__len__() * val_sep)
            separator_test = int(data.__len__() * test_sep)
            # separator_test_final = int(int(data.__len__() * 0.9))
            data_train = pd.DataFrame(data=data[:separator_val], columns=names)
            data_val = pd.DataFrame(data=data[separator_val:separator_test], columns=names)
            data_test = pd.DataFrame(data=data[separator_test:], columns=names)
            data_test_final = []
            # data_test_final = pd.DataFrame(data=data[separator_test_final:], columns=names)
            self.print_metadata(data_train, "train")
            self.print_metadata(data_val, "val")
            self.print_metadata(data_test, "test")
            if save_to_self:
                self.data_train = data_train
                self.data_val = data_val
                self.data_test = data_test
            self.N = data.shape[0]
            return data_train, data_val, data_test, data_test_final, teams_encoded
        else:
            data = pd.DataFrame(data=data, columns=names)
            if save_to_self:
                self.data = data
            return self.data, teams_encoded

    def to_tensor(self,data):
        home = torch.tensor(data['HomeTeam'].values.astype(int)).to(torch.int64)
        away = torch.tensor(data['AwayTeam'].values.astype(int)).to(torch.int64)
        label = data['lwd'].values.astype(int).reshape(-1,1)
        # self.ohe = OneHotEncoder()
        # self.ohe.fit(label)
        # label = self.ohe.transform(label).toarray()

        label = torch.tensor(label).to(torch.int64)
        return home, away, label

    def get_train_data(self):
        self.print_metadata(self.data_train, "Information on Train data: ")
        home, away, label = self.to_tensor(self.data_train)
        return home, away, label

    def get_test_data(self):
        self.print_metadata(self.data_test, "Information on Test data: ")
        home, away, label = self.to_tensor(self.data_test)
        return home, away, label

    @staticmethod
    def print_metadata(data, message=""):
        # print some metadata
        
        won = data[data['FTR'] == "H"].shape[0]
        lost = data[data['FTR'] == "A"].shape[0]
        draw = data[data['FTR'] == "D"].shape[0]
        total = data.shape[0]
        if total==0: return
        # print("Won:", won, won / total * 100, ", Lost:", lost, lost / total * 100)
        print("Total {} data points: {}, Won: {}%, Lost: {}%, Draw: {}%".format(message, total, won*100 / total, lost*100 / total, draw*100 / total))
        # print("The number of data points in the data set is:", total)