import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import numpy as np
import copy
from sklearn.metrics import r2_score


class GlobalTempNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, edge_weight, num_nodes, style):
        super(GlobalTempNet, self).__init__()
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.style = style
        self.edge_weight_1 = nn.Parameter(copy.deepcopy(edge_weight))
        self.edge_weight_2 = nn.Parameter(copy.deepcopy(edge_weight))
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.drop = nn.Dropout()
        self.relu = nn.ReLU()
        self.gcn1 = gnn.GCNConv(in_dim, hid_dim)
        self.gcn2 = gnn.GCNConv(hid_dim, out_dim)
        self.cheb1 = gnn.ChebConv(in_dim, hid_dim, K=2)
        self.cheb2 = gnn.ChebConv(hid_dim, out_dim, K=2)
        kr, pd = (5, 5), (2, 2)
        self.cnn1 = nn.Conv2d(1, hid_dim, kernel_size=kr, padding=pd)
        self.cnn2 = nn.Conv2d(hid_dim, hid_dim, kernel_size=kr, padding=pd)
        self.cnn3 = nn.Conv2d(hid_dim, hid_dim, kernel_size=kr, padding=pd)
        self.cnn4 = nn.Conv2d(hid_dim, hid_dim, kernel_size=kr, padding=pd)
        self.cnn5 = nn.Conv2d(hid_dim, hid_dim, kernel_size=kr, padding=pd)
        self.cnn6 = nn.Conv2d(hid_dim, 1, kernel_size=kr, padding=pd)
        self.bn = nn.BatchNorm2d(hid_dim)
        self.lstm1 = nn.LSTM(hid_dim, hid_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hid_dim, hid_dim, batch_first=True)
        self.gru1 = nn.GRU(hid_dim, hid_dim, batch_first=True)
        self.gru2 = nn.GRU(hid_dim, hid_dim, batch_first=True)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index, self.edge_weight_1)
        h_0 = h
        h = h.unsqueeze(0).unsqueeze(0)
        out = self.cnn1(self.drop(h))

        out_0 = out
        out = self.cnn2(self.drop(self.bn(out)))
        out = self.cnn3(self.drop(self.bn(out)))
        out = out + out_0

        out_2 = out.squeeze(0)
        out_2, (h, c) = self.lstm1(self.drop(out_2))
        out_2, (_, _) = self.lstm2(self.drop(out_2), (self.drop(h), self.drop(c)))
        # out_2, (_) = self.gru1(self.drop(out_2))
        # out_2, (_) = self.gru2(self.drop(out_2))
        out = out + out_2.unsqueeze(0)

        out_1 = out
        out = self.cnn4(self.drop(self.bn(out)))
        out = self.cnn5(self.drop(self.bn(out)))
        out = out + out_1 + out_0

        out = self.cnn6(self.drop(self.bn(out)))
        h = out.squeeze(0).squeeze(0)

        h = h + h_0
        h = self.gcn2(h, edge_index, self.edge_weight_1)
        return h


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(MLP, self).__init__()
        self.drop = nn.Dropout()
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear2 = nn.Linear(hid_dim, hid_dim)
        self.linear3 = nn.Linear(hid_dim ,hid_dim)
        self.linear4 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        h = self.linear1(x)
        h = self.linear2(self.drop(h))
        h = self.linear3(self.drop(h))
        h = self.linear4(h)
        return h


class LSTM(nn.Module):
    def __init__(self, hid_dim, l_x, out_dim):
        super(LSTM, self).__init__()
        self.drop = nn.Dropout()
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.lstm1 = nn.LSTM(1, hid_dim, batch_first=True)
        self.lstm3 = nn.LSTM(hid_dim, 1, batch_first=True)
        self.linear = nn.Linear(l_x, out_dim)

    def forward(self, x):
        out, (_, _) = self.lstm1(x.unsqueeze(2))
        out, (_, _) = self.lstm3(self.pre(out))
        out = self.linear(out.squeeze(2))
        return out


class GRU(nn.Module):
    def __init__(self, hid_dim, l_x, out_dim):
        super(GRU, self).__init__()
        self.drop = nn.Dropout()
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.gru1 = nn.GRU(1, hid_dim, batch_first=True)
        self.gru3 = nn.GRU(hid_dim, 1, batch_first=True)
        self.linear = nn.Linear(l_x, out_dim)

    def forward(self, x):
        out, (_) = self.gru1(x.unsqueeze(2))
        out, (_) = self.gru3(self.pre(out))
        out = self.linear(out.squeeze(2))
        return out


class TDLSTM(nn.Module):
    def __init__(self, out_dim, length):
        super(TDLSTM, self).__init__()
        self.length = length
        self.length_half = length_half = length // 2
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.drop = nn.Dropout()
        self.lstm_fr_1 = nn.LSTM(1, length_half, batch_first=True)
        self.lstm_be_1 = nn.LSTM(length_half, length_half, batch_first=True)
        self.linear1 = nn.Linear(length_half, out_dim)

    def forward(self, x):
        out_fr, out_be = x[:, :self.length_half].unsqueeze(2), x[:, self.length_half:].unsqueeze(1)
        out, (h, c) = self.lstm_fr_1(out_fr)
        out, (h, c) = self.lstm_be_1(out_be, (h, c))
        out = self.linear1(out.squeeze(1))
        return out


class JPSN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(JPSN, self).__init__()
        self.in_dim, self.hid_dim, self.out_dim = in_dim, hid_dim, out_dim
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.drop = nn.Dropout()
        self.linear1 = nn.Linear(in_dim, hid_dim)
        self.linear3 = nn.Linear(hid_dim, out_dim)

    def forward(self, x):
        h = self.linear1(x)
        h = self.linear3(self.drop(h))
        return h


class CFCCLSTM(nn.Module):
    def __init__(self, hid_dim, out_dim, length):
        super(CFCCLSTM, self).__init__()
        self.length = length
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.drop = nn.Dropout()
        self.lstm1 = nn.LSTM(1, length, batch_first=True)
        self.lstm2 = nn.LSTM(length, length, batch_first=True)
        self.cnn1 = nn.Conv2d(1, hid_dim, kernel_size=(5, 5), padding=(2, 2))
        self.cnn2 = nn.Conv2d(hid_dim, 1, kernel_size=(5, 5), padding=(2, 2))
        self.linear1 = nn.Linear(length, 1)
        self.linear2 = nn.Linear(length, out_dim)

    def forward(self, x):
        out, (_, _) = self.lstm1(x.unsqueeze(2))
        out, (_, _) = self.lstm2(self.drop(out))
        out = self.cnn1(self.drop(out.unsqueeze(1)))
        out = self.cnn2(self.drop(out)).squeeze(1)
        out = self.linear1(self.drop(out))
        out = self.linear2(out.squeeze(2))
        return out


# calculate the coefficient of determination, new edition
def cal_r2(y1, y2, axis):
    y1, y2 = be(y1, "n"), be(y2, "n")
    if y1.shape != y2.shape:
        raise ValueError("shape of y1 and y2 must be the same, but got {} and {}".format(y1.shape, y2.shape))
    if axis == 1:
        r2 = r2_score(y1, y2)
    else:
        raise ValueError("axis must be 1, but got {}!".format(axis))
    return r2


# let input data be tensor (pytorch) or array (numpy)
def be(x, style):
    if style == "t":
        if type(x) is np.ndarray:
            x = torch.from_numpy(x)
    elif style == "n":
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
    else:
        raise TypeError("style must be 't' or 'n', but got {}!".format(style))
    return x


def get_r2_score(y1, y2, axis):
    if (type(y1) is np.ndarray) & (type(y2) is np.ndarray):      # numpy数组类型
        pass
    elif (torch.is_tensor(y1)) & (torch.is_tensor(y2)):          # pytorch张量类型
        y1 = y1.detach().cpu().numpy()
        y2 = y2.detach().cpu().numpy()
    else:
        raise TypeError("type of y1 and y must be the same, but got {} and {}".format(type(y1), type(y2)))
    if y1.shape != y2.shape:
        raise ValueError("shape of y1 and y2 must be the same, but got {} and {}".format(y1.shape, y2.shape))
    if y1.ndim == 1:
        y1 = np.expand_dims(y1, axis=1)
        y2 = np.expand_dims(y2, axis=1)
    if axis == 0:
        num_col = y1.shape[0]
    elif axis == 1:
        num_col = y1.shape[1]
    else:
        raise TypeError("axis must be equal as 0 or 1, but got {}".format(axis))
    r2_all = 0
    for i in range(num_col):
        if axis == 0:
            y1_one = y1[i, :]
            y2_one = y2[i, :]
        elif axis == 1:
            y1_one = y1[:, i]
            y2_one = y2[:, i]
        else:
            raise TypeError("axis must be equal as 0 or 1, but got {}".format(axis))
        r2_one = r2_score(y1_one, y2_one)
        r2_all = r2_all + r2_one
    r2 = r2_all / num_col
    return r2

# calculate root mean square error
def get_rmse(y1, y2, axis):
    y1, y2 = be(y1, "n"), be(y2, "n")
    if y1.shape != y2.shape:
        raise ValueError("shape of y1 and y2 must be the same, but got {} and {}".format(y1.shape, y2.shape))
    if y1.ndim == 1:
        y1 = np.expand_dims(y1, axis=1)
        y2 = np.expand_dims(y2, axis=1)
    rmse = 0
    if axis == 1:
        for i in range(y1.shape[1]):
            y1_one, y2_one = y1[:, i], y2[:, i]
            rmse_one = np.sqrt(np.mean(np.square(y1_one - y2_one)))
            rmse = rmse + rmse_one
        rmse = rmse / (y1.shape[1])
    elif axis == 0:
        for i in range(y1.shape[0]):
            y1_one, y2_one = y1[i, :], y2[i, :]
            rmse_one = np.sqrt(np.mean(np.square(y1_one - y2_one)))
            rmse = rmse + rmse_one
        rmse = rmse / (y1.shape[0])
    else:
        raise ValueError("axis must be 0 or 1, but got {}!".format(axis))
    return rmse


def cal_mse(a, b):
    return np.mean(np.square(a - b))


def cal_rmse(a, b):
    return np.sqrt(np.mean(np.square(a - b)))


def cal_mae(a, b):
    return np.mean(np.abs(a - b))


def cal_mape(a, b):
    return np.mean(np.abs((a - b) / a))


def cal_smape(a, b):
    return 2 * np.mean(np.abs(a - b) / (np.abs(a) + np.abs(b)))
