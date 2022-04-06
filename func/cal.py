import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
import pandas as pd
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
device = "cuda:0" if torch.cuda.is_available() else "cpu"


def eval_on_features(regress, x_train, y_train, x_test, y_test, title, iv, way, fig_size=(12, 8), alpha=0.5):
    num_train, num_test = x_train.shape[0], x_test.shape[0]
    num = num_train + num_test
    y_length = y_train.shape[1]
    for i in range(y_length):
        y_train_one = y_train[:, i]
        regress.fit(x_train, y_train_one)
        y_train_pred_one = regress.predict(x_train).reshape(-1, 1)
        y_test_pred_one = regress.predict(x_test).reshape(-1, 1)
        if i == 0:
            y_train_pred = y_train_pred_one
            y_test_pred = y_test_pred_one
        else:
            y_train_pred = np.concatenate((y_train_pred, y_train_pred_one), axis=1)
            y_test_pred = np.concatenate((y_test_pred, y_test_pred_one), axis=1)

    rmse_train = get_rmse(y_train_pred, y_train)
    rmse_test = get_rmse(y_test_pred, y_test)
    r2_train = get_r2_score(y_train_pred, y_train)
    r2_test = get_r2_score(y_test_pred, y_test)

    plt.figure(figsize=fig_size)
    # plt.title(title, fontsize=30)
    length_train = y_train_pred[:, 0].shape[0]
    length_test = y_test_pred[:, 0].shape[0]
    lim_train = np.arange(length_train // iv)
    lim_test = np.arange(length_train // iv, length_train // iv + length_test // iv)
    plt.plot(lim_train, sam(y_train[:, 0], iv, way), label="$y^{train}$", alpha=alpha)
    plt.plot(lim_train, sam(y_train_pred[:, 0], iv, way), label="$\hat{y}^{train}$", alpha=alpha)
    plt.plot(lim_test, sam(y_test[:, 0], iv, way), label="$y^{test}$", alpha=alpha)
    plt.plot(lim_test, sam(y_test_pred[:, 0], iv, way), label="$\hat{y}^{test}$", alpha=alpha)
    plt.legend(fontsize=25, loc=1)
    plt.ylabel("Value", fontsize=30)
    plt.xlabel("Time", fontsize=30)
    return rmse_train, rmse_test, r2_train, r2_test


def create_inout_sequences(input_data, x_length=32, y_length=4, style="list", ml_dim=0, ld1=False):
    seq_list, seq_arr, label_arr = [], None, None
    data_length = len(input_data)
    x_y_length = x_length + y_length
    if style == "list":
        for i in range(data_length - x_y_length + 1):
            if input_data.ndim == 2:
                seq = input_data[i: (i + x_length), :]
                if ld1:
                    label = input_data[(i + x_length): (i + x_length + y_length), ml_dim]
                else:
                    label = input_data[(i + x_length): (i + x_length + y_length), :]
            elif input_data.ndim == 1:
                seq = input_data[i: (i + x_length)]
                label = input_data[(i + x_length): (i + x_length + y_length)]
            elif input_data.ndim == 3:
                seq = input_data[i: (i + x_length), :, :]
                if ld1:
                    label = input_data[(i + x_length): (i + x_length + y_length), :, ml_dim]
                else:
                    label = input_data[(i + x_length): (i + x_length + y_length), :, :]
            seq_list.append((seq, label))
        return seq_list

    elif style == "arr":
        for i in range(data_length - x_y_length + 1):
            if input_data.ndim == 2:
                seq = input_data[i: (i + x_length), :]
                label = input_data[(i + x_length): (i + x_length + y_length), ml_dim].reshape(1, -1)
                seq = np.expand_dims(seq, 0)
            elif input_data.ndim == 1:
                seq = input_data[i: (i + x_length)]
                label = input_data[(i + x_length): (i + x_length + y_length)]
                seq, label = seq.reshape(1, -1), label.reshape(1, -1)
            if (seq_arr is None) & (label_arr is None):
                seq_arr, label_arr = seq, label
            else:
                seq_arr, label_arr = np.vstack([seq_arr, seq]), np.vstack([label_arr, label])
        return seq_arr, label_arr


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


class GNNTime(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, edge_weight, gnn_style, num_layers=1):
        super(GNNTime, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.edge_weight = nn.Parameter(edge_weight)
        self.gnn_style = gnn_style
        self.num_layers = num_layers
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())
        self.gcn1 = gnn.GCNConv(input_dim, hidden_dim)
        self.gcn2 = gnn.GCNConv(hidden_dim, output_dim)
        self.che1 = gnn.ChebConv(input_dim, hidden_dim, K=2)
        self.che2 = gnn.ChebConv(hidden_dim, output_dim, K=2)
        self.sage1 = gnn.SAGEConv(input_dim, hidden_dim)
        self.sage2 = gnn.SAGEConv(hidden_dim, output_dim)
        self.gin1 = gnn.GraphConv(input_dim, hidden_dim)
        self.gin2 = gnn.GraphConv(hidden_dim, output_dim)
        self.tran1 = gnn.TransformerConv(input_dim, hidden_dim)
        self.tran2 = gnn.TransformerConv(hidden_dim, output_dim)
        self.tag1 = gnn.TAGConv(input_dim, hidden_dim)
        self.tag2 = gnn.TAGConv(hidden_dim, output_dim)
        self.cnn1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.cnn2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.cnn3 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.cnn4 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x, edge_index):
        if self.gnn_style == "GCN":             # Graph Convolution Network Model
            h = self.gcn1(x, edge_index, self.edge_weight)
            h = self.pre(h)
            h = self.gcn2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "Cheb":           # Chebyshev Network
            h = self.che1(x, edge_index)
            h = self.pre(h)
            h = self.che2(h, edge_index)
        elif self.gnn_style == "GraphSage":          # GraphSAGE Model
            h = self.sage1(x, edge_index)
            h = self.pre(h)
            h = self.sage2(h, edge_index)
        elif self.gnn_style == "GIN":           # Graph Isomorphic Network Model
            h = self.gin1(x, edge_index, self.edge_weight)
            h = self.pre(h)
            h = self.gin2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "Tran":
            h = self.tran1(x, edge_index)
            h = self.pre(h)
            h = self.tran2(h, edge_index)
        elif self.gnn_style == "Tag":
            h = self.tag1(x, edge_index, self.edge_weight)
            h = self.pre(h)
            h = self.tag2(h, edge_index, self.edge_weight)
        elif self.gnn_style == "ResGraphNet":
            h = self.sage1(x, edge_index)
            h = self.pre(h)
            h = h.unsqueeze(0)
            h = h.unsqueeze(0)
            out = self.cnn1(h)
            out_0 = out
            out = self.cnn2(self.pre(out))
            out = self.cnn3(self.pre(out))
            out = out + out_0
            out = self.cnn4(self.pre(out))
            h = out.squeeze(0)
            h = h.squeeze(0)
            h = self.pre(h)
            h = self.sage2(h, edge_index)
        else:
            raise TypeError("{} is unknown for gnn style".format(self.gnn_style))
        return h


def path_graph(m):
    adm = np.zeros(shape=(m, m))
    for i in range(m - 1):
        adm[i, i + 1] = 1
    return adm


def tran_adm_to_edge_index(adm):
    if isinstance(adm, np.ndarray):
        u, v = np.nonzero(adm)
        num_edges = u.shape[0]
        edge_index = np.vstack([u.reshape(1, -1), v.reshape(1, -1)])
        edge_weight = np.zeros(shape=u.shape)
        for i in range(num_edges):
            edge_weight_one = adm[u[i], v[i]]
            edge_weight[i] = edge_weight_one
        edge_index = torch.from_numpy(edge_index.astype(np.int64))
        edge_weight = torch.from_numpy(edge_weight.astype(np.float32))
        return edge_index, edge_weight
    elif torch.is_tensor(adm):
        u, v = torch.nonzero(adm, as_tuple=True)
        num_edges = u.shape[0]
        edge_index = torch.cat((u.view(1, -1), v.view(1, -1)))
        edge_weight = torch.zeros(size=u.shape)
        for i in range(num_edges):
            edge_weight[i] = adm[u[i], v[i]]
        return edge_index, edge_weight


def get_rmse(a1, a2):
    a1, a2 = a1.reshape(-1), a2.reshape(-1)
    m_1, m_2 = a1.shape[0], a2.shape[0]
    if m_1 < m_2:
        m = m_1
    else:
        m = m_2
    a1_m, a2_m = a1[:m], a2[:m]
    result = np.sqrt(np.sum(np.square(a1_m - a2_m)) / m)
    return result


def plot_2curve(a1, a2, l1=None, l2=None, title=None, x_name=None, y_name=None, alpha=0.5, fig_size=(12, 12),
                font_size=20, font_size_le=15, style='plot', s=10, marker=False):
    plt.figure(figsize=fig_size)
    if style == 'stem':
        plt.stem(a1, label=l1, alpha=alpha)
        plt.stem(a2, label=l2, alpha=alpha)
    elif style == 'plot':
        if marker:
            plt.plot(a1, label=l1, marker="o", markersize=10, alpha=alpha)
            plt.plot(a2, label=l2, marker="o", markersize=10, alpha=alpha)
        else:
            plt.plot(a1, label=l1, alpha=alpha)
            plt.plot(a2, label=l2, alpha=alpha)
    elif style == 'scatter':
        plt.scatter(x=np.linspace(1, a1.shape[0], a1.shape[0]), y=a1, label=l1, s=s)
        plt.scatter(x=np.linspace(1, a2.shape[0], a2.shape[0]), y=a2, label=l2, s=s)
    plt.xlabel(x_name, fontsize=font_size)
    plt.ylabel(y_name, fontsize=font_size)
    if (l1 is None) & (l2 is None):
        pass
    else:
        plt.legend(fontsize=font_size_le)
    plt.title(title, fontsize=font_size)
    return None


def plot_distribute(x, bins, jump, title=None, x_name=None, y_name="Frequency"):
    if x.ndim == 1:
        pass
    elif (x.ndim == 2) & (x.shape[1] == 1):
        x = x.reshape(-1)
    else:
        raise TypeError("x must be 1d or row vector of 2d, but got {}d".format(x.shape[1]))
    x_label, x_bins = pd.cut(x, bins=bins, retbins=True)
    x_label = pd.DataFrame(x_label)
    x_label_vc = pd.DataFrame(x_label).value_counts()
    interval = x_label_vc.index.tolist()
    interval_sum = x_label_vc.values
    mid_all = []
    for i in range(bins):
        interval_one = interval[i][0]
        mid = (interval_one.left + interval_one.right) / 2
        mid_all.append(mid)
    mid_all = np.array(mid_all)
    sort_index = np.argsort(mid_all)
    mid_all_sort = mid_all[sort_index]
    mid_all_sort = np.around(mid_all_sort, 2)
    interval_sum_sort = interval_sum[(sort_index)]

    plt.figure(figsize=(12, 12))
    if title is not None:
        plt.title(title, fontsize=30)
    plt.bar(x=np.arange(bins), height=interval_sum_sort)
    labels = mid_all_sort[np.arange(0, bins, jump)]
    ticks = np.arange(bins)[np.arange(0, bins, jump)]
    plt.xticks(ticks=ticks, labels=labels)
    if x_name is not None:
        plt.xlabel(x_name, fontsize=30)
    if y_name is not None:
        plt.ylabel(y_name, fontsize=30)
    return None


# 计算真实标签与预测结果的r2分数
def get_r2_score(output, y):
    if (type(output) is np.ndarray) & (type(y) is np.ndarray):      # numpy数组类型
        output = output.reshape(-1)
        y = y.reshape(-1)
    elif (torch.is_tensor(output)) & (torch.is_tensor(y)):          # pytorch张量类型
        output = output.detach().cpu().numpy().reshape(-1)
        y = y.detach().cpu().numpy().reshape(-1)
    else:
        raise TypeError("type of output and y must be the same, but got {} and {}".format(type(output), type(y)))
    r2 = r2_score(y, output)
    return r2


class ResNet(nn.Module):
    def __init__(self, hidden_dim, output_dim, x_length):
        super(ResNet, self).__init__()
        self.lin_pre = nn.Linear(1, hidden_dim)
        self.cnn1 = nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1))
        self.cnn2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.cnn3 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1))
        self.cnn4 = nn.Conv2d(32, 1, kernel_size=(3, 3), padding=(1, 1))
        self.last1 = nn.Linear(hidden_dim, 1)
        self.last2 = nn.Linear(x_length, output_dim)
        self.pre = nn.Sequential(nn.ReLU(), nn.Dropout())

    def forward(self, x):
        h = x.unsqueeze(1).unsqueeze(3)
        h = self.lin_pre(h)
        h = self.cnn1(h)
        h_0 = h
        h = self.cnn2(self.pre(h))
        h = self.cnn3(self.pre(h))
        h = h + h_0
        h = self.cnn4(self.pre(h))
        h = self.last1(self.pre(h)).squeeze(3).squeeze(1)
        h = self.last2(h)
        return h


class MyData(Dataset):
    def __init__(self, x, y):
        super(MyData, self).__init__()
        self.x = torch.from_numpy(x).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        x_one = self.x[idx, :]
        y_one = self.y[idx, :]
        return x_one, y_one


# 等间隔采样
def sam(arr, iv, way="one"):
    num = arr.shape[0] // iv
    arr_sam = []
    for i in range(num):
        if way == "one":                    # 等间隔采样
            arr_one = arr[i * iv]
        elif way == "mean":                 # 等间隔划分区间，取区间内的平均值
            arr_one_range = arr[i * iv: (i + 1) * iv]
            arr_one = np.mean(arr_one_range)
        else:
            raise TypeError("Unknown type of way!")
        arr_sam.append(arr_one)
    arr_sam = np.array(arr_sam)
    return arr_sam
