import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import os.path as osp
import pandas as pd
import matplotlib.patheffects as PathEffects
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KernelDensity
from matplotlib.collections import LineCollection


def read(root, f_list):
    f_address = root
    for i in f_list:
        f_address = osp.join(f_address, i)
    f_style = f_list[-1].split('.')[-1]
    file_name = f_list[-1].split('.')[0]
    a = f_list[:-1]
    a.append(file_name)
    f_name = "_".join(a)
    if f_style == 'npy':
        temp = np.load(f_address)
        return temp, [], f_name
    elif f_style == 'txt':
        if f_list[0] == 'HadSST3':
            txt = np.loadtxt(f_address, dtype=str)
            temp = txt[:, 1].astype(float)
            time = txt[:, 0]
            return temp, time, f_name
        else:
            raise TypeError("Unknown dataset type of f_list[0]!")
    elif f_style == 'csv':
        df = pd.read_csv(f_address)
        if f_list[0] == 'HadCRUT5':
            temp = df.loc[:, 'Anomaly (deg C)'].values.reshape(-1)
            time = df.loc[:, 'Time'].values.reshape(-1)
            return temp, time, f_name
        elif f_list[0] == 'Solar':
            solar = df.loc[:, 'Power(MW)'].values.reshape(-1)
            time = df.loc[:, 'LocalTime'].values.reshape(-1)
            return solar, time, f_name
        elif f_list[0] == 'Exchange_Rate':
            rate = df.loc[:, 'KOREA - WON/US$'].values.reshape(-1)
            time = df.loc[:, 'Time Serie'].values.reshape(-1)
            idx = np.argwhere(rate != 'ND').reshape(-1)
            rate, time = rate[idx].astype(float), time[idx]
            return rate, time, f_name
        elif f_list[0] == 'PM25':
            df = df.dropna(subset=['PM_US Post'])
            pm = df.loc[:, 'PM_US Post'].values.reshape(-1)
            time = df.loc[:, ['year', 'month', 'day', 'hour']].values.reshape(-1)
            return pm, time, f_name
        else:
            raise TypeError("Unknown dataset type of f_list[0]!")
    elif f_style == 'xlsx':
        if f_list[0] == 'Traffic':
            df = pd.read_excel(f_address, sheet_name='Report Data')
            observe = df.loc[:, '% Observed'].values.reshape(-1)
            time = df.loc[:, 'Hour'].values.reshape(-1)
            return observe, time, f_name
    else:
        raise TypeError("Unknown Type of data file!")


def get_train_or_test_idx(num, num_train):
    idx_all = np.arange(num)
    idx_train = np.sort(np.random.choice(num, num_train, replace=False))
    idx_test = np.array(list(set(idx_all) - set(idx_train)))
    return idx_train.tolist(), idx_test.tolist()


class SelfData(Dataset):
    def __init__(self, data, label, *args):
        super(SelfData, self).__init__()
        self.data = data
        self.label = label
        self.args = args
        self.data_else = self.get_data_else()

    def get_data_else(self):
        num = len(self.args)
        if num != 0:
            data_else = []
            for i in range(num):
                data_else_one = self.args[i]
                data_else.append(data_else_one)
        else:
            data_else = None
        return data_else

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        if self.data.dim() == 3:
            data_one = self.data[item, :, :]
        elif self.data.dim() == 2:
            data_one = self.data[item, :]
        elif self.data.dim() == 4:
            data_one = self.data[item, :, :, :]
        else:
            raise ValueError("data.dim() must be 3 or 4, but got {}".format(self.data.dim()))
        if self.label.dim() == 1:
            label_one = self.label[item]
        elif self.label.dim() == 2:
            label_one = self.label[item, :]
        elif self.label.dim() == 3:
            label_one = self.label[item, :, :]
        else:
            raise ValueError("label.dim() must be 1 or 2, but got {}".format(self.label.dim()))
        return_all = [data_one, label_one]
        data_else_one = []
        if self.data_else is not None:
            num = len(self.data_else)
            for i in range(num):
                x = self.data_else[i]
                if x.dim() == 2:
                    x_one = x[item, :]
                elif x.dim() == 1:
                    x_one = x[item]
                elif x.dim() == 3:
                    x_one = x[item, :, :]
                else:
                    raise ValueError("data_else dim() must be 1, 2 or 3, but got {}".format(x_one.dim()))
                data_else_one.append(x_one)
        return_all = return_all + data_else_one
        return_all.append(item)
        return_all = tuple(return_all)
        return return_all


def ts_un(n, k):
    adm = np.zeros(shape=(n, n))
    if k < 1:
        raise ValueError("k must be greater than or equal to 1")
    else:
        for i in range(n):
            if i < (n - k):
                for k_one in range(1, k + 1):
                    adm[i, i + k_one] = 1.
            else:
                for k_one in range(1, k + 1):
                    if (k_one + i) >= n:
                        mod = (k_one + i) % n
                        adm[i, mod] = 1.
                    else:
                        adm[i, i + k_one] = 1.
    adm = (adm.T + adm) / 2
    adm = adm * 0.5
    return adm


def path_graph(m):
    adm = np.zeros(shape=(m, m))
    for i in range(m - 1):
        adm[i + 1, i] = 1
    return adm


def tran_adm_to_edge_index(adm):
    u, v = np.nonzero(adm)
    num_edges = u.shape[0]
    edge_index = np.vstack([u.reshape(1, -1), v.reshape(1, -1)])
    edge_weight = np.zeros(shape=u.shape)
    for i in range(num_edges):
        edge_weight_one = adm[u[i], v[i]]
        edge_weight[i] = edge_weight_one
    edge_index = torch.from_numpy(edge_index).long()
    edge_weight = torch.from_numpy(edge_weight).float()
    return edge_index, edge_weight


def prep_tran(train_, test_, prep_style, be_torch=False):
    if prep_style == "sta":
        prep = StandardScaler()
    elif prep_style == "min":
        prep = MinMaxScaler()
    elif prep_style is None:
        return train_, test_, None
    else:
        raise TypeError("Unknown Type of prep_style!")
    train = train_.reshape(-1, 1)
    prep.fit(train)
    train_prep = prep.transform(train)
    train_prep = train_prep.reshape(train_.shape)
    if test_ is not None:
        test = test_.reshape(-1, 1)
        test_prep = prep.transform(test)
        test_prep = test_prep.reshape(test_.shape)
        if be_torch:
            train_prep = torch.from_numpy(train_prep).float()
            test_prep = torch.from_numpy(test_prep).float()
        return train_prep, test_prep, prep
    else:
        if be_torch:
            train_prep = torch.from_numpy(train_prep).float()
        return train_prep, prep


def prep_inv(prep, *args):
    if prep is not None:
        num = len(args)
        if num == 0:
            raise ValueError("Please input data for inverse-normalization!")
        inv = []
        for i in range(num):
            one_ = args[i]
            one = one_.reshape(-1, 1)
            one_inv = prep.inverse_transform(one)
            one_inv = one_inv.reshape(one_.shape)
            inv.append(one_inv)
        if num == 1:
            inv = inv[0]
        else:
            inv = tuple(inv)
        return inv
    else:
        return args


def get_xy(data, lx, ly, style="list"):
    xy, x, y = [], [], []
    m = len(data)
    if style == "list":
        for i in range(m - lx - ly + 1):
            if data.ndim == 2:
                x = data[i: (i + lx), :]
                y = data[(i + lx): (i + lx + ly), :]
            elif data.ndim == 1:
                x = data[i: (i + lx)]
                y = data[(i + lx): (i + lx + ly)]
            elif data.ndim == 3:
                x = data[i: (i + lx), :, :]
                y = data[(i + lx): (i + lx + ly), :, :]
            else:
                raise TypeError("!")
            xy.append((x, y))
        return xy
    elif style == "arr":
        x_arr, y_arr = [], []
        for i in range(m - lx - ly + 1):
            if data.ndim == 2:
                x = data[i: (i + lx), :]
                y = data[(i + lx): (i + lx + ly)].reshape(1, -1)
                x = np.expand_dims(x, 0)
            elif data.ndim == 1:
                x = data[i: (i + lx)]
                y = data[(i + lx): (i + lx + ly)]
                x, y = x.reshape(1, -1), y.reshape(1, -1)
            else:
                raise TypeError("!")
            if i == 0:
                x_arr, y_arr = x, y
            else:
                x_arr, y_arr = np.vstack([x_arr, x]), np.vstack([y_arr, y])
        return x_arr, y_arr


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = 1
    return mask


def get_lc(x_list, month_idx, month_points, c_list, ls_list, cmap_list, deg, max_label):
    r_max, r_min, lc_list = -float('inf'), float('inf'), []
    for i in range(len(x_list)):
        x_ori = x_list[i]
        if max_label is not None:
            x_ori = np.append(x_ori, max_label)
        x = x_ori - deg
        x_vals, y_vals, r_max_, r_min_ = xy_vals(x, month_idx, month_points)
        if max_label is not None:
            x, x_vals, y_vals = x[:-1], x_vals[:-1], y_vals[:-1]
        if r_max_ > r_max:
            r_max = r_max_
        if r_min_ < r_min:
            r_min = r_min_

        pts = np.array([x_vals, y_vals]).T.reshape(-1, 1, 2)
        segments = np.concatenate([pts[:-1], pts[1:]], axis=1)
        if len(c_list) == 0:
            lc = LineCollection(segments, cmap=plt.get_cmap(cmap_list[i]), linestyles=ls_list[i], linewidths=3)
        elif len(cmap_list) == 0:
            lc = LineCollection(segments, color=c_list[i], linestyles=ls_list[i], linewidths=3)
        else:
            raise TypeError("c_list and cmap_list must have one to be []!")
        lc.set_array(np.asarray(x))
        lc_list.append(lc)
    return r_min, r_max, lc_list


def spirals(x_list, ls_list, c_list, cmap_list, fig_si, fo_si, fo_ti_si, title, title_c="white", deg=0, max_label=None):
    months = ["Mar", "Feb", "Jan", "Dec", "Nov", "Oct", "Sep", "Aug", "Jul", "Jun", "May", "Apr"]
    month_idx = [2, 1, 0, 11, 10, 9, 8, 7, 6, 5, 4, 3]
    month_points = segment_circle(len(months))

    r_min, r_max, lc_list = get_lc(x_list, month_idx, month_points, c_list, ls_list, cmap_list, deg, max_label)

    fig, ax = plt.subplots(figsize=fig_si)
    fig.patch.set_facecolor('white')
    txt = ax.set_title(title, fontsize=fo_si, pad=50, color=title_c, loc='left')
    # txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
    ax.axis('equal')
    ax.set(xlim=(-r_max * 1.2, r_max * 1.2), ylim=(-r_max * 1.2, r_max * 1.2))

    circle = plt.Circle((0, 0), r_max * 1.3, fc='black')
    ax.add_patch(circle)
    circle_max = plt.Circle((0, 0), r_max, ec='white', fc=None, fill=False, lw=3.0)
    ax.add_patch(circle_max)
    # circle_min = plt.Circle((0, 0), r_min, ec='red', fc=None, fill=False, lw=3.0)
    # ax.add_patch(circle_min)

    props_months = {'ha': 'center', 'va': 'center', 'fontsize': fo_ti_si, 'color': 'black'}
    props_temp = {'ha': 'center', 'va': 'center', 'fontsize': fo_ti_si, 'color': 'white'}
    ax.text(0, -r_max * 1.1, '{}°C'.format(round(r_max + deg, 2)), props_temp, bbox=dict(facecolor='black', edgecolor="black"))
    # ax.text(0, -r_min * 0.6, '{}°C'.format(round(r_min + deg, 2)), props_temp, bbox=dict(facecolor='black', edgecolor="black"))

    for j in range(0, len(months)):
        x_unit_r, y_unit_r, angle = month_points[j]
        x_radius, y_radius = r_max * x_unit_r * 1.4, r_max * y_unit_r * 1.4
        angle = angle - 0.5 * np.pi
        ax.text(x_radius, y_radius, months[j], props_months, rotation=np.rad2deg(angle), )

    for lc in lc_list:
        plt.gca().add_collection(lc)
    ax.autoscale()
    ax.axis("off")
    return fig


def segment_circle(num_segments):
    segment_rad = 2 * np.pi / num_segments
    segment_rads = segment_rad * np.arange(num_segments)
    coordX = np.cos(segment_rads)
    coordY = np.sin(segment_rads)
    return np.c_[coordX, coordY, segment_rads]


def xy_vals(x, month_idx, month_points):
    x_vals, y_vals = [], []
    for i in range(0, len(x)):
        x_unit_r, y_unit_r = month_points[month_idx[i % 12], :2]
        x_r, y_r = (x[i] * x_unit_r, x[i] * y_unit_r)
        x_vals.append(x_r), y_vals.append(y_r)
    r_max = np.max(np.sqrt(np.square(np.array(x_vals)) + np.square(np.array(y_vals))))
    r_min = np.min(np.sqrt(np.square(np.array(x_vals)) + np.square(np.array(y_vals))))
    return x_vals, y_vals, r_max, r_min


def cal_dist(x, bins):
    x_label, x_bins = pd.cut(x, bins=bins, retbins=True)
    x_label = pd.DataFrame(x_label)
    x_label_vc = pd.DataFrame(x_label).value_counts()
    interval = x_label_vc.index.tolist()
    interval_sum = x_label_vc.values
    mid_all, left, right = [], float('inf'), -float('inf')
    for i in range(bins):
        interval_one = interval[i][0]
        left_one, right_one = interval_one.left, interval_one.right
        mid = (left_one + right_one) / 2
        mid_all.append(mid)
        if left_one < left:
            left = left_one
        if right_one > right:
            right = right_one
    mid_all = np.array(mid_all)
    sort_index = np.argsort(mid_all)
    mid_all_sort = mid_all[sort_index]
    mid_all_sort = np.around(mid_all_sort, 2)
    interval_sum_sort = interval_sum[(sort_index)]
    return mid_all_sort, interval_sum_sort, left, right


def plot_dist(x, bins, jump, pos, title, fig_si, fo_si, fo_ti_si, fo_te, x_name, y_name="Frequency"):
    if x.ndim == 1:
        pass
    elif (x.ndim == 2) & (x.shape[1] == 1):
        x = x.reshape(-1)
    else:
        raise TypeError("x must be 1d or row vector of 2d, but got {}d".format(x.shape[1]))
    mid_all_sort, interval_sum_sort, left, right = cal_dist(x, bins)

    fig = plt.figure(figsize=fig_si)
    ax = fig.add_subplot(111)
    if title is not None:
        ax.set_title(title, fontsize=fo_si)
    ax.bar(x=np.arange(bins), height=interval_sum_sort, color="lightcoral", edgecolor="black", linewidth=2,
           label="Distribution")
    ax.patch.set_facecolor('grey')
    ax.patch.set_alpha(0.3)
    ax.set_axisbelow(True)
    ax.yaxis.grid(c="white", linewidth=5)
    ax.xaxis.grid(c="white", linewidth=5)
    if pos != []:
        mean, std = np.mean(x), np.std(x)
        axes = plt.gca()
        lim_x_min, lim_x_max = axes.get_xlim()
        lim_y_min, lim_y_max = axes.get_ylim()
        lim_x_length, lim_y_length = lim_x_max - lim_x_min, lim_y_max - lim_y_min
        x_loc = pos[0] * lim_x_length + lim_x_min
        y_loc = pos[1] * lim_y_length + lim_y_min
        t = ax.text(x_loc, y_loc, "Mean = {:.5f}\n Std = {:.5f}".format(mean, std), fontsize=fo_te)
        t.set_bbox(dict(facecolor="lightcoral", alpha=0.5, edgecolor="lightcoral"))

    labels = mid_all_sort[np.arange(0, bins, jump)]
    ticks = np.arange(bins)[np.arange(0, bins, jump)]
    plt.xticks(ticks=ticks, labels=labels, fontsize=fo_ti_si)
    plt.yticks(fontsize=fo_ti_si)
    if x_name is not None:
        plt.xlabel(x_name, fontsize=fo_si, labelpad=20)
    if y_name is not None:
        plt.ylabel(y_name, fontsize=fo_si, labelpad=20)
    return fig

