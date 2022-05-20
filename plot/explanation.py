"""
Draw images to enhance interpretability and explaination
"""
import numpy as np
import pandas as pd
import torch
import os
import os.path as osp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

import sys
sys.path.append("..")
import func.cal as cal


def bnp(h):
    return h.detach().cpu().numpy()


device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
l_x = 60                   # Data sequence length
l_y = 1                    # Label sequence length
lr = 0.0001                  # Learning rate
weight_decay = 5e-4
epochs = 4000
hidden_dim = 64
save_fig = True                  # Whether to save picture
ratio_train = 0.5               # Proportion of training datasets
fig_size = (16, 16)
ts_name_all = ["cli_dash", "HadCRUT5", "temp_month", "temp_year", "elect", "traffic", "sales"]
ts_name_folder = "HadCRUT5"    # Name of the folder where the data resides
ts_name = "HadCRUT5_global"       # Name of the selected time series
iv = 1                          # sampling interval, used for plotting curves
way = "mean"                    # The style of plot curves of real data and predict results

x_address = osp.join("../datasets", ts_name_folder, ts_name + ".npy")
x = np.load(x_address)
num = x.shape[0]                # The length of time series

graph_address = osp.join("../graph", ts_name)
pca = PCA(n_components=1)

num_train = int(ratio_train * num)
data_train, data_test = x[:num_train], x[num_train:num]     # get training dataset and test dataset

"""
ResGraphNet
"""
mid = np.array(data_test.tolist() + [data_test[-1]] * 9)

len_interp = l_y + 6
data_test_ = np.array(mid[:-l_y].tolist() + mid[-len_interp-l_y:-l_y].tolist() + mid[-l_y:].tolist())

# Using Graph Neural network, prepare data information
x_train, y_train = cal.create_inout_sequences(data_train, l_x, l_y, style="arr")
x_test, y_test = cal.create_inout_sequences(data_test_, l_x, l_y, style="arr")

x_train = torch.from_numpy(x_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)
num_nodes = x_train.shape[0] + x_test.shape[0]
num_train = x_train.shape[0]

x = torch.cat((x_train, x_test), dim=0).to(device)
y = torch.cat((y_train, y_test), dim=0).to(device)

adm = cal.path_graph(num_nodes)
edge_index, edge_weight = cal.tran_adm_to_edge_index(adm)
edge_index = edge_index.to(device)

train_index = torch.arange(num_train, dtype=torch.long)
test_index = torch.arange(num_train, num_nodes, dtype=torch.long)
train_mask = cal.index_to_mask(train_index, num_nodes).to(device)
test_mask = cal.index_to_mask(test_index, num_nodes).to(device)

ResGraphNet_address = osp.join("../result", ts_name, "ResGraphNet")
model = torch.load(osp.join(ResGraphNet_address, "ResGraphNet.pkl")).to(device)
sage1, sage2 = model.sage1, model.sage2
cnn1, cnn2, cnn3, cnn4, cnn5, cnn6 = model.cnn1, model.cnn2, model.cnn3, model.cnn4, model.cnn5, model.cnn6
linear1, linear2 = model.linear1, model.linear2
drop = model.drop

out_sage1 = sage1(x, edge_index)
out_cnn1 = cnn1(out_sage1.unsqueeze(0).unsqueeze(0))
out_cnn3 = cnn3(drop(cnn2(drop(out_cnn1)))) + out_cnn1
out_cnn5 = cnn5(drop(cnn4(drop(out_cnn3)))) + out_cnn3
out_cnn6 = cnn6(out_cnn5).squeeze(0).squeeze(0)
out_sage2 = sage2(out_cnn6, edge_index)

out_sage1 = pca.fit_transform(bnp(out_sage1[test_mask, :][:-len_interp, :]))
out_res = pca.fit_transform(bnp(out_cnn6[test_mask, :][:-len_interp, :]))
out_sage2 = out_sage2[test_mask, :][:-len_interp, -1]

y = y[test_mask][:-len_interp, -1]

plt.figure(figsize=fig_size)
plt.plot(bnp(y), alpha=0.5, linestyle='--', label="$y$", c="b")
plt.plot(out_sage1, label="$f_{1}$", alpha=0.5, c="g")
plt.plot(out_res, label="$f_{2}$", alpha=0.5, c="orange")
plt.plot(bnp(out_sage2), label="$\hat{y}$", alpha=0.5, c="r")
plt.legend(fontsize=30)
plt.xlabel("Year", fontsize=40)
plt.ylabel("Anomaly ($^{\circ}$C)", fontsize=40)
x_tick = [0, 240, 480, 720, 960]
x_label = ["1940", "1960", "1980", "2000", "2020"]
plt.xticks(x_tick, x_label, fontsize=25)
plt.yticks(fontsize=25)
# plt.title("ResGraphNet", fontsize=40)
if save_fig:
    plt.savefig(osp.join(graph_address, "explanation_ResGraphNet.png"))

"""
GNN Model
"""
x_train, y_train = cal.create_inout_sequences(data_train, l_x, l_y, style="arr")
x_test, y_test = cal.create_inout_sequences(data_test, l_x, l_y, style="arr")

x_train = torch.from_numpy(x_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)
num_nodes = x_train.shape[0] + x_test.shape[0]

x = torch.cat((x_train, x_test), dim=0)
y = torch.cat((y_train, y_test), dim=0)

adm = cal.path_graph(num_nodes)
edge_index, edge_weight = cal.tran_adm_to_edge_index(adm)
edge_index = edge_index.to(device)

train_index = torch.arange(num_train, dtype=torch.long)
test_index = torch.arange(num_train, num_nodes, dtype=torch.long)
train_mask = cal.index_to_mask(train_index, num_nodes).to(device)
test_mask = cal.index_to_mask(test_index, num_nodes).to(device)

GNNModel_address = osp.join("../result", ts_name, "GNNModel")
model = torch.load(osp.join(GNNModel_address, "GraphSage.pkl")).to(device)
sage1, sage2 = model.sage1, model.sage2
drop = model.drop

out_sage1 = sage1(x, edge_index)
out_sage2 = sage2(drop(out_sage1), edge_index)

out_sage1 = pca.fit_transform(bnp(out_sage1[test_mask, :]))
out_sage2 = out_sage2[test_mask, :][:, -1]
y = y[test_mask, :][:, -1]

plt.figure(figsize=fig_size)
plt.plot(bnp(y), alpha=0.5, linestyle='--', label="$y$", c="b")
plt.plot(out_sage1, label="$f_{1}^{\ \ ''}$", alpha=0.5, c="g")
plt.plot(bnp(out_sage2), label="$\hat{y}^{\ ''}$", alpha=0.5, c="r")
plt.legend(fontsize=30)
# plt.title("GNNModel", fontsize=40)
plt.xlabel("Year", fontsize=40)
plt.ylabel("Anomaly ($^{\circ}$C)", fontsize=40)
x_tick = [0, 240, 480, 720, 960]
x_label = ["1940", "1960", "1980", "2000", "2020"]
plt.xticks(x_tick, x_label, fontsize=25)
plt.yticks(fontsize=25)
if save_fig:
    plt.savefig(osp.join(graph_address, "explanation_GNNModel.png"))

"""
RES Model
"""
batch_size = 32
x_train, y_train = cal.create_inout_sequences(data_train, l_x, l_y, style="arr")
x_test, y_test = cal.create_inout_sequences(data_test, l_x, l_y, style="arr")

train_dataset = cal.MyData(x_train, y_train)
test_dataset = cal.MyData(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

RESModel_address = osp.join("../result", ts_name, "RESModel")
model = torch.load(osp.join(RESModel_address, "RESModel.pkl")).to(device)

lin_pre = model.lin_pre
cnn1, cnn2, cnn3, cnn4, cnn5, cnn6 = model.cnn1, model.cnn2, model.cnn3, model.cnn4, model.cnn5, model.cnn6
last1, last2 = model.last1, model.last2
drop = model.drop

out_fc1, out_res, out_fc2, true = [], [], [], []
for item, (x_test, y_test) in enumerate(test_loader):
    x_test, y_test = x_test.to(device), y_test.to(device)

    out_fc1_one = lin_pre(x_test.unsqueeze(1).unsqueeze(3))

    out_cnn1_one = cnn1(out_fc1_one)
    out_cnn2_one = cnn2(drop(out_cnn1_one))
    out_cnn3_one = cnn3(drop(out_cnn2_one)) + out_cnn1_one

    out_cnn4_one = cnn4(drop(out_cnn3_one))
    out_cnn5_one = cnn5(drop(out_cnn4_one)) + out_cnn3_one
    out_res_one = cnn6(out_cnn5_one)

    out_fc2_one = last2(last1(out_res_one).squeeze(3).squeeze(1))

    out_fc1_one = out_fc1_one.detach().cpu().numpy()[:, 0, :, :]
    out_fc1_one = pca.fit_transform(np.max(out_fc1_one, axis=2))
    out_res_one = out_res_one.detach().cpu().numpy()[:, 0, :, :]
    out_res_one = pca.fit_transform(np.max(out_res_one, axis=2))
    out_fc2_one = out_fc2_one.detach().cpu().numpy()[:, -1]
    true_one = y_test.detach().cpu().numpy()[:, -1]
    if item == 0:
        out_fc1 = out_fc1_one
        out_res = out_res_one
        out_fc2 = out_fc2_one
        true = true_one
    else:
        out_fc1 = np.concatenate((out_fc1, out_fc1_one), axis=0)
        out_res = np.concatenate((out_res, out_res_one), axis=0)
        out_fc2 = np.concatenate((out_fc2, out_fc2_one), axis=0)
        true = np.concatenate((true, true_one), axis=0)

plt.figure(figsize=fig_size)
plt.plot(true, alpha=0.5, linestyle='--', label="$y$", c="b")
plt.plot(out_fc1, label="$f_{1}^{\ \ '}$", alpha=0.5, c="g")
plt.plot(out_res, label="$f_{2}^{\ \ '}$", alpha=0.5, c="orange")
plt.plot(out_fc2, label="$\hat{y}^{\ '}$", alpha=0.5, c="r")
plt.legend(fontsize=30)
plt.xlabel("Year", fontsize=40)
plt.ylabel("Anomaly ($^{\circ}$C)", fontsize=40)
x_tick = [0, 240, 480, 720, 960]
x_label = ["1940", "1960", "1980", "2000", "2020"]
plt.xticks(x_tick, x_label, fontsize=25)
plt.yticks(fontsize=25)
# plt.title("RESModel", fontsize=40)
if save_fig:
    plt.savefig(osp.join(graph_address, "explanation_RESModel.png"))

print()
plt.show()
print()
