"""
Test the performance of ResGraphNet under different l_y
The paper is available in:
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import matplotlib.pyplot as plt

import sys
sys.path.append("..")
import func.cal as cal


device = "cuda:1" if torch.cuda.is_available() else "cpu"
l_x = 60                   # Data sequence length
lr_gnn = 0.0001                  # Gnn Learning rate
lr_res = 0.0001                   # Rnn Learning rate
weight_decay = 5e-4
epochs_gnn = 4000
epochs_res = 50
hidden_dim = 64
gnn_style = "ResGraphNet"     # The proposed Model (This variable is not recommended for modification)
num_layers = 1
save_fig = True                  # Whether Save picture
save_np = True
ratio_train = 0.5               # Proportion of training datasets
fig_size = (12, 10)
ts_name_all = ["HadCRUT5", "temp_month", "temp_year"]
ts_name_folder = "HadCRUT5"    # Name of the folder where the data resides
ts_name = "HadCRUT5_global"       # Name of the selected time series
iv = 1                          # sampling interval, used for plotting curves
way = "mean"                    # The style of plot curves of real data and predict results

x_address = osp.join("../datasets", ts_name_folder, ts_name + ".npy")
x = np.load(x_address)
num = x.shape[0]                # The length of time series

result_address = osp.join("../result", ts_name)
if not(osp.exists(result_address)):
    os.makedirs(result_address)

num_train = int(ratio_train * num)
data_train, data_test = x[:num_train], x[num_train:num]     # get training dataset and test dataset

# scale = MinMaxScaler()
# scale.fit(data_train)
# data_train = scale.transform(data_train)
# data_test = scale.transform(data_test)

max_ly = 20                 # Must be a multiple of 5
ly_all = np.arange(1, max_ly + 1)
r2_all = []
bins = 5
x_tick = np.arange(4, max_ly, bins)
x_label = x_tick + 1

for i in range(ly_all.shape[0]):
    l_y = ly_all[i]

    len_interp = l_y + 5
    data_test_ = np.array(
        data_test[:-l_y].tolist() + data_test[-len_interp - l_y:-l_y].tolist() + data_test[-l_y:].tolist())

    # Using Graph Neural network, prepare data information
    print("\nly={}: Running, ResGraphNet".format(l_y))
    x_train, y_train = cal.create_inout_sequences(data_train, l_x, l_y, style="arr")
    x_test, y_test = cal.create_inout_sequences(data_test_, l_x, l_y, style="arr")

    x_train = torch.from_numpy(x_train).float().to(device)
    x_test = torch.from_numpy(x_test).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)
    num_nodes = x_train.shape[0] + x_test.shape[0]

    x = torch.cat((x_train, x_test), dim=0)
    y = torch.cat((y_train, y_test), dim=0)

    adm = cal.path_graph(num_nodes)
    edge_index, edge_weight = cal.tran_adm_to_edge_index(adm)

    train_index = torch.arange(num_train, dtype=torch.long)
    test_index = torch.arange(num_train, num_nodes, dtype=torch.long)
    train_mask = cal.index_to_mask(train_index, num_nodes).to(device)
    test_mask = cal.index_to_mask(test_index, num_nodes).to(device)

    """
    Using ResGraphNet, predicting time series (The Proposed Network Model)
    """
    model = cal.GNNTime(l_x, hidden_dim, l_y, edge_weight, gnn_style, num_nodes).to(device)
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_gnn, weight_decay=weight_decay)
    edge_index = edge_index.to(device)

    print("Running, {}".format(gnn_style))
    for epoch in range(epochs_gnn):
        model.train()
        optimizer.zero_grad()
        output = model(x, edge_index)
        output_train, y_train = output[train_mask], y[train_mask]
        train_loss = criterion(output_train[:, -1], y_train[:, -1])
        train_loss.backward()
        optimizer.step()

        model.eval()
        output_test, y_test = output[test_mask][:-len_interp], y[test_mask][:-len_interp]
        test_loss = criterion(output_test[:, -1], y_test[:, -1])

        train_true = y_train.detach().cpu().numpy()[:, -1]
        train_predict = output_train.detach().cpu().numpy()[:, -1]
        test_true = y_test.detach().cpu().numpy()[:, -1]
        test_predict = output_test.detach().cpu().numpy()[:, -1]

        r2_train = cal.get_r2_score(train_predict, train_true, axis=1)
        r2_test = cal.get_r2_score(test_predict, test_true, axis=1)

        if (epoch + 1) % 100 == 0:
            print("Epoch: {:05d}  Loss_Train: {:.5f}  Loss_Test: {:.5f}  R2_Train: {:.7f}  R2_Test: {:.7f}".
                  format(epoch + 1, train_loss.item(), test_loss.item(), r2_train, r2_test))

    r2_train = cal.get_r2_score(train_predict, train_true, axis=1)
    r2_test = cal.get_r2_score(test_predict, test_true, axis=1)

    r2_all.append(r2_test)

plt.figure(figsize=fig_size)
plt.plot(r2_all)
plt.xticks(x_tick, x_label, fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel("$l_y$", fontsize=40)
plt.ylabel("$r^{test}$", fontsize=40)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_ResGraphNet_r_ly" + ".png"))
if save_np:
    r2_all = np.array(r2_all)
    np.save(osp.join(result_address, ts_name + "_ResGraphNet_r_ly" + ".npy"), r2_all)

print()
plt.show()
print()
