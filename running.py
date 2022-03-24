"""
Using RNN, GNN and ML to predict the time series respectively
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import os.path as osp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from func.cal import path_graph, create_inout_sequences, tran_adm_to_edge_index, index_to_mask, GNNTime,\
    eval_on_features, get_rmse, plot_2curve, plot_distribute, get_r2_score, ResNet, MyData


device = "cuda:0" if torch.cuda.is_available() else "cpu"
x_length = 60                   # Data sequence length
y_length = 1                    # Label sequence length
lr_gnn = 0.0001                  # Gnn Learning rate
lr_res = 0.0001                   # Rnn Learning rate
weight_decay = 5e-4
epochs_gnn = 4000
epochs_res = 50
hidden_dim = 64
gnn_style_all = ["gcn", "che", "sage", "gin", "tran", "tag", "sage_res"]
gnn_style = "sage"               # GNN style
num_layers = 1
save_fig = True                  # If Save picture
save_txt = False
ratio_train = 0.5               # Proportion of training datasets
fig_size = (12, 12)
ts_name_all = ["seismic_trace", "electricity", "sales", "solar", "pm25", "traffic", "temperature", "WellData"]
ts_name_folder = "temperature"    # Name of the folder where the data resides
ts_name = "Afghanistan"       # Name of data
batch_size = 32

address = os.getcwd()
x_address = osp.join(address, "datasets", ts_name_folder, ts_name + ".npy")
x = np.load(x_address)
num = x.shape[0]

result_address = osp.join(address, "result", ts_name)
if not(osp.exists(result_address)):
    os.makedirs(result_address)

num_train = int(ratio_train * num)
data_train, data_test = x[:num_train], x[num_train:num]

# scale = MinMaxScaler()
# scale.fit(data_train)
# data_train = scale.transform(data_train)
# data_test = scale.transform(data_test)

"""
Using Graph Neural network, predicting time series
"""
print("\nRunning, GNN or ResGraphNet")
x_train, y_train = create_inout_sequences(data_train, x_length, y_length, style="arr")
x_test, y_test = create_inout_sequences(data_test, x_length, y_length, style="arr")

x_train = torch.from_numpy(x_train).float().to(device)
x_test = torch.from_numpy(x_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)
num_nodes = x_train.shape[0] + x_test.shape[0]

x = torch.cat((x_train, x_test), dim=0)
y = torch.cat((y_train, y_test), dim=0)

adm = path_graph(num_nodes)
edge_index, edge_weight = tran_adm_to_edge_index(adm)

train_index = torch.arange(num_train, dtype=torch.long)
test_index = torch.arange(num_train, num_nodes, dtype=torch.long)
train_mask = index_to_mask(train_index, num_nodes).to(device)
test_mask = index_to_mask(test_index, num_nodes).to(device)

model = GNNTime(x_length, hidden_dim, y_length, edge_weight, gnn_style, num_layers).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_gnn, weight_decay=weight_decay)
edge_index = edge_index.to(device)

print("Running, {}".format(gnn_style))
for epoch in range(epochs_gnn):
    model.train()
    optimizer.zero_grad()
    output = model(x, edge_index)
    output_train, y_train = output[train_mask], y[train_mask]
    train_loss = criterion(output_train, y_train)
    train_loss.backward()
    optimizer.step()

    model.eval()
    output_test, y_test = output[test_mask], y[test_mask]
    test_loss = criterion(output_test, y_test)

    r2_train = get_r2_score(output_train, y_train)
    r2_test = get_r2_score(output_test, y_test)

    if (epoch + 1) % 100 == 0:
        print("Epoch: {:05d}  Loss_Train: {:.5f}  Loss_Test: {:.5f}  R2_Train: {:.7f}  R2_Test: {:.7f}".
              format(epoch + 1, train_loss.item(), test_loss.item(), r2_train, r2_test))

train_true = y_train.detach().cpu().numpy()[:, 0]
train_predict = output_train.detach().cpu().numpy()[:, 0]
test_true = y_test.detach().cpu().numpy()[:, 0]
test_predict = output_test.detach().cpu().numpy()[:, 0]

rmse_train = get_rmse(train_true, train_predict)
rmse_test = get_rmse(test_true, test_predict)

r2_train_gnn = get_r2_score(train_true, train_predict)
r2_test_gnn = get_r2_score(test_true, test_predict)

e_gnn = test_true - test_predict
plot_distribute(e_gnn, 40, 4, x_name="e")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_" + gnn_style + "_error_distribution.png"))

plt.figure(figsize=fig_size)
x_lim_train = np.arange(num_train)
x_lim_test = np.arange(num_train, num_nodes)
plt.plot(x_lim_train, train_true, label="$y^{train}$", alpha=0.5)
plt.plot(x_lim_train, train_predict, label="$\hat{y}^{train}$", alpha=0.5)
plt.plot(x_lim_test, test_true, label="$y^{test}$", alpha=0.5)
plt.plot(x_lim_test, test_predict, label="$\hat{y}^{test}$", alpha=0.5)
plt.xlabel("Time", fontsize=30)
plt.ylabel("Value", fontsize=30)
plt.legend(fontsize=25)
# title = "{}, {}".format(ts_name, gnn_style)
# plt.title(title, fontsize=30)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_" + gnn_style + ".png"))
print("{}:  RMSE_Train={:.5f}  RMSE_Test={:.5f}  R2_Train={:.7f}  R2_Test={:.7f}".
      format(gnn_style, rmse_train, rmse_test, r2_train, r2_test))

"""
Using machine learning to predict time series
"""
print("\nRunning, Machine Learning")
x_train, y_train = create_inout_sequences(data_train, x_length, y_length, style="arr")         # 训练集序列
x_test, y_test = create_inout_sequences(data_test, x_length, y_length, style="arr")           # 测试集序列

forest = RandomForestRegressor()
linear = LinearRegression()
svr = SVR()
sgd = SGDRegressor()

rmse_train_forest, rmse_test_forest, r2_train_forest, r2_test_forest = eval_on_features(
    forest, x_train, y_train, x_test, y_test, ts_name + ", Random Forest", fig_size)
print("forest:  RMSE_Train: {:.5f}  RMSE_Test: {:.5f}  R2_Train: {:.6f}  R2_Test: {:.6f}".
      format(rmse_train_forest, rmse_test_forest, r2_train_forest, r2_test_forest))
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_Forest.png"))

rmse_train_linear, rmse_test_linear, r2_train_linear, r2_test_linear = eval_on_features(
    linear, x_train, y_train, x_test, y_test, ts_name + ", Linear Regression", fig_size)
print("linear:  RMSE_Train: {:.5f}  RMSE_Test: {:.5f}  R2_Train: {:.6f}  R2_Test: {:.6f}".
      format(rmse_train_linear, rmse_test_linear, r2_train_linear, r2_test_linear))
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_Linear.png"))

e_linear = test_true - test_predict
plot_distribute(e_linear, 40, 4, x_name="e")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_Linear_error_distribution.png"))

rmse_train_svr, rmse_test_svr, r2_train_svr, r2_test_svr = eval_on_features(
    svr, x_train, y_train, x_test, y_test, ts_name + ", Support Vector Regression", fig_size)
print("svr:  RMSE_Train: {:.5f}  RMSE_Test: {:.5f}  R2_Train: {:.6f}  R2_Test: {:.6f}".
      format(rmse_train_svr, rmse_test_svr, r2_train_svr, r2_test_svr))
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_SVR.png"))

rmse_train_sgd, rmse_test_sgd, r2_train_sgd, r2_test_sgd = eval_on_features(
    sgd, x_train, y_train, x_test, y_test, ts_name + "，Random Gradient Descent", fig_size)
print("sgd:  RMSE_Train: {:.5f}  RMSE_Test: {:.5f}  R2_Train: {:.6f}  R2_Test: {:.6f}".
      format(rmse_train_sgd, rmse_test_sgd, r2_train_sgd, r2_test_sgd))
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_RGD.png"))

"""
Using ResNet to predict time series
"""
print("\nRunning, ResNet")
x_train, y_train = create_inout_sequences(data_train, x_length, y_length, style="arr")
x_test, y_test = create_inout_sequences(data_test, x_length, y_length, style="arr")

train_dataset = MyData(x_train, y_train)
test_dataset = MyData(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = ResNet(hidden_dim, y_length, x_length).to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_res, weight_decay=weight_decay)

for epoch in range(epochs_res):
    loss_train_all, loss_test_all = 0, 0
    train_true, train_predict, test_true, test_predict = [], [], [], []
    for idx, (x_train, y_train) in enumerate(train_loader):
        x_train, y_train = x_train.to(device), y_train.to(device)
        optimizer.zero_grad()
        output_train = model(x_train)
        loss_train = criterion(output_train, y_train)
        loss_train.backward()
        optimizer.step()
        loss_train_all = loss_train_all + loss_train.item()

        train_predict_one = output_train.detach().cpu().numpy()[:, 0]
        train_true_one = y_train.detach().cpu().numpy()[:, 0]
        if idx == 0:
            train_true = train_true_one
            train_predict = train_predict_one
        else:
            train_true = np.concatenate((train_true, train_true_one), axis=0)
            train_predict = np.concatenate((train_predict, train_predict_one), axis=0)

    for idx, (x_test, y_test) in enumerate(test_loader):
        x_test, y_test = x_test.to(device), y_test.to(device)
        output_test = model(x_test)
        loss_test = criterion(output_test, y_test)
        loss_test_all = loss_test_all + loss_test.item()

        test_predict_one = output_test.detach().cpu().numpy()[:, 0]
        test_true_one = y_test.detach().cpu().numpy()[:, 0]
        if idx == 0:
            test_true = test_true_one
            test_predict = test_predict_one
        else:
            test_true = np.concatenate((test_true, test_true_one), axis=0)
            test_predict = np.concatenate((test_predict, test_predict_one), axis=0)

    rmse_train = get_rmse(train_predict, train_true)
    rmse_test = get_rmse(test_predict, test_true)
    r2_train = get_r2_score(train_predict, train_true)
    r2_test = get_r2_score(test_predict, test_true)
    print("Epoch: {:04d}  RMSE_Train: {:.7f}  RMSE_Test: {:.7f}  R2_Train: {:.7f}  R2_Test: {:.7f}".
          format(epoch, rmse_train, rmse_test, r2_train, r2_test))

rmse_train_res = get_rmse(train_predict, train_true)
rmse_test_res = get_rmse(test_predict, test_true)
r2_train_res = get_r2_score(train_predict, train_true)
r2_test_res = get_r2_score(test_predict, test_true)

e_res = test_true - test_predict
plot_distribute(e_res, 40, 4, x_name="e")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_ResNet_error_distribution.png"))

plt.figure(figsize=fig_size)
len_train, len_test = train_true.shape[0], test_true.shape[0]
train_lim, test_lim = np.arange(len_train), np.arange(len_train, len_train + len_test)
plt.plot(train_lim, train_true, label="$y^{train}$", alpha=0.5)
plt.plot(train_lim, train_predict, label="$\hat{y}^{train}$", alpha=0.5)
plt.plot(test_lim, test_true, label="$y^{test}$", alpha=0.5)
plt.plot(test_lim, test_predict, label="$\hat{y}^{test}$", alpha=0.5)
# plt.title("{}， ResNet".format(ts_name), fontsize=30)
plt.legend(fontsize=25)
plt.xlabel("Time", fontsize=30)
plt.ylabel("Value", fontsize=30)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_ResNet.png"))
print("ResNet:  RMSE_Train: {:.5f}  RMSE_Test: {:.5f}  R2_Train: {:.6f}  R2_Test: {:.6f}".
      format(rmse_train_res, rmse_test_res, r2_train_res, r2_test_res))

plt.show()
