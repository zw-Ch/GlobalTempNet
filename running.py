"""
Using ResGraphNet, GNN and ML to predict the time series respectively
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
import func.cal as cal
from func.cal import sam


device = "cuda:0" if torch.cuda.is_available() else "cpu"
l_x = 60                   # Data sequence length
l_y = 10                    # Label sequence length
lr_gnn = 0.0001                  # Gnn Learning rate
lr_res = 0.0001                   # Rnn Learning rate
weight_decay = 5e-4
epochs_gnn = 8000
epochs_res = 50
hidden_dim = 64
gnn_style_all = ["GCN", "Cheb", "GraphSage", "GIN", "Tran", "Tag", "ResGraphNet"]
gnn_style_1 = "ResGraphNet"     # The proposed Model (This variable is not recommended for modification)
gnn_style_2 = "GraphSage"       # The traditional GNN (This variable can be modified for comparison)
num_layers = 1
save_fig = True                  # Whether Save picture
save_txt = True                  # Whether Save txt
ratio_train = 0.5               # Proportion of training datasets
fig_size = (16, 12)
ts_name_all = ["climate", "electricity", "sales", "solar", "pm25", "traffic", "temperature", "noaa"]
ts_name_folder = "electricity"    # Name of the folder where the data resides
ts_name = "electricity_sam"       # Name of the selected time series
batch_size = 64
iv = 4                          # sampling interval, used for plotting curves
way = "mean"                    # The style of plot curves of real data and predict results

address = os.getcwd()
x_address = osp.join(address, "datasets", ts_name_folder, ts_name + ".npy")
x = np.load(x_address)
num = x.shape[0]                # The length of time series

result_address = osp.join(address, "result", ts_name)
if not(osp.exists(result_address)):
    os.makedirs(result_address)

num_train = int(ratio_train * num)
data_train, data_test = x[:num_train], x[num_train:num]     # get training dataset and test dataset

# scale = MinMaxScaler()
# scale.fit(data_train)
# data_train = scale.transform(data_train)
# data_test = scale.transform(data_test)


# Using Graph Neural network, prepare data information
print("\nRunning, GNN or ResGraphNet")
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

train_index = torch.arange(num_train, dtype=torch.long)
test_index = torch.arange(num_train, num_nodes, dtype=torch.long)
train_mask = cal.index_to_mask(train_index, num_nodes).to(device)
test_mask = cal.index_to_mask(test_index, num_nodes).to(device)

"""
Using ResGraphNet, predicting time series (The Proposed Network Model)
"""
model_1 = cal.GNNTime(l_x, hidden_dim, l_y, edge_weight, gnn_style_1, num_layers).to(device)
criterion_1 = torch.nn.MSELoss().to(device)
optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=lr_gnn, weight_decay=weight_decay)
edge_index_1 = edge_index.to(device)

print("Running, {}".format(gnn_style_1))
for epoch in range(epochs_gnn):
    model_1.train()
    optimizer_1.zero_grad()
    output = model_1(x, edge_index_1)
    output_train, y_train = output[train_mask], y[train_mask]
    train_loss = criterion_1(output_train, y_train)
    train_loss.backward()
    optimizer_1.step()

    model_1.eval()
    output_test, y_test = output[test_mask], y[test_mask]
    test_loss = criterion_1(output_test, y_test)

    r2_train = cal.get_r2_score(output_train, y_train)
    r2_test = cal.get_r2_score(output_test, y_test)

    if (epoch + 1) % 100 == 0:
        print("Epoch: {:05d}  Loss_Train: {:.5f}  Loss_Test: {:.5f}  R2_Train: {:.7f}  R2_Test: {:.7f}".
              format(epoch + 1, train_loss.item(), test_loss.item(), r2_train, r2_test))

train_true = y_train.detach().cpu().numpy()
train_predict = output_train.detach().cpu().numpy()
test_true = y_test.detach().cpu().numpy()
test_predict = output_test.detach().cpu().numpy()

rmse_train_gnn_1 = cal.get_rmse(train_predict, train_true)
rmse_test_gnn_1 = cal.get_rmse(test_predict, test_true)

r2_train_gnn_1 = cal.get_r2_score(train_predict, train_true)
r2_test_gnn_1 = cal.get_r2_score(test_predict, test_true)

e_gnn = test_true[:, 0] - test_predict[:, 0]
cal.plot_distribute(e_gnn, 40, 4, x_name="e")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_" + gnn_style_1 + "_error_distribution.png"))

plt.figure(figsize=fig_size)
length_train = train_true[:, 0].shape[0]
length_test = test_true[:, 0].shape[0]
lim_train = np.arange(length_train // iv)
lim_test = np.arange(length_train // iv, length_train // iv + length_test // iv)
plt.plot(lim_train, sam(train_true[:, 0], iv, way), label="$y^{train}$", alpha=0.5)
plt.plot(lim_train, sam(train_predict[:, 0], iv, way), label="$\hat{y}^{train}$", alpha=0.5)
plt.plot(lim_test, sam(test_true[:, 0], iv, way), label="$y^{test}$", alpha=0.5)
plt.plot(lim_test, sam(test_predict[:, 0], iv, way), label="$\hat{y}^{test}$", alpha=0.5)
plt.xlabel("Time", fontsize=30)
plt.ylabel("Value", fontsize=30)
plt.legend(fontsize=25)
# title = "{}, {}".format(ts_name, gnn_style)
# plt.title(title, fontsize=30)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_" + gnn_style_1 + ".png"))
print("{}: RMSE_Train={:.5f}  RMSE_Test={:.5f}  R2_Train={:.7f}  R2_Test={:.7f}".
      format(gnn_style_1, rmse_train_gnn_1, rmse_test_gnn_1, r2_train_gnn_1, r2_test_gnn_1))

"""
Using GraphSage, predicting time series (The traditional GNN Model)
"""
model_2 = cal.GNNTime(l_x, hidden_dim, l_y, edge_weight, gnn_style_2, num_layers).to(device)
criterion_2 = torch.nn.MSELoss().to(device)
optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=lr_gnn, weight_decay=weight_decay)
edge_index_2 = edge_index.to(device)

print("\nRunning, {}".format(gnn_style_2))
for epoch in range(epochs_gnn):
    model_2.train()
    optimizer_2.zero_grad()
    output = model_2(x, edge_index_2)
    output_train, y_train = output[train_mask], y[train_mask]
    train_loss = criterion_2(output_train, y_train)
    train_loss.backward()
    optimizer_2.step()

    model_2.eval()
    output_test, y_test = output[test_mask], y[test_mask]
    test_loss = criterion_2(output_test, y_test)

    r2_train = cal.get_r2_score(output_train, y_train)
    r2_test = cal.get_r2_score(output_test, y_test)

    if (epoch + 1) % 100 == 0:
        print("Epoch: {:05d}  Loss_Train: {:.5f}  Loss_Test: {:.5f}  R2_Train: {:.7f}  R2_Test: {:.7f}".
              format(epoch + 1, train_loss.item(), test_loss.item(), r2_train, r2_test))

train_true = y_train.detach().cpu().numpy()
train_predict = output_train.detach().cpu().numpy()
test_true = y_test.detach().cpu().numpy()
test_predict = output_test.detach().cpu().numpy()

rmse_train_gnn_2 = cal.get_rmse(train_predict, train_true)
rmse_test_gnn_2 = cal.get_rmse(test_predict, test_true)

r2_train_gnn_2 = cal.get_r2_score(train_predict, train_true)
r2_test_gnn_2 = cal.get_r2_score(test_predict, test_true)

e_gnn = test_true[:, 0] - test_predict[:, 0]
cal.plot_distribute(e_gnn, 40, 4, x_name="e")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_" + gnn_style_2 + "_error_distribution.png"))

plt.figure(figsize=fig_size)
length_train = train_true[:, 0].shape[0]
length_test = test_true[:, 0].shape[0]
lim_train = np.arange(length_train // iv)
lim_test = np.arange(length_train // iv, length_train // iv + length_test // iv)
plt.plot(lim_train, sam(train_true[:, 0], iv, way), label="$y^{train}$", alpha=0.5)
plt.plot(lim_train, sam(train_predict[:, 0], iv, way), label="$\hat{y}^{train}$", alpha=0.5)
plt.plot(lim_test, sam(test_true[:, 0], iv, way), label="$y^{test}$", alpha=0.5)
plt.plot(lim_test, sam(test_predict[:, 0], iv, way), label="$\hat{y}^{test}$", alpha=0.5)
plt.xlabel("Time", fontsize=30)
plt.ylabel("Value", fontsize=30)
plt.legend(fontsize=25)
# title = "{}, {}".format(ts_name, gnn_style)
# plt.title(title, fontsize=30)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_" + gnn_style_2 + ".png"))
print("{}:  RMSE_Train={:.5f}  RMSE_Test={:.5f}  R2_Train={:.7f}  R2_Test={:.7f}".
      format(gnn_style_2, rmse_train_gnn_2, rmse_test_gnn_2, r2_train_gnn_2, r2_test_gnn_2))

"""
Using machine learning to predict time series
"""
print("\nRunning, Machine Learning")
x_train, y_train = cal.create_inout_sequences(data_train, l_x, l_y, style="arr")         # 训练集序列
x_test, y_test = cal.create_inout_sequences(data_test, l_x, l_y, style="arr")           # 测试集序列

forest = RandomForestRegressor()
linear = LinearRegression()
svr = SVR()
sgd = SGDRegressor()

rmse_train_forest, rmse_test_forest, r2_train_forest, r2_test_forest = cal.eval_on_features(
    forest, x_train, y_train, x_test, y_test, ts_name + ", Random Forest", iv, way, fig_size)
print("forest:  RMSE_Train: {:.5f}  RMSE_Test: {:.5f}  R2_Train: {:.6f}  R2_Test: {:.6f}".
      format(rmse_train_forest, rmse_test_forest, r2_train_forest, r2_test_forest))
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_Forest.png"))

rmse_train_linear, rmse_test_linear, r2_train_linear, r2_test_linear = cal.eval_on_features(
    linear, x_train, y_train, x_test, y_test, ts_name + ", Linear Regression", iv, way, fig_size)
print("linear:  RMSE_Train: {:.5f}  RMSE_Test: {:.5f}  R2_Train: {:.6f}  R2_Test: {:.6f}".
      format(rmse_train_linear, rmse_test_linear, r2_train_linear, r2_test_linear))
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_Linear.png"))

e_linear = test_true[:, 0] - test_predict[:, 0]
cal.plot_distribute(e_linear, 40, 4, x_name="e")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_Linear_error_distribution.png"))

rmse_train_svr, rmse_test_svr, r2_train_svr, r2_test_svr = cal.eval_on_features(
    svr, x_train, y_train, x_test, y_test, ts_name + ", Support Vector Regression", iv, way, fig_size)
print("svr:  RMSE_Train: {:.5f}  RMSE_Test: {:.5f}  R2_Train: {:.6f}  R2_Test: {:.6f}".
      format(rmse_train_svr, rmse_test_svr, r2_train_svr, r2_test_svr))
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_SVR.png"))

rmse_train_sgd, rmse_test_sgd, r2_train_sgd, r2_test_sgd = cal.eval_on_features(
    sgd, x_train, y_train, x_test, y_test, ts_name + "，Random Gradient Descent", iv, way, fig_size)
print("sgd:  RMSE_Train: {:.5f}  RMSE_Test: {:.5f}  R2_Train: {:.6f}  R2_Test: {:.6f}".
      format(rmse_train_sgd, rmse_test_sgd, r2_train_sgd, r2_test_sgd))
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_RGD.png"))

"""
Using ResNet to predict time series
"""
print("\nRunning, ResNet")
x_train, y_train = cal.create_inout_sequences(data_train, l_x, l_y, style="arr")
x_test, y_test = cal.create_inout_sequences(data_test, l_x, l_y, style="arr")

train_dataset = cal.MyData(x_train, y_train)
test_dataset = cal.MyData(x_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = cal.ResNet(hidden_dim, l_y, l_x).to(device)
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

        train_predict_one = output_train.detach().cpu().numpy()
        train_true_one = y_train.detach().cpu().numpy()
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

        test_predict_one = output_test.detach().cpu().numpy()
        test_true_one = y_test.detach().cpu().numpy()
        if idx == 0:
            test_true = test_true_one
            test_predict = test_predict_one
        else:
            test_true = np.concatenate((test_true, test_true_one), axis=0)
            test_predict = np.concatenate((test_predict, test_predict_one), axis=0)

    rmse_train = cal.get_rmse(train_predict, train_true)
    rmse_test = cal.get_rmse(test_predict, test_true)
    r2_train = cal.get_r2_score(train_predict, train_true)
    r2_test = cal.get_r2_score(test_predict, test_true)
    print("Epoch: {:04d}  RMSE_Train: {:.7f}  RMSE_Test: {:.7f}  R2_Train: {:.7f}  R2_Test: {:.7f}".
          format(epoch, rmse_train, rmse_test, r2_train, r2_test))

rmse_train_res = cal.get_rmse(train_predict, train_true)
rmse_test_res = cal.get_rmse(test_predict, test_true)
r2_train_res = cal.get_r2_score(train_predict, train_true)
r2_test_res = cal.get_r2_score(test_predict, test_true)

e_res = test_true[:, 0] - test_predict[:, 0]
cal.plot_distribute(e_res, 40, 4, x_name="e")
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_ResNet_error_distribution.png"))

plt.figure(figsize=fig_size)
length_train = train_true[:, 0].shape[0]
length_test = test_true[:, 0].shape[0]
lim_train = np.arange(length_train // iv)
lim_test = np.arange(length_train // iv, length_train // iv + length_test // iv)
plt.plot(lim_train, sam(train_true[:, 0], iv), label="$y^{train}$", alpha=0.5)
plt.plot(lim_train, sam(train_predict[:, 0], iv), label="$\hat{y}^{train}$", alpha=0.5)
plt.plot(lim_test, sam(test_true[:, 0], iv), label="$y^{test}$", alpha=0.5)
plt.plot(lim_test, sam(test_predict[:, 0], iv), label="$\hat{y}^{test}$", alpha=0.5)
# plt.title("{}， ResNet".format(ts_name), fontsize=30)
plt.legend(fontsize=25)
plt.xlabel("Time", fontsize=30)
plt.ylabel("Value", fontsize=30)
if save_fig:
    plt.savefig(osp.join(result_address, ts_name + "_ResNet.png"))
print("ResNet:  RMSE_Train: {:.5f}  RMSE_Test: {:.5f}  R2_Train: {:.6f}  R2_Test: {:.6f}".
      format(rmse_train_res, rmse_test_res, r2_train_res, r2_test_res))

"""
将各个模型输出结果，追加写入文件
"""
if save_txt:
    info_txt_address = osp.join(result_address, "../result.txt")  # 保存参数信息的txt文件地址
    info_df_address = osp.join(result_address, "../result.csv")  # 保存参数信息的csv文件地址
    f = open(info_txt_address, 'a')
    if osp.getsize(info_txt_address) == 0:          # 若此前未曾运行，在文本第一行加入各个特征的名称
        f.write("ts_name gnn_style_1 r2_test_gnn_1 gnn_style_2 r2_test_gnn_2 r2_test_res r2_test_forest r2_test_linear "
                "r2_test_svr r2_test_sgd l_x l_y hidden_dim batch_size num_layers epochs_gnn lr_gnn epochs_res lr_res\n")
    f.write(str(ts_name) + "\t")
    f.write(str(gnn_style_1) + "\t")
    f.write(str(r2_test_gnn_1) + "\t")
    f.write(str(gnn_style_2) + "\t")
    f.write(str(r2_test_gnn_2) + "\t")
    f.write(str(r2_test_res) + "\t")
    f.write(str(r2_test_forest) + "\t")
    f.write(str(r2_test_linear) + "\t")
    f.write(str(r2_test_svr) + "\t")
    f.write(str(r2_test_sgd) + "\t")
    f.write(str(l_x) + "\t")
    f.write(str(l_y) + "\t")
    f.write(str(hidden_dim) + "\t")
    f.write(str(batch_size) + "\t")
    f.write(str(num_layers) + "\t")
    f.write(str(epochs_gnn) + "\t")
    f.write(str(lr_gnn) + "\t")
    f.write(str(epochs_res) + "\t")
    f.write(str(lr_res) + "\t")

    f.write("\n")                           # 准备下一次运行
    f.close()                               # 关闭文件

    # 保存为dataframe格式
    info = np.loadtxt(info_txt_address, dtype=str)
    columns = info[0, :].tolist()           # 特征的名称，作为列的名称
    values = info[1:, :]                    # 特征的取值
    info_df = pd.DataFrame(values, columns=columns)
    info_df.to_csv(info_df_address)

print()
# plt.show()
print()
