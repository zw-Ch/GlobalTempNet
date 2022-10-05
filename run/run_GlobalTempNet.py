"""
Testing ResGraphNet
"""
import datetime
import numpy as np
import pandas as pd
import torch
import os
import os.path as osp
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append("..")
import func.process as pro
import func.net as net


device = "cuda:0" if torch.cuda.is_available() else "cpu"
# device = "cpu"
l_x = 120                   # Data sequence length
l_y = 2                    # Label sequence length
lr = 0.001                 # Learning rate
weight_decay = 0.001
epochs = 4000
hid_dim = 64
save_fig = False                  # Whether to save picture
save_txt = True                  # Whether to save txt
save_np = True                  # Whether to save np file
save_model = True               # Whether to save network model
ratio_train = 0.75               # Proportion of training datasets
fig_size = (12, 12)
random = False

root = "../datasets"
f_list = ['HadCRUT5', 'monthly', 'global_1850_01_2022_06.csv']             # https://www.metoffice.gov.uk/hadobs/hadcrut5/data/current/download.html
# f_list = ['HadCRUT5', 'monthly', 'northern_1850_01_2022_06.csv']
# f_list = ['HadCRUT5', 'monthly', 'southern_1850_01_2022_06.csv']
# f_list = ['ERSSTv4', 'monthly', '1854_01_2020_02.npy']    # https://psl.noaa.gov/data/gridded/data.noaa.ersst.v4.html
# f_list = ['ERSSTv3b', 'monthly', '1854_01_2020_02.npy']   # https://psl.noaa.gov/data/gridded/data.noaa.ersst.v3.html
# f_list = ['ERSSTv5', 'monthly', '1854_01_2022_08.npy']    # https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html
# f_list = ['ERA5', 'monthly', '1959_01_2021_12.npy']     # https://cds.climate.copernicus.eu/cdsapp#!/dataset/10.24381/cds.bd0915c6?tab=overview
# f_list = ['HadSST3', 'monthly', 'global_1850_01_2022_02.txt']   # https://www.metoffice.gov.uk/hadobs/hadsst3/data/download.html
# f_list = ['HadSST3', 'monthly', 'northern_1850_01_2022_02.txt']
# f_list = ['HadSST3', 'monthly', 'southern_1850_01_2022_02.txt']
# f_list = ['NCEP_DOE', 'monthly', '1979_01_2022_07.npy']     # https://psl.noaa.gov/data/gridded/data.ncep.reanalysis2.html
# f_list = ['NCEP_NCAR', 'monthly', '1948_01_2021_12.npy']    # https://www.psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
# f_list = ['OISSTv2', 'monthly', '1981_12_2022_08.npy']      # https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html
# f_list = ['NCEP_CIRES', 'monthly', '1871_01_2012_12.npy']   # https://psl.noaa.gov/data/gridded/data.20thC_ReanV2.html
# f_list = ['Solar', 'hourly', '2001_2012.csv']
# f_list = ['Exchange_Rate', 'daily', '2000_2020.csv']
# f_list = ['Traffic', 'hourly', '2018_2022.xlsx']
# f_list = ['PM25', 'hourly', '2010_2011.csv']
temp, time, f_name = pro.read(root, f_list)
temp_min, temp_max = np.min(temp), np.max(temp)

# plt.figure()
# plt.plot(temp)
# plt.show()

re_ad = osp.join("../result", "GlobalTempNet", "lx_{}_ly_{}".format(l_x, l_y))
if not(osp.exists(re_ad)):
    os.makedirs(re_ad)

prep_temp = StandardScaler()
temp = prep_temp.fit_transform(temp.reshape(-1, 1)).reshape(-1)

x, y = pro.get_xy(temp, l_x, l_y, style='arr')
x, y = torch.from_numpy(x).float().to(device), torch.from_numpy(y).float().to(device)
num = x.shape[0]
num_train = int(num * ratio_train)

# adm = pro.path_graph(num)
adm = pro.ts_un(num, 1)
edge_index, edge_weight = pro.tran_adm_to_edge_index(adm)

if not random:
    np.random.seed(2)
train_index, test_index = pro.get_train_or_test_idx(num - 1, num_train)
test_index.append(num - 1)
train_index = torch.LongTensor(train_index)
test_index = torch.LongTensor(test_index)
train_mask = pro.index_to_mask(train_index, num).to(device)
test_mask = pro.index_to_mask(test_index, num).to(device)

model = net.GlobalTempNet(l_x, hid_dim, l_y, edge_weight, num, 'gcn').to(device)
criterion = torch.nn.MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
edge_index = edge_index.to(device)

start_time = datetime.datetime.now()
print("Running, GlobalTempNet")
for epoch in range(epochs):
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

    train_true = y_train.detach().cpu().numpy()
    train_predict = output_train.detach().cpu().numpy()
    test_true = y_test.detach().cpu().numpy()
    test_predict = output_test.detach().cpu().numpy()

    train_true, train_predict, test_true, test_predict = pro.prep_inv(
        prep_temp, train_true, train_predict, test_true, test_predict)

    rmse_train = net.get_rmse(train_predict, train_true, axis=1)
    rmse_test = net.get_rmse(test_predict, test_true, axis=1)
    r2_train = net.get_r2_score(train_predict, train_true, axis=1)
    r2_test = net.get_r2_score(test_predict, test_true, axis=1)

    if (epoch + 1) % 100 == 0:
        print("Epoch: {:05d}  RMSE_Train: {:.8f}  RMSE_Test: {:.8f}  R2_Train: {:.7f}  R2_Test: {:.7f}".
              format(epoch + 1, rmse_train, rmse_test, r2_train, r2_test))

if save_np:
    np.save(osp.join(re_ad, "train_true_{}.npy".format(f_name)), train_true)
    np.save(osp.join(re_ad, "train_predict_{}.npy".format(f_name)), train_predict)
    np.save(osp.join(re_ad, "test_true_{}.npy".format(f_name)), test_true)
    np.save(osp.join(re_ad, "test_predict_{}.npy".format(f_name)), test_predict)
    np.save(osp.join(re_ad, "train_index_{}.npy".format(f_name)), train_index.detach().cpu().numpy())
    np.save(osp.join(re_ad, "test_index_{}.npy".format(f_name)), test_index.detach().cpu().numpy())
if save_model:
    torch.save(model.state_dict(), osp.join(re_ad, "GlobalTempNet_{}.pth".format(f_name)))

if save_txt:
    info_txt_address = osp.join(re_ad, "GlobalTempNet_result.txt")
    info_df_address = osp.join(re_ad, "GlobalTempNet_result.csv")
    f = open(info_txt_address, 'a')
    if osp.getsize(info_txt_address) == 0:
        f.write('rmse_test rmse_train lx ly hid_dim ratio_train lr weight_decay epochs f_name\n')
    f.write(str(round(rmse_test, 4)) + "  ")
    f.write(str(round(rmse_train, 4)) + "  ")
    f.write(str(l_x) + "  ")
    f.write(str(l_y) + "  ")
    f.write(str(hid_dim) + "  ")
    f.write(str(ratio_train) + "  ")
    f.write(str(lr) + "  ")
    f.write(str(weight_decay) + "  ")
    f.write(str(epochs) + "  ")
    f.write(str(f_name) + "  ")
    f.write("\n")
    f.close()

    info = np.loadtxt(info_txt_address, dtype=str)
    columns = info[0, :].tolist()
    values = info[1:, :]
    info_df = pd.DataFrame(values, columns=columns)
    info_df.to_csv(info_df_address)

train_predict_plot, train_true_plot = train_predict[:, -1], train_true[:, -1]
test_predict_plot, test_true_plot = test_predict[:, -1], test_true[:, -1]
rmse = net.get_rmse(test_predict_plot, test_true_plot, axis=1)
r2 = net.get_r2_score(test_predict_plot, test_true_plot, axis=1)
print("\nTest Results:\nRMSE: {:.8f}    R2: {:.8f}".format(rmse, r2))

x_range = np.arange(train_true_plot.shape[0])
y_range = np.arange(train_true_plot.shape[0], train_true_plot.shape[0] + test_true_plot.shape[0])
plt.figure(figsize=fig_size)
plt.plot(x_range, train_true_plot, label="Train True", alpha=0.5)
plt.plot(x_range, train_predict_plot, label="Train Predict", alpha=0.5)
plt.plot(y_range, test_true_plot, label="Test True", alpha=0.5)
plt.plot(y_range, test_predict_plot, label="Test Predict", alpha=0.5)
plt.legend(fontsize=30)

train_true = train_true[-1, -l_y:]
train_predict = train_predict[-1, -l_y:]
test_true = test_true[-1, -l_y:]
test_predict = test_predict[-1, -l_y:]
pro.spirals([test_true, test_predict], ['-', (0, (5, 5))], ['white', 'yellow'], [], (16, 16), 50, 50,
             f_list[0], 'blue', deg=np.min(np.concatenate((test_true, test_predict), axis=0)) - 0.5)
plt.figure()
plt.plot(test_true, label="true")
plt.plot(test_predict, label="predict")
plt.legend()

rmse = net.get_rmse(test_predict, test_true, axis=1)
r2 = net.get_r2_score(test_predict, test_true, axis=1)
print("\nFuture Prediction:\nRMSE: {:.8f}    R2: {:.8f}\nTest True:    {}\nTest Predict: {}".
      format(rmse, r2, np.round(test_true, 4), np.round(test_predict, 4)))

print()
plt.show()
print()
