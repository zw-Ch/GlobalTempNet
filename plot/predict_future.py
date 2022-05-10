"""
Predict future temperature changes
"""
import numpy as np
import os
import os.path as osp
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append("..")
import func.cal as cal


device = "cuda:1" if torch.cuda.is_available() else "cpu"
l_x = 60
l_y = 12
num_nodes = 962
fig_size = (12, 12)
save_fig = True

x_address = osp.join("../datasets", "HadCRUT", "HadCRUT2.npy")
x = np.load(x_address)
adm = cal.path_graph(num_nodes)
edge_index, edge_weight = cal.tran_adm_to_edge_index(adm)

data = []
for i in range(num_nodes):
    x_one = x[-i-l_x:]
    data_one = x_one[:l_x].reshape(1, -1)
    if i == 0:
        data = data_one
    else:
        data = np.concatenate((data_one, data), axis=0)

result_address = osp.join("../result/HadCRUT2/ResGraphNet/ResGraphNet.pkl")
model = torch.load(result_address).to(device)

data = torch.from_numpy(data).float().to(device)
edge_index = edge_index.to(device)
out = model(data, edge_index)

result = out.detach().cpu().numpy()[-1, :]              # the predict result of
plt.figure(figsize=fig_size)
plt.plot(result)
plt.xticks(np.arange(12), np.arange(1, 13), fontsize=25)
plt.yticks(fontsize=25)
plt.xlabel("Month", fontsize=30)
plt.ylabel("Temperature", fontsize=30)
# plt.title("The Temperature Prediction in 2022", fontsize=30)
if save_fig:
    plt.savefig("../graph/HadCRUT2/predict_future.png")


print()
plt.show()
print()
