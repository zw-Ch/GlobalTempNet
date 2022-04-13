import matplotlib.pyplot as plt
import numpy as np
import func.cal as cal
import os.path as osp


l1 = ["train", "test"]
l2 = ["true", "predict_ResGraphNet", "predict_ResNet", "predict_GraphSage", "predict_Cheb", "predict_GCN",
      "predict_GIN", "predict_Tran", "predict_Tag", "predict_Forest", "predict_Linear", "predict_SVR", "predict_SGD"]
i1 = 1
i2 = 1
x_name = l1[i1] + "_" + l2[i2]
x_address = osp.join("result/HadCRUT", x_name + ".npy")
x = np.load(x_address)

cal.plot_spiral(x, x_name)
plt.savefig(osp.join("result/HadCRUT", x_name + ".png"))

print()
plt.show()
print()
