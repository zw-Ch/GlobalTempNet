import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


address = "HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.csv"
df = pd.read_csv(address)
value = df.loc[:, "Anomaly (deg C)"].values
np.save("HadCRUT5_global.npy", value)
# plt.figure()
# plt.plot(value)

address = "HadCRUT.5.0.1.0.analysis.summary_series.northern_hemisphere.monthly.csv"
df = pd.read_csv(address)
value = df.loc[:, "Anomaly (deg C)"].values
np.save("HadCRUT5_northern.npy", value)
# plt.figure()
# plt.plot(value)

address = "HadCRUT.5.0.1.0.analysis.summary_series.southern_hemisphere.monthly.csv"
df = pd.read_csv(address)
value = df.loc[:, "Anomaly (deg C)"].values
np.save("HadCRUT5_southern.npy", value)
# plt.figure()
# plt.plot(value)


print()
plt.show()
print()
