import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


address = "HadCRUT.5.0.1.0.analysis.summary_series.global.monthly.csv"
df = pd.read_csv(address)
value_full = df.loc[:, "Anomaly (deg C)"].values
np.save("HadCRUT5_global_full.npy", value_full)

low_con = df.loc[:, "Lower confidence limit (2.5%)"].values
up_con = df.loc[:, "Upper confidence limit (97.5%)"].values

value = value_full[:-3]
np.save("HadCRUT5_global.npy", value)

plt.figure(figsize=(12, 12))
plt.plot(value_full, c="red", label="Observation", alpha=0.5)
# plt.plot(low_con, c="blue", label="Lower confidence limit", alpha=0.3)
x_tick = [0, 360, 720, 1080, 1440, 1800]
x_label = ["1850", "1880", "1910", "1940", "1970", "2000"]
plt.xticks(x_tick, x_label, fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Year", fontsize=30)
plt.ylabel("Anomaly ($^{\circ}$C)", fontsize=30)
# plt.legend(fontsize=30)
plt.savefig("HadCRUT5_global.png")


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
