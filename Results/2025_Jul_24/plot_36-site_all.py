import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches

samples_J2_1 = pd.read_table("mean_energy_run_J1=1.0_J2=0.02_36-site_4_GCNN_layers_8_width_1024_samples_2025-07-24.txt", header=None)
samples_J2_1['x'] = np.arange(0,len(samples_J2_1[0]), 1)
samples_J2_1[0] /= 4.0

samples_J2_2 = pd.read_table("mean_energy_run_J1=1.0_J2=0.05_36-site_4_GCNN_layers_8_width_1024_samples_2025-07-24.txt", header=None)
samples_J2_2['x'] = np.arange(0,len(samples_J2_2[0]), 1)
samples_J2_2[0] /= 4.0

samples_J2_3 = pd.read_table("mean_energy_run_J1=1.0_J2=0.07_36-site_4_GCNN_layers_8_width_1024_samples_2025-07-24.txt", header=None)
samples_J2_3['x'] = np.arange(0,len(samples_J2_3[0]), 1)
samples_J2_3[0] /= 4.0

samples_J2_4 = pd.read_table("mean_energy_run_J1=1.0_J2=0.15_36-site_4_GCNN_layers_8_width_1024_samples_2025-07-24.txt", header=None)
samples_J2_4['x'] = np.arange(0,len(samples_J2_4[0]), 1)
samples_J2_4[0] /= 4.0

samples_J2_5 = pd.read_table("mean_energy_run_J1=1.0_J2=0.17_36-site_4_GCNN_layers_8_width_1024_samples_2025-07-24.txt", header=None)
samples_J2_5['x'] = np.arange(0,len(samples_J2_5[0]), 1)
samples_J2_5[0] /= 4.0

samples_J2_6 = pd.read_table("mean_energy_run_J1=1.0_J2=0.2_36-site_4_GCNN_layers_8_width_1024_samples_2025-07-24.txt", header=None)
samples_J2_6['x'] = np.arange(0,len(samples_J2_6[0]), 1)
samples_J2_6[0] /= 4.0


mean_J2_1 = np.mean(samples_J2_1[0][-100:])
mean_J2_2 = np.mean(samples_J2_2[0][-100:])
mean_J2_3 = np.mean(samples_J2_3[0][-100:])
mean_J2_4 = np.mean(samples_J2_4[0][-100:])
mean_J2_5 = np.mean(samples_J2_5[0][-100:])
mean_J2_6 = np.mean(samples_J2_6[0][-100:])

std_J2_1 = np.std(samples_J2_1[0][-100:])
std_J2_2 = np.std(samples_J2_2[0][-100:])
std_J2_3 = np.std(samples_J2_3[0][-100:])
std_J2_4 = np.std(samples_J2_4[0][-100:])
std_J2_5 = np.std(samples_J2_5[0][-100:])
std_J2_6 = np.std(samples_J2_6[0][-100:])

print(mean_J2_1, " +- ", std_J2_1)
print(mean_J2_2, " +- ", std_J2_2)
print(mean_J2_3, " +- ", std_J2_3)
print(mean_J2_4, " +- ", std_J2_4)
print(mean_J2_5, " +- ", std_J2_5)
print(mean_J2_6, " +- ", std_J2_6)

J2_values = [0.02, 0.05, 0.07, 0.15, 0.17, 0.2]
mean_values_GCNN = [mean_J2_1, mean_J2_2, mean_J2_3, mean_J2_4, mean_J2_5, mean_J2_6]
std_GCNN = [std_J2_1, std_J2_2, std_J2_3, std_J2_4, std_J2_5, std_J2_6]
mean_values_ED = [-19.85241090879676, -19.41284536126786, -19.14994526250388, -18.36870574161985, -18.31693177415766, -18.34671856732309]


fig,axes = plt.subplots(1,2)
print(axes[0])
axes[0].set_xscale('log')
#plt.axhline(y = -25.6396983848311, color = 'r', linestyle = '-')
axes[0].plot(samples_J2_1['x'], samples_J2_1[0], label="J2 = 0.02")
axes[0].plot(samples_J2_2['x'], samples_J2_2[0], label="J2 = 0.05")
axes[0].plot(samples_J2_3['x'], samples_J2_3[0], label="J2 = 0.07")
axes[0].plot(samples_J2_4['x'], samples_J2_4[0], label="J2 = 0.15")
axes[0].plot(samples_J2_5['x'], samples_J2_5[0], label="J2 = 0.17")
axes[0].plot(samples_J2_6['x'], samples_J2_6[0], label="J2 = 0.20")
axes[0].set_xlabel("Iterations")
axes[0].set_ylabel("Variational Ground State Energy")
axes[0].set_title("Ground state optimization for 36-site J1-J2 Model")
axes[0].legend()


axes[1].scatter(J2_values, mean_values_ED, label="Exact Diagonalization")
axes[1].scatter(J2_values, mean_values_GCNN, label="GCNN")
axes[1].set_xlabel(r"$J_2$")
axes[1].set_ylabel("Ground State Energy")
axes[1].set_title("ED vs GCNN Ground State Energies for 36-site J1-J2 Model")
axes[1].legend()
plt.show()

