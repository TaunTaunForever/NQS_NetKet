import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches

samples_J2_1 = pd.read_table("mean_energy_run_J1=1.0_J2=0.02_64-site_4_GCNN_layers_8_width_1024_samples_2025-07-26.txt", header=None)
samples_J2_1['x'] = np.arange(0,len(samples_J2_1[0]), 1)
samples_J2_1[0] /= 4.0

samples_J2_2 = pd.read_table("mean_energy_run_J1=1.0_J2=0.05_64-site_4_GCNN_layers_8_width_1024_samples_2025-07-26.txt", header=None)
samples_J2_2['x'] = np.arange(0,len(samples_J2_2[0]), 1)
samples_J2_2[0] /= 4.0

samples_J2_3 = pd.read_table("mean_energy_run_J1=1.0_J2=0.07_64-site_4_GCNN_layers_8_width_1024_samples_2025-07-26.txt", header=None)
samples_J2_3['x'] = np.arange(0,len(samples_J2_3[0]), 1)
samples_J2_3[0] /= 4.0

samples_J2_4 = pd.read_table("mean_energy_run_J1=1.0_J2=0.17_64-site_4_GCNN_layers_8_width_1024_samples_2025-07-27.txt", header=None)
samples_J2_4['x'] = np.arange(0,len(samples_J2_4[0]), 1)
samples_J2_4[0] /= 4.0




fig,axes = plt.subplots(1,1)
axes.set_xscale('log')
#plt.axhline(y = -25.6396983848311, color = 'r', linestyle = '-')
plt.plot(samples_J2_1['x'], samples_J2_1[0], label="J2 = 0.02")
plt.plot(samples_J2_2['x'], samples_J2_2[0], label="J2 = 0.05")
plt.plot(samples_J2_3['x'], samples_J2_3[0], label="J2 = 0.07")
plt.plot(samples_J2_4['x'], samples_J2_4[0], label="J2 = 0.17")
plt.xlabel("Iterations")
plt.ylabel("Variational Ground State Energy")
plt.title("Ground State Energies for 64-site J1-J2 Triangular Model")
plt.legend()
plt.show()

