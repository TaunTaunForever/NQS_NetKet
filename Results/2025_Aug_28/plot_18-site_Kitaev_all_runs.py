import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches

samples_1 = pd.read_table("mean_energy_run_K=-1.0_18-site_4_blocks_24_width_1024_samples_2025-08-28_SRt.txt", header=None)
samples_1['x'] = np.arange(0,len(samples_1[0]), 1)

samples_2 = pd.read_table("mean_energy_run_K=-1.0_18-site_4_blocks_18_width_2048_samples_2025-08-30_minSR.txt", header=None)
samples_2['x'] = np.arange(0,len(samples_2[0]), 1)


samples_3 = pd.read_table("mean_energy_run_K=-1.0_18-site_4_blocks_18_width_4096_samples_2025-08-29_minSR.txt", header=None)
samples_3['x'] = np.arange(0,len(samples_3[0]), 1)

'''
samples_4 = pd.read_table("mean_energy_run_K=-1.0_24-site_4_layers_20_width_16384_samples_run-4_2025-06-16_minSR_run_4.txt", header=None)
samples_4['x'] = np.arange(0,len(samples_4[0]), 1)

samples_J2_6 = pd.read_table("mean_energy_run_J1=1.0_J2=0.2_36-site_4_GCNN_layers_8_width_1024_samples_2025-07-24.txt", header=None)
samples_J2_6['x'] = np.arange(0,len(samples_J2_6[0]), 1)
samples_J2_6[0] /= 4.0
'''




fig,axes = plt.subplots(1,1)
axes.set_xscale('log')
#plt.axhline(y = -25.6396983848311, color = 'r', linestyle = '-')
plt.plot(samples_1['x'][:100000], samples_1[0][:100000]/4, label="1024")
plt.plot(samples_2['x'][:100000], samples_2[0][:100000]/4, label="4096")
plt.plot(samples_3['x'][:100000], samples_3[0][:100000]/4, label="8192")
#plt.plot(samples_4['x'], samples_4[0], label="16384")
'''
plt.plot(samples_J2_5['x'], samples_J2_5[0], label="J2 = 0.17")
plt.plot(samples_J2_6['x'], samples_J2_6[0], label="J2 = 0.20")
'''

plt.axhline(y = -14.291502622129276/4, color = 'r', linestyle = '-')
plt.xlabel("Iterations")
plt.ylabel("Variational Ground State Energy")
plt.title("Ground State Energies for 18-site Kitaev Honeycomb Model")
plt.legend()
plt.show()

