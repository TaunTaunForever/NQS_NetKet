import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

samples = pd.read_table("mean_energy_run_K=-1.0_18-site_4_ResGCNN_blocks_6_width_65536_samples_2025-01-30.txt", header=None)
samples['x'] = np.arange(0,len(samples[0]), 1)
#samples[0] -= -14.291502622129254
fig,axes = plt.subplots(1,1)
axes = plt.plot(samples['x'], samples[0])

fig.set_size_inches(10, 8)
plt.title("18-site Kitaev Honeycomb Model (ResGCNN)")
plt.xlabel("Iterations")
plt.ylabel("Variational Energy - ED Ground State Energy")
plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.show()

new_samples = samples[99000:]
fig,axes = plt.subplots(1,1)
axes = plt.plot(new_samples['x'], new_samples[0])

lower, median, upper = np.percentile(new_samples[0], [16,50,84], axis=0)

neg_error = median - lower
pos_error = upper - median
print('value = ', median, ' + ', pos_error, ' - ', neg_error)


plt.hist(samples[0], bins = 1000)
plt.show()

#axes.ax




