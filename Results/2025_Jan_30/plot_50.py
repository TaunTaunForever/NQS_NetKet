import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

samples = pd.read_table("mean_energy_run_K=-1.0_50-site_4_blocks_6_width_32768_samples_2025-01-29.txt", header=None)
samples['x'] = np.arange(0,len(samples[0]), 1)

fig,axes = plt.subplots(1,1)
axes = plt.plot(samples['x'], samples[0])
#plt.show()
new_samples = samples[19000:]
fig,axes = plt.subplots(1,1)
axes = plt.plot(new_samples['x'], new_samples[0])

lower, median, upper = np.percentile(new_samples[0], [16,50,84], axis=0)

neg_error = median - lower
pos_error = upper - median
print('value = ', median, ' + ', pos_error, ' - ', neg_error)


#axes.ax





plt.grid()
plt.show()