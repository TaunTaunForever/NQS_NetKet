import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

samples = pd.read_table("mean_energy_run_K=-1.0_G=1.0_24-site_6_blocks_6_width_16384_samples_2025-02-21.txt", header=None)
samples['x'] = np.arange(0,len(samples[0]), 1)
#samples[0] -= (-4.76429835869187*4)
fig,axes = plt.subplots(1,1)
axes.set_xscale('log')
#plt.axhline(y = 0, color = 'r', linestyle = '-')
axes = plt.plot(samples['x'], samples[0])
plt.show()


new_samples = samples[500:]
'''
fig,axes = plt.subplots(1,1)
axes = plt.plot(new_samples['x'], new_samples[0])
'''

lower, median, upper = np.percentile(new_samples[0], [16,50,84], axis=0)

neg_error = median - lower
pos_error = upper - median
print('value = ', median, ' + ', pos_error, ' - ', neg_error)

'''
fig,axes = plt.subplots(1,1)
#axes.set_xlim([-18.8, -18.4])

plt.hist(new_samples[0], bins = 100)
plt.show()
#axes.ax
#plt.grid()
#plt.show()
'''