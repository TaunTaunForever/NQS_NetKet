import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches

samples = pd.read_table("mean_energy_run_G=1.0_128-site_4_blocks_20_width_1024_samples_2025-05-22_minSR.txt", header=None)
samples['x'] = np.arange(0,len(samples[0]), 1)

samples_2 = pd.read_table("mean_energy_run_G=1.0_128-site_4_layers_20_width_1024_samples_run-2_2025-05-27_minSR.txt", header=None)
samples_2['x'] = np.arange(20000, 40000, 1)

samples_3 = pd.read_table("mean_energy_run_G=1.0_128-site_4_layers_20_width_1024_samples_run-3_2025-05-27_minSR.txt", header=None)
samples_3['x'] = np.arange(40000, 60000, 1)

samples_4 = pd.read_table("mean_energy_run_G=1.0_128-site_4_layers_20_width_1024_samples_run-4_2025-05-28_minSR.txt", header=None)
samples_4['x'] = np.arange(60000, 80000, 1)

samples_5 = pd.read_table("mean_energy_run_G=1.0_128-site_4_layers_20_width_1024_samples_run-5_2025-05-29_minSR.txt", header=None)
samples_5['x'] = np.arange(80000, 100000, 1)
#samples[0] -= (-4.76429835869187*4)
fig,axes = plt.subplots(1,1)
axes.set_xscale('log')
#plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.plot(samples['x'], samples[0])
plt.plot(samples_2['x'], samples_2[0])
plt.plot(samples_3['x'], samples_3[0])
plt.plot(samples_4['x'], samples_4[0])
plt.plot(samples_5['x'], samples_5[0])
plt.xlabel("Iterations")
plt.ylabel("Variational Ground State Energy")
#plt.title("50-site GSE For Gamma Heisenberg Honeycomb Model \nusing Kernel Formalation of SGD+SR")

new_samples = samples[-2000:]


x1, x2, y1, y2 = 99000, 100000, -70.45, -70.18  # subregion of the original image
axins = axes.inset_axes(
    [0.5, 0.5, 0.47, 0.47],
    xlim=(x1, x2), ylim=(y1, y2))
axins.plot(new_samples['x'], new_samples[0])

ticks = axins.get_xticks()

#print(type(ticks))
#print(ticks)
axins.set_xticks(ticks)
axins.set_xticklabels(axins.get_xticklabels(), rotation=-20, **{'fontsize': 6.})
#axins.set_xticklabels(axins.get_xticklabels(), rotation=-45, ha='left', **{'fontsize': 6.})
axins.set_yticklabels(axins.get_yticklabels(), ha='right', **{'fontsize': 6.})
axes.indicate_inset_zoom(axins, edgecolor="black")


'''
left, bottom, width, height = [0.55, 0.55, 0.3, 0.3]
ax2 = fig.add_axes([left, bottom, width, height])
ax2.plot(new_samples['x'], new_samples[0], color='green')


origin =  10**5-1000
'''
plt.show()

lower, median, upper = np.percentile(new_samples[0], [16,50,84], axis=0)

neg_error = median - lower
pos_error = upper - median
print('value = ', median, ' + ', pos_error, ' - ', neg_error)

