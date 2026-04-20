import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.patches as patches

samples = pd.read_table("mean_energy_run_G=1.0_18-site_4_layers_3_width_1024_samples_2025-05-17_minSR.txt", header=None)
samples['x'] = np.arange(0,len(samples[0]), 1)
#samples[0] -= (-4.76429835869187*4)
fig,axes = plt.subplots(1,1)
axes.set_xscale('log')
#plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.plot(samples['x'], samples[0])
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

