import sys
import json
import os
import numpy as np
from math import log10, floor

# The last number of iterations over which we take the mean of our observables
ITERATIONS = 1000
#LOG_FILE = "./variational_states/out_GCNN_G=1.0_18-site_10_layers_8_width_1024_samples_2025-05-07_minSR.log"
LOG_FILE = "./out_ResGCNN_G=1.0_18-site_4_layers_18_width_1024_samples_2025-05-20_minSR.log"


data = json.load(open(LOG_FILE))

def round_to_1sf(x):
        return round(x, -int(floor(log10(abs(x)))))


# Reading in Energy and Variance of Energy
energy = []
variance = []
errors = []
taucorr = []

for en in data["Energy"]["Mean"]["real"]:
        energy.append(en)
for var in data["Energy"]["Variance"]:
        variance.append(var)
for err in data["Energy"]["Sigma"]:
        errors.append(err)
for autocor in data["Energy"]["TauCorr"]:
        taucorr.append(autocor)


print("Number of Iterations: ", len(energy))
print(energy[-20:])
print(variance[-20:])

en_est = np.mean(energy[-ITERATIONS:])
err_est = np.mean(errors[-ITERATIONS:])
var_est = np.mean(variance[-ITERATIONS:])
auto_correlation_est = np.mean(taucorr[-ITERATIONS:])

print("Energy Estimate: ", en_est)
print("Error Estimate: ", err_est)
print("Variance in Energy: ", var_est)
print("Autocorellation time Estimate: ", auto_correlation_est)

std = round_to_1sf(np.sqrt(var_est))
print("Standard Deviation: ", std)


#print("Final Figure: ", round(en_est, std), " +- ", std)

