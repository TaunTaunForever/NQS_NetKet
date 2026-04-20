import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import scipy


def func_1(N, e_0, a, b):
    return e_0 - a*np.exp(-N/b)

def func_2(N, e_0, a, b, c):
    return e_0 - a/N - b/(N**2) - c/(N**3)

def func_3(N, e_0, a, b, alpha):
    return e_0 - a/(N**alpha) - b/(N**(2*alpha))


per_spin_energy = np.asarray([-0.3551, -0.3558,\
                              -0.3558, -0.3559,\
                              -0.35596, -0.35598,\
                              -0.35599])

num_sites = np.asarray([3, 6, 9, 12, 15, 18, 21])
#num_sites = 1/num_sites

# popt are the optimal values for the parameters so that the squared residuals 
# are minimized
# pcov are the estimated approximated covariance of popt
# curve_fit fits the x and y data to the function, f, which takes the first 
# variable as the independent variable
popt_1, pcov_1 = scipy.optimize.curve_fit(func_1, num_sites, per_spin_energy)
popt_2, pcov_2 = scipy.optimize.curve_fit(func_2, num_sites, per_spin_energy)
popt_3, pcov_3 = scipy.optimize.curve_fit(func_3, num_sites, per_spin_energy)

print(per_spin_energy)
print(num_sites)

x_range = np.arange(0.0, 24, 0.1)
#x_range = 1/x_range
fig = plt.gcf()
ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
fig.set_size_inches(7.5, 5)
plt.errorbar(num_sites, per_spin_energy, yerr=[0.0004, 0.0002, 0.0002, 0.0001, 0.0001, 0.00007, 0.00005], fmt='none')
plt.scatter(num_sites, per_spin_energy, s=20, marker='x')
#plt.plot(x_range, func_1(x_range, *popt_1), 'r-', label='fit 1: e_0=%5.3f, a=%5.3f, b=%5.3f' % tuple(popt_1))
#plt.plot(x_range, func_2(x_range, *popt_2), 'g--', label='fit 2: e_0=%5.3f, a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt_2))
#plt.plot(x_range, func_3(x_range, *popt_3), label='fit 3: e_0=%5.3f, a=%5.3f, b=%5.3f, alpha=%5.3f' % tuple(popt_3))




plt.xlabel("Number of Nodes Per Layer")
plt.ylabel("Energy Per Spin")
#plt.xlim(0, 1/min(num_sites))
plt.ylim(-0.3565, -0.354)
#plt.legend()
#plt.title("18-site GSE For FM Gamma Honeycomb Model \nusing Kernel Formalation of SGD+SR")
plt.show()

