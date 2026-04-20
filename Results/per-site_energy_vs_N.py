import matplotlib.pyplot as plt
import numpy as np
import scipy


def func_1(N, E_0, a, b):
    return E_0 - a*np.exp(-N/b)

def func_2(N, E_0, a, b, c):
    return E_0 - a/N - b/(N**2) - c/(N**3)

def func_3(N, E_0, a, b, alpha):
    return E_0 - a/(N**alpha) - b/(N**(2*alpha))

def func_4(N, E_0, a):
    return E_0 + a/N

per_spin_energy_all = np.asarray([-0.35598, -0.35707,\
                              -0.3532, -0.3515,\
                              -0.3510, -0.3519,\
                              -0.3490, -0.3502,\
                              -0.3443])

per_spin_energy_family_1 =  np.asarray([-0.35598, -0.3532, \
                                        -0.3515,  -0.3519, \
                                        -0.3502,  -0.3443])

num_sites_all = np.asarray([18, 24, 32, 50, 54, 72, 96, 98, 128])
num_sites_family_1 = np.asarray([18, 32, 50, 72, 98, 128])


popt_1, pcov_1 = scipy.optimize.curve_fit(func_1, num_sites_family_1, per_spin_energy_family_1, maxfev=2000)
popt_2, pcov_2 = scipy.optimize.curve_fit(func_2, num_sites_family_1, per_spin_energy_family_1, maxfev=5000)
popt_3, pcov_3 = scipy.optimize.curve_fit(func_3, num_sites_family_1, per_spin_energy_family_1, maxfev=5000)
popt_4, pcov_4 = scipy.optimize.curve_fit(func_4, num_sites_family_1, per_spin_energy_family_1, maxfev=5000)

x_range = np.arange(18, 10000, 0.1)

plt.errorbar((1/num_sites_all), per_spin_energy_all, yerr=[0.00005, 0.00009, 0.0001, 0.0003, \
                                               0.0001, 0.0002, 0.0003, 0.0002, \
                                               0.0004], fmt='x', markersize=5, ecolor='0')

fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Sites")
plt.ylabel("Energy Per Spin")
plt.ylim(-0.36, -0.34)
plt.xlim(0.0, 0.06)
plt.show()




# Second plot of [X, X] family of spin systems

# popt are the optimal values for the parameters so that the squared residuals 
# are minimized
# pcov are the estimated approximated covariance of popt
# curve_fit fits the x and y data to the function, f, which takes the first 
# variable as the independent variable

fig = plt.gcf()
axes = plt.gca()
plt.errorbar(1/num_sites_family_1, per_spin_energy_family_1, yerr=[0.00005, 0.0001, 0.0003, \
                                                                     0.0002, 0.0002, 0.0004], \
                                                                     fmt='x', markersize=5, ecolor='0')
plt.plot(1/x_range, func_1(x_range, *popt_1), 'r-', label=r'$E_0-ae^{-N/b}$')
plt.plot(1/x_range, func_2(x_range, *popt_2), 'g--', label=r'$E_0 - a/N -b/N^2 - c/N^3$')
plt.plot(1/x_range, func_3(x_range, *popt_3), label=r'$E_0 - a/N^{\alpha} - b/N^{2\alpha}$')
plt.plot(1/x_range, func_4(x_range, *popt_4), label=r'$E_0 -a/N$'+ r': $E_0=$%5.3f, $a=$%5.3f' % tuple(popt_4))
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Sites")
plt.ylabel("Energy Per Spin")
plt.xlim(0, 1/min(num_sites_family_1)+0.01)
plt.ylim(-0.358, -0.3425)
#plt.xlim(1/max(num_sites_family_1), 0.06)
plt.legend()
#plt.title("18-site GSE For FM Gamma Honeycomb Model \nusing Kernel Formalation of SGD+SR")
plt.show()

fig = plt.gcf()
axes = plt.gca()
plt.errorbar(num_sites_family_1, per_spin_energy_family_1, yerr=[0.00005, 0.0001, 0.0003, \
                                                                     0.0002, 0.0002, 0.0004], \
                                                                     fmt='x', markersize=5, ecolor='0')
plt.plot(x_range, func_1(x_range, *popt_1), label=r'$E_0-ae^{-N/b}$'+ r': $E_0=$%5.3f, $a=$%5.3f, $b=$%5.3f' % tuple(popt_1))
plt.plot(x_range, func_2(x_range, *popt_2), label=r'$E_0 - a/N -b/N^2 - c/N^3$' + r': $E_0=$%5.3f, $a=$%5.3f, $b=$%5.3f, $c=$%5.3f' % tuple(popt_2))
plt.plot(x_range, func_3(x_range, *popt_3), label=r'$E_0 - a/N^{\alpha} - b/N^{2\alpha}$' + r': $E_0=$%5.3f, $a=$%5.3f, $b=$%5.3f, $\alpha=$%5.3f' % tuple(popt_3))
plt.plot(x_range, func_4(x_range, *popt_4), label=r'$E_0 -a/N$'+ r': $E_0=$%5.3f, $a=$%5.3f' % tuple(popt_4))
plt.xlim(0, 130)
plt.ylim(-0.358, -0.3425)
plt.legend()
plt.show()

fig = plt.gcf()
axes = plt.gca()
plt.errorbar(1/num_sites_family_1, per_spin_energy_family_1, yerr=[0.00005, 0.0001, 0.0003, \
                                                                     0.0002, 0.0002, 0.0004], \
                                                                     fmt='x', markersize=5, ecolor='0')
plt.plot(1/x_range, func_1(x_range, *popt_1), 'r-', label=r'$E_0-ae^{-N/b}$'+ r': $E_0=$%5.3f, $a=$%5.3f, $b=$%5.3f' % tuple(popt_1))
plt.plot(1/x_range, func_2(x_range, *popt_2), 'g--', label=r'$E_0 - a/N -b/N^2 - c/N^3$' + r': $E_0=$%5.3f, $a=$%5.3f, $b=$%5.3f, $c=$%5.3f' % tuple(popt_2))
plt.plot(1/x_range, func_3(x_range, *popt_3), label=r'$E_0 - a/N^{\alpha} - b/N^{2\alpha}$' + r': $E_0=$%5.3f, $a=$%5.3f, $b=$%5.3f, $\alpha=$%5.3f' % tuple(popt_3))
plt.plot(1/x_range, func_4(x_range, *popt_4), label=r'$E_0 -a/N$'+ r': $E_0=$%5.3f, $a=$%5.3f' % tuple(popt_4))
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Sites")
plt.ylabel("Energy Per Spin")
plt.xlim(0, 1/min(num_sites_family_1)+0.01)
plt.ylim(-0.358, -0.3425)
#plt.xlim(1/max(num_sites_family_1), 0.06)
plt.legend()
#plt.title("18-site GSE For FM Gamma Honeycomb Model \nusing Kernel Formalation of SGD+SR")
plt.show()