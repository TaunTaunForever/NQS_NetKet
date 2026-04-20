import matplotlib.pyplot as plt
import numpy as np
import scipy

##########################################################################
# On-site Expectations <X>, <Y>, <Z>
X_expectations =  np.asarray([0.0009, 0.0006,\
                              0.0001, 0.0014, 0.0212, 0.0019, 0.0012,\
                            -0.0004, -0.0014])
X_expectations_error = np.asarray([0.0015, 0.0013, 0.001, 0.0012, 0.0018, \
                                   0.0027, 0.0019, 0.0018, 0.0016])

Y_expectations =  np.asarray([0.0005,  -0.0008,\
                               -0.0002, 0.0016, -0.0357, -0.0006, 0.001, \
                                -0.0047, -0.002])
Y_expectations_error = np.asarray([0.0014, 0.0012, 0.001, 0.0012, 0.0017, \
                                   0.0028, 0.0018, 0.0018, 0.0016])

Z_expectations =  np.asarray([-0.0027, 0.0019,\
                               0.0004,  -0.0027, 0.0151, -0.0012, -0.0015,\
                               0.0065, 0.005])
Z_expectations_error = np.asarray([0.002,  0.0017, 0.0013, 0.0014, 0.0021, \
                                   0.0036, 0.0021, 0.0025, 0.0025])

##########################################################################
# On-site Expectations <X>, <Y>, <Z> of systems of extent: [X,X]
X_expectations_family =  np.asarray([0.0009, 0.0001, 0.0014, 0.0019, -0.0004, -0.0014])
X_expectations_error_family = np.asarray([0.0015, 0.001, 0.0012, 0.0027, 0.0018, 0.0016])

Y_expectations_family =  np.asarray([0.0005, -0.0002, 0.0016, -0.0006, -0.0047, -0.002])
Y_expectations_error_family = np.asarray([0.0014, 0.001, 0.0012, 0.0028, 0.0018, 0.0016])

Z_expectations_family =  np.asarray([-0.0027, 0.0004,  -0.0027, -0.0012, 0.0065, 0.005])
Z_expectations_error_family = np.asarray([0.002, 0.0013, 0.0014, 0.0036, 0.0025, 0.0025])

num_sites_family = np.asarray([18, 32, 50, 72, 98, 128])
num_sites_family = 1/num_sites_family

##########################################################################
# XX, YY, and ZZ Correlations across Y-bond
XX_expectations_Y =  np.asarray([-0.00035, 0.00005,\
                               0.00351, 0.0058, -0.01439, -0.00281, -0.01169,
                                0.0068, -0.0012])
XX_expectations_error_Y = np.asarray([0.00050,  0.000042, 0.00037, 0.0004, \
                                      0.00056, 0.00093, 0.00057, 0.00057, 0.005])

YY_expectations_Y =  np.asarray([0.04843, 0.08359,\
                               0.08243, 0.06217, 0.10062, 0.12287, 0.10626,\
                                0.05771, 0.04117])
YY_expectations_error_Y = np.asarray([0.00048,  0.00066, 0.00051, 0.00043, \
                                      0.00054, 0.00092, 0.00056, 0.00057, 0.0049])

ZZ_expectations_Y =  np.asarray([0.00008, 0.00055,\
                               0.00289, -0.00364, -0.01706, -0.0031, -0.00306,
                               -0.00453, -0.00686])
ZZ_expectations_error_Y = np.asarray([0.0005,  0.000048, 0.00042, 0.00045, \
                                      0.00063, 0.00055, 0.00061, 0.00064, 0.00055])

##########################################################################
# XX, YY, and ZZ Correlations across X-bond
XX_expectations_X =  np.asarray([0.14541, 0.08470,\
                               0.08705, 0.1398, 0.08134, -0.0008, 0.14021, 
                               0.05840, 0.04167])
XX_expectations_error_X = np.asarray([0.00047,  0.00065, 0.00049, 0.00038, \
                                      0.0055, 0.0011, 0.00051, 0.00057, 0.00048])

YY_expectations_X =  np.asarray([-0.00024, 0.00097,\
                               0.00349, 0.00981, -0.00035, -0.0046, -0.00878, 
                               0.00617, -0.00126])
YY_expectations_error_X = np.asarray([0.00048,  0.0004, 0.000036, 0.00041, \
                                      0.00053, 0.0011, 0.0006, 0.00058, 0.00051])

ZZ_expectations_X =  np.asarray([0.00022, -0.000129,\
                               0.00421, 0.01083, 0.01211, 0.1259, -0.00544,
                               -0.00378, -0.00707])
ZZ_expectations_error_X = np.asarray([0.00047,  0.00052, 0.00042, 0.00045, \
                                      0.00062, 0.001, 0.00063, 0.00065, 0.00056])

##########################################################################
# XX, YY, and ZZ Correlations across Z-bond
XX_expectations_Z =  np.asarray([-0.00001, 0.00007,\
                               0.0034, 0.00194, 0.01286, -0.0008, 0.00635, \
                                -0.017, -0.02295])
XX_expectations_error_Z = np.asarray([0.00057,  0.00065, 0.00044, 0.00048, \
                                      0.00064, 0.0011, 0.00063, 0.00076, 0.00068])

YY_expectations_Z =  np.asarray([-0.00021, 0.00097,\
                               0.00278, -0.0046, -0.00351, -0.0046, 0.00567,\
                                -0.01679, -0.02325])
YY_expectations_error_Z = np.asarray([0.00059, 0.0004, 0.00044, 0.00045, \
                                      0.00060, 0.0011, 0.00063, 0.00076, 0.00051])

ZZ_expectations_Z =  np.asarray([0.09502, -0.00129,\
                               0.0835, 0.05483, 0.07222, 0.1259, -0.00068, \
                                0.13499, -0.00707])
ZZ_expectations_error_Z = np.asarray([0.00062,  0.00052, 0.00063, 0.00048, \
                                      0.00058, 0.001, 0.00055, 0.00056, 0.0006])

##########################################################################
# XY across Y-bond, X-bond and Z-bond
XY_expectations_Y =  np.asarray([0.00104, -0.00013,\
                               0.00733, 0.01959, 0.01626, 0.00099, -0.00801, \
                                -0.00842, -0.01615])
XY_expectations_error_Y = np.asarray([0.00046,  0.00042, 0.00042, 0.00041, \
                                      0.00058, 0.00053, 0.00042, 0.00056, 0.00037])

XY_expectations_X =  np.asarray([-0.00006, -0.00015,\
                               0.00623, 0.0175, -0.02518, -0.02518, -0.00825, \
                                -0.00825, -0.01594])
XY_expectations_error_X = np.asarray([0.00053,  0.00043, 0.00042, 0.00043, \
                                      0.00058, 0.00052, 0.00045, 0.00056, 0.00035])

XY_expectations_Z =  np.asarray([-0.11829, -0.11980,\
                               -0.11776, -0.10284, -0.11081, -0.13231, -0.08456, \
                                -0.13971, -0.14397])
XY_expectations_error_Z = np.asarray([0.00050,  0.00048, 0.00040, 0.00039, \
                                      0.00051, 0.00045, 0.00037, 0.00057, 0.00034])

##########################################################################
# XZ across Y-bond, X-bond and Z-bond
XZ_expectations_Y =  np.asarray([-0.09224, -0.11810,\
                               -0.11681, -0.10562, -0.12444, -0.13322, -0.12275, \
                                -0.10506, -0.09984])
XZ_expectations_error_Y = np.asarray([0.00047,  0.00045, 0.00042, 0.00041, \
                                      0.00050, 0.00045, 0.00039, 0.00055, 0.00037])

XZ_expectations_X =  np.asarray([0.0000, -0.00072,\
                               0.00608, -0.00087, -0.00182, -0.00642, 0.00378, \
                                -0.02952, -0.03956])
XZ_expectations_error_X = np.asarray([0.00055,  0.00048, 0.00050, 0.00045, \
                                      0.00060, 0.00060, 0.00044, 0.00071, 0.00044])

XZ_expectations_Z =  np.asarray([0.00053, 0.00042,\
                               0.00669, 0.019984, 0.0206, 0.00084, -0.01553, \
                                0.00343, -0.00893])
XZ_expectations_error_Z = np.asarray([0.00049,  0.00044, 0.00043, 0.00041, \
                                      0.00054, 0.0005, 0.00042, 0.00057, 0.00034])

##########################################################################
# YZ across Y-bond, X-bond and Z-bond
YZ_expectations_Y =  np.asarray([0.00027, 0.00076,\
                               0.00611, 0.00691, 0.01484, -0.00387, 0.00469, \
                                -0.02968, -0.03982])
YZ_expectations_error_Y = np.asarray([0.00051,  0.00047, 0.00050, 0.00047, \
                                      0.00061, 0.00057, 0.00045, 0.00068, 0.00045])

YZ_expectations_X =  np.asarray([-0.14536, -0.11914,\
                               -0.11897, -0.14355, -0.11546, -0.08660, -0.14184, \
                                -0.10541, -0.10012])
YZ_expectations_error_X = np.asarray([0.00046,  0.00042, 0.00035, 0.00036, \
                                      0.00054, 0.00043, 0.00037, 0.00059, 0.00035])

YZ_expectations_Z =  np.asarray([-0.00034, -0.00011,\
                               0.00673, 0.00901, -0.02291, 0.00366, -0.01425, \
                                0.00343, -0.00865])
YZ_expectations_error_Z = np.asarray([0.00047,  0.00045, 0.00042, 0.00044, \
                                      0.00058, 0.0005, 0.00045, 0.00059, 0.00045])

##########################################################################
# YX across Y-bond, X-bond and Z-bond
YX_expectations_Y =  np.asarray([0.0005, 0.00007,\
                               0.0077, 0.02007, 0.01557, 0.00108, -0.00687, \
                                -0.00831, -0.01593])
YX_expectations_error_Y = np.asarray([0.00045,  0.00044, 0.0004, 0.00043, \
                                      0.00059, 0.00053, 0.00042, 0.00055, 0.00034])

YX_expectations_X =  np.asarray([-0.00033, -0.00009,\
                               -0.00618, -0.00189, -0.02611, -0.00623, -0.00866, \
                                -0.00893, -0.01619])
YX_expectations_error_X = np.asarray([0.00053,  0.00044, 0.00042, 0.00043, \
                                      0.00060, 0.00053, 0.00045, 0.00056, 0.00034])

YX_expectations_Z =  np.asarray([-0.11847, -0.11995,\
                               -0.11741, -0.10218, -0.11045, -0.13285, -0.08441, \
                                -0.13977, -0.14407])
YX_expectations_error_Z = np.asarray([0.00051,  0.00049, 0.00038, 0.00038, \
                                      0.00052, 0.00045, 0.00037, 0.00057, 0.00033])

##########################################################################
# ZX across Y-bond, X-bond and Z-bond
ZX_expectations_Y =  np.asarray([-0.09209, -0.11798,\
                               -0.11661, -0.10554, -0.12507, -0.13280, -0.12238, \
                                -0.10545, -0.09998])
ZX_expectations_error_Y = np.asarray([0.00046,  0.00045, 0.00035, 0.00038, \
                                      0.00053, 0.00045, 0.00039, 0.00055, 0.00036])

ZX_expectations_X =  np.asarray([-0.0005, -0.00073,\
                                0.00563, -0.00015, -0.00181, -0.00685, 0.00383, \
                                -0.02938, -0.03970])
ZX_expectations_error_X = np.asarray([0.00056,  0.00048, 0.00049, 0.00047, \
                                      0.00061, 0.00061, 0.00047, 0.00069, 0.00045])

ZX_expectations_Z =  np.asarray([0.00039, 0.00011,\
                                0.00773, 0.01988, 0.00227, 0.00134, -0.0157, \
                                0.00403, -0.00659])
ZX_expectations_error_Z = np.asarray([0.00048,  0.00044, 0.00043, 0.00043, \
                                      0.00056, 0.00050, 0.00043, 0.00058, 0.00044])

##########################################################################
# ZY across Y-bond, X-bond and Z-bond
ZY_expectations_Y =  np.asarray([-0.00073, -0.0005,\
                                0.00664,  0.00646, 0.01463, -0.00375, 0.00522, \
                                -0.03006, -0.03982])
ZY_expectations_error_Y = np.asarray([0.00052,  0.00049, 0.00049, 0.00046, \
                                      0.00060, 0.00055, 0.00044, 0.00070, 0.00044])

ZY_expectations_X =  np.asarray([-0.14560, -0.11918,\
                                -0.11573, -0.14338, -0.11573, -0.08628, -0.14215, \
                                -0.10516, -0.10066])
ZY_expectations_error_X = np.asarray([0.00046,  0.00048, 0.00036, 0.00036, \
                                      0.00051, 0.00046, 0.00037, 0.00057, 0.00035])

ZY_expectations_Z =  np.asarray([0.00029, 0.00022,\
                                0.00672, 0.00864, 0.02372, -0.00435, -0.01381, \
                                0.00335, -0.00652])
ZY_expectations_error_Z = np.asarray([0.00048,  0.00043, 0.00041, 0.00043, \
                                      0.00061, 0.00051, 0.00043, 0.00059, 0.00043])

num_sites = np.asarray([18, 24, 32, 50, 54, 72, 96, 98, 128])
num_sites = 1/num_sites

#x_range = 1/x_range
#plt.scatter(num_sites, per_spin_energy, s=20)

##########################################################################
# Plots of On-site Expectations <X>, <Y>, <Z>
fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected X Per Spin")
plt.errorbar(num_sites, X_expectations, yerr=X_expectations_error, fmt='o')
plt.show()

fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected Y Per Spin")
plt.errorbar(num_sites, Y_expectations, yerr=Y_expectations_error, fmt='o')
plt.show()

fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected Z Per Spin")
plt.errorbar(num_sites, Z_expectations, yerr=Z_expectations_error, fmt='o')
plt.show()

##########################################################################
# Plots of On-site Expectations <X>, <Y>, <Z> [X, X] family
fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected X Per Spin")
plt.errorbar(num_sites_family, X_expectations_family, yerr=X_expectations_error_family, fmt='o')
plt.show()

fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Number of Spins")
plt.ylabel("Expected Y Per Spin")
plt.errorbar(num_sites_family, Y_expectations_family, yerr=Y_expectations_error_family, fmt='o')
plt.show()

fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected Z Per Spin")
plt.errorbar(num_sites_family, Z_expectations_family, yerr=Z_expectations_error_family, fmt='o')
#plt.scatter(num_sites_family, Z_expectations_family, s=20)
#plt.ylim(-0.01, 0.01)
plt.show()


##########################################################################
# Plots of XX, YY, and ZZ Correlations by bond-type
fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected XX Per Spin")
plt.errorbar(num_sites, XX_expectations_Y, yerr=XX_expectations_error_Y, fmt='o', label="Y-type")
plt.errorbar(num_sites, XX_expectations_X, yerr=XX_expectations_error_X, fmt='o', label="X-type")
plt.errorbar(num_sites, XX_expectations_Z, yerr=XX_expectations_error_Z, fmt='o', label="Z-type")
plt.legend()
plt.show()

fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected YY Per Spin")
plt.errorbar(num_sites, YY_expectations_Y, yerr=YY_expectations_error_Y, fmt='o', label="Y-type")
plt.errorbar(num_sites, YY_expectations_X, yerr=YY_expectations_error_X, fmt='o', label="X-type")
plt.errorbar(num_sites, YY_expectations_Z, yerr=YY_expectations_error_Z, fmt='o', label="Z-type")
plt.legend()
plt.show()

fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected ZZ Per Spin")
plt.errorbar(num_sites, ZZ_expectations_Y, yerr=ZZ_expectations_error_Y, fmt='o', label="Y-type")
plt.errorbar(num_sites, ZZ_expectations_X, yerr=ZZ_expectations_error_X, fmt='o', label="X-type")
plt.errorbar(num_sites, ZZ_expectations_Z, yerr=ZZ_expectations_error_Z, fmt='o', label="Z-type")
plt.legend()
plt.show()

##########################################################################
# Plots of XY, XZ, YX, YZ, ZX, ZY Correlations by bond-type
fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected XY Per Spin")
plt.errorbar(num_sites, XY_expectations_Y, yerr=XY_expectations_error_Y, fmt='o', label="Y-type")
plt.errorbar(num_sites, XY_expectations_X, yerr=XY_expectations_error_X, fmt='o', label="X-type")
plt.errorbar(num_sites, XY_expectations_Z, yerr=XY_expectations_error_Z, fmt='o', label="Z-type")
plt.legend()
plt.show()

fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected XZ Per Spin")
plt.errorbar(num_sites, XZ_expectations_Y, yerr=XZ_expectations_error_Y, fmt='o', label="Y-type")
plt.errorbar(num_sites, XZ_expectations_X, yerr=XZ_expectations_error_X, fmt='o', label="X-type")
plt.errorbar(num_sites, XZ_expectations_Z, yerr=XZ_expectations_error_Z, fmt='o', label="Z-type")
plt.legend()
plt.show()

fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected YZ Per Spin")
plt.errorbar(num_sites, YZ_expectations_Y, yerr=YZ_expectations_error_Y, fmt='o', label="Y-type")
plt.errorbar(num_sites, YZ_expectations_X, yerr=YZ_expectations_error_X, fmt='o', label="X-type")
plt.errorbar(num_sites, YZ_expectations_Z, yerr=YZ_expectations_error_Z, fmt='o', label="Z-type")
plt.legend()
plt.show()

fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected YX Per Spin")
plt.errorbar(num_sites, YX_expectations_Y, yerr=YX_expectations_error_Y, fmt='o', label="Y-type")
plt.errorbar(num_sites, YX_expectations_X, yerr=YX_expectations_error_X, fmt='o', label="X-type")
plt.errorbar(num_sites, YX_expectations_Z, yerr=YX_expectations_error_Z, fmt='o', label="Z-type")
plt.legend()
plt.show()

fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected ZX Per Spin")
plt.errorbar(num_sites, ZX_expectations_Y, yerr=ZX_expectations_error_Y, fmt='o', label="Y-type")
plt.errorbar(num_sites, ZX_expectations_X, yerr=ZX_expectations_error_X, fmt='o', label="X-type")
plt.errorbar(num_sites, ZX_expectations_Z, yerr=ZX_expectations_error_Z, fmt='o', label="Z-type")
plt.legend()
plt.show()

fig = plt.gcf()
fig.set_size_inches(7.5, 5)
plt.xlabel("Inverse Number of Spins")
plt.ylabel("Expected ZY Per Spin")
plt.errorbar(num_sites, ZY_expectations_Y, yerr=ZY_expectations_error_Y, fmt='o', label="Y-type")
plt.errorbar(num_sites, ZY_expectations_X, yerr=ZY_expectations_error_X, fmt='o', label="X-type")
plt.errorbar(num_sites, ZY_expectations_Z, yerr=ZY_expectations_error_Z, fmt='o', label="Z-type")
plt.legend()
plt.show()