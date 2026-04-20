import sys
import json
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
from datetime import date    
import netket as nk
import numpy as np
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax,sigmaz, sigmay
from scipy.sparse.linalg import eigsh
import tracemalloc
import netket.experimental as nkx
import flax
import expectations

PREV_VARIABLES = "out_GCNN_G=1.0_128-site_4_layers_20_width_1024_samples_run-5_2025-05-29_minSR"
NUM_BLOCKS = 4
WIDTH = 20
NUM_SAMPLES = 2**15
CHUNK_SIZE = 1024
NUM_SITES = 128

#################################################################################
# Defining Kitaev Honeycomb Graph and Finding Symmetries
#################################################################################
print("Defining {}-site Honeycomb lattice from scratch".format(NUM_SITES))
print("___________________________________________________")
if   NUM_SITES == 4:
        graph = nk.graph.KitaevHoneycomb(extent=[2, 1], pbc=True)
        EXTENT_STR = "[2,1]"

elif NUM_SITES == 8:
        graph = nk.graph.KitaevHoneycomb(extent=[2, 2], pbc=True)
        EXTENT_STR = "[2,2]"

elif NUM_SITES == 12:
        graph = nk.graph.KitaevHoneycomb(extent=[2, 3], pbc=True)
        EXTENT_STR = "[2,3]"

elif NUM_SITES == 18:
        graph = nk.graph.KitaevHoneycomb(extent=[3, 3], pbc=True)
        EXTENT_STR = "[3,3]"

elif NUM_SITES == 24:
        edges = [(0, 1, 2), (0, 21, 1), (0, 3, 0), (1, 4, 1), (1, 18, 0), (2, 23, 1),
                 (2, 3, 2), (2, 6, 0), (3, 7, 1), (4, 8, 0), (4, 5, 2), (5, 22, 0),
                 (5, 9, 1), (6, 9, 2), (6, 10, 1), (7, 11, 0), (7, 8, 2), (8, 12, 1),
                 (9, 13, 0), (10, 11, 2), (10, 14, 0), (11, 15, 1), (12, 16, 0), (12, 13, 2),
                 (13, 17, 1), (14, 17, 2), (14, 18, 1), (15, 19, 0), (15, 16, 2), (16, 20, 1),
                 (17, 21, 0), (18, 19, 2), (19, 22, 1), (20, 23, 0), (20, 21, 2), (22, 23, 2)]
        graph = nk.graph.Graph(edges, NUM_SITES)
        EXTENT_STR = "[3,4]"

elif NUM_SITES == 32:
        graph = nk.graph.KitaevHoneycomb(extent=[4, 4], pbc=True)
        EXTENT_STR = "[4,4]"

elif NUM_SITES == 50:
        graph = nk.graph.KitaevHoneycomb(extent=[5, 5], pbc=True)
        EXTENT_STR = "[5,5]"

elif NUM_SITES == 54:
        edges = [(0, 37, 1), (0, 1, 2), (0, 8, 0), (1, 2, 1), (1, 48, 0), (2, 3, 2),
                 (2, 10, 0), (3, 4, 1), (3, 50, 0), (4, 12, 0), (4, 5, 2), (5, 6, 1),
                 (5, 52, 0), (6, 14, 0), (6, 27, 2), (7, 46, 1), (7, 8, 2), (7, 17, 0),
                 (8, 9, 1), (9, 19, 0), (9, 10, 2), (10, 11, 1), (11, 12, 2), (11, 21, 0),
                 (12, 13, 1), (13, 23, 0), (13, 14, 2), (14, 15, 1), (15, 25, 0), (15, 38, 2),
                 (16, 53, 1), (16, 17, 2), (16, 27, 0), (17, 18, 1), (18, 19, 2), (18, 29, 0),
                 (19, 20, 1), (20, 21, 2), (20, 31, 0), (21, 22, 1), (22, 33, 0), (22, 23, 2),
                 (23, 24, 1), (24, 35, 0), (24, 25, 2), (25, 26, 1), (26, 47, 2), (26, 37, 0),
                 (27, 28, 1), (28, 29, 2), (28, 38, 0), (29, 30, 1), (30, 40, 0), (30, 31, 2),
                 (31, 32, 1), (32, 42, 0), (32, 33, 2), (33, 34, 1), (34, 44, 0), (34, 35, 2),
                 (35, 36, 1), (36, 37, 2), (38, 39, 1), (39, 47, 0), (39, 40, 2), (40, 41, 1),
                 (41, 49, 0), (41, 42, 2), (42, 43, 1), (43, 51, 0), (43, 44, 2), (44, 45, 1),
                 (45, 46, 2), (47, 48, 1), (48, 49, 2), (49, 50, 1), (50, 51, 2), (51, 52, 1),
                 (52, 53, 2), (36, 46, 0), (45, 53, 0)]
        graph = nk.graph.Graph(edges, NUM_SITES)
        EXTENT_STR = "NA"

elif NUM_SITES == 72:
        graph = nk.graph.KitaevHoneycomb(extent=[6,6], pbc=True)
        EXTENT_STR = "[6,6]"

elif NUM_SITES == 96:
        edges = [(0, 62, 1), (0, 1, 2), (0, 10, 0), (1, 2, 1), (1, 88, 0), (2, 3, 2),
                 (2, 12, 0), (3, 90, 0), (3, 4, 1), (4, 14, 0), (4, 5, 2), (5, 92, 0),
                 (5, 6, 1), (6, 16, 0), (6, 7, 2), (7, 94, 0), (7, 8, 1), (8, 18, 0),
                 (8, 48, 2), (9, 75, 1), (9, 10, 2), (9, 21, 0), (10, 11, 1), (11, 23, 0),
                 (11, 12, 2), (12, 13, 1), (13, 25, 0), (13, 14, 2), (14, 15, 1), (15, 27, 0),
                 (15, 16, 2), (16, 17, 1), (17, 29, 0), (17, 18, 2), (18, 19, 1), (19, 31, 0),
                 (19, 63, 2), (20, 86, 1), (20, 34, 0), (20, 21, 2), (21, 22, 1), (22, 36, 0),
                 (23, 24, 1), (22, 23, 2), (24, 38, 0), (24, 25, 2), (25, 26, 1), (26, 40, 0),
                 (26, 27, 2), (27, 28, 1), (28, 42, 0), (28, 29, 2), (29, 30, 1), (30, 44, 0),
                 (30, 31, 2), (31, 32, 1), (32, 46, 0), (32, 76, 2), (33, 95, 1), (33, 34, 2),
                 (33, 48, 0), (34, 35, 1), (35, 50, 0), (35, 36, 2), (36, 37, 1), (37, 52, 0),
                 (37, 38, 2), (38, 39, 1), (39, 54, 0), (39, 40, 2), (40, 41, 1), (41, 56, 0),
                 (41, 42, 2), (42, 43, 1), (43, 58, 0), (43, 44, 2), (44, 45, 1), (45, 60, 0),
                 (45, 46, 2), (46, 47, 1), (47, 62, 0), (47, 87, 2), (48, 49, 1), (49, 63, 0),
                 (49, 50, 2), (50, 51, 1), (51, 65, 0), (51, 52, 2), (52, 53, 1), (53, 67, 0),
                 (53, 54, 2), (54, 55, 1), (55, 69, 0), (55, 56, 2), (56, 57, 1), (57, 71, 0),
                 (57, 58, 2), (58, 59, 1), (59, 73, 0), (59, 60, 2), (60, 61, 1), (61, 75, 0),
                 (61, 62, 2), (63, 64, 1), (64, 76, 0), (64, 65, 2), (65, 66, 1), (66, 78, 0),
                 (66, 67, 2), (67, 68, 1), (68, 80, 0), (68, 69, 2), (69, 70, 1), (70, 82, 0),
                 (70, 71, 2), (71, 72, 1), (72, 73, 2), (72, 84, 0), (73, 74, 1), (74, 86, 0),
                 (74, 75, 2), (76, 77, 1), (77, 87, 0), (77, 78, 2), (78, 79, 1),
                 (79, 89, 0), (79, 80, 2), (80, 81, 1), (81, 91, 0), (81, 82, 2), (82, 83, 1),
                 (83, 93, 0), (83, 84, 2), (84, 85, 1), (85, 95, 0), (85, 86, 2), (87, 88, 1),
                 (88, 89, 2), (89, 90, 1), (90, 91, 2), (91, 92, 1), (92, 93, 2), (93, 94, 1),
                 (94, 95, 2)]
        graph = nk.graph.Graph(edges, NUM_SITES)
        EXTENT_STR = ""

elif NUM_SITES == 98:
        graph = nk.graph.KitaevHoneycomb(extent=[7,7], pbc=True)
        EXTENT_STR = "[7,7]"

elif NUM_SITES == 128:
        graph = nk.graph.KitaevHoneycomb(extent=[8,8], pbc=True)
        EXTENT_STR = "[8,8]"

#Use Netket to find symmetries of the graph
symmetries = graph.automorphisms()
print("Basic KitaevHoneycomb object with extent {}".format(EXTENT_STR))
print("___________________________________________________")
print("Number of Nodes = ", graph.n_nodes)
print("Number of Edges = ", graph.n_edges)
print("Number of Symmetries = ", len(symmetries))
print("Edges = ", graph.edges(return_color=True))
print('\n')


hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)

K = 0.0
G = 1.0
Kx = K
Ky = K
Kz = K

Gx = G
Gy = G
Gz = G

# Define the Bond Operators
SxSx = np.array(
            [
             	[0,0,0,1],
                [0,0,1,0],
                [0,1,0,0],
                [1,0,0,0]
            ])

SySy = np.array(
            [
             	[0,0,0,-1],
                [0,0,1,0],
                [0,1,0,0],
                [-1,0,0,0]
            ])

SxSz = np.array(
            [
                [0,0,1,0],
                [0,0,0,-1],
                [1,0,0,0],
                [0,-1,0,0]
            ])

SxSy = np.array(
            [
                [0,0,0,-1.j],
                [0,0,1.j,0],
                [0,-1.j,0,0],
                [1.j,0,0,0]
            ])

SySx = np.array(
            [
                [0,0,0,-1.j],
                [0,0,-1.j,0],
                [0,1.j,0,0],
                [1.j,0,0,0]
            ])

SySz = np.array(
            [
             	[0,0,-1.j,0],
                [0,0,0,1.j],
                [1.j,0,0,0],
                [0,-1.j,0,0]
            ])

SzSx = np.array(
            [
                [0,1,0,0],
                [1,0,0,0],
                [0,0,0,-1],
                [0,0,-1,0]
            ])

SzSy = np.array(
            [
                [0,-1.j,0,0],
                [1.j,0,0,0],
                [0,0,0,1.j],
                [0,0,-1.j,0]
            ])

SzSz = np.array(
            [
             	[1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,1]
            ])

SySx_SxSy = np.array(
            [
             	[0, 0, 0, -2.0j],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [2.0j, 0, 0, 0]
            ]
            )

SzSx_SxSz = np.array(
            [
             	[0, 1., 1., 0],
                [1., 0, 0, -1.],
                [1., 0, 0, -1.],
                [0, -1., -1., 0]
            ]
            )

SySz_SzSy = np.array(
            [
                [0, -1.j, -1.j, 0],
                [1.j, 0, 0, 1.j],
                [1.j, 0, 0, 1.j],
                [0, -1.j, -1.j, 0]
            ]
            )



# Defining the Hamiltonian
ha = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                bond_ops=[Kx*SxSx,Kz*SzSz, Ky*SySy, Gy*SzSx_SxSz, \
                                          Gz*SySx_SxSy, Gx*SySz_SzSy],\
                                bond_ops_colors=[1,2,0,0,2,1],dtype=np.complex128)
ha = ha.to_jax_operator()

if NUM_SITES <= 18:
        sp_h = ha.to_sparse()

        eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
        E_gs = eig_vals[0]

        print("Ground State Energy = ", E_gs)

#Feature dimensions of hidden layers, from first to last
feature_dims = WIDTH

#Define the ResGCNN
ma = nk.models.GCNN(symmetries = symmetries, layers = NUM_BLOCKS, features = feature_dims, param_dtype=np.complex128)

#Metropolis-Hastings with two spins flipped that are at most second nearest neighbors
sa = nk.sampler.MetropolisLocal(hilbert = hi)

#Stochastic reconfiguration
sr = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianPyTree, diag_scale=0.01, diag_shift=0.001)

#Define a variational state so we can keep the parameters if we like
vstate_1 = nk.vqs.MCState(sampler=sa, model=ma, n_samples=NUM_SAMPLES, chunk_size=CHUNK_SIZE)

#with open(PREV_VARIABLES+".mpack", 'rb') as file:
#        vstate_1 = flax.serialization.from_bytes(vstate_1, file.read())

vars = nkx.vqs.variables_from_file(PREV_VARIABLES, vstate_1.variables)
vstate_1.variables = vars


# Define a dictionary of observables
obs = {}

# Correlations Between site 0 and every other site in the lattice
print("------------Correlations (XX,YY,Z) between site 0 and site i------------")
for i in range(1, hi.size):
        # Defining XX, YY, and ZZ correlators between site 0 and site i
        XX = nk.operator.spin.sigmax(hi,0)*nk.operator.spin.sigmax(hi, i)/4
        YY = nk.operator.spin.sigmay(hi,0)*nk.operator.spin.sigmay(hi, i)/4
        ZZ = nk.operator.spin.sigmaz(hi,0)*nk.operator.spin.sigmaz(hi, i)/4
        obs[f"XX_[0,{i}]"] = XX
        obs[f"YY_[0,{i}]"] = YY
        obs[f"ZZ_[0,{i}]"] = ZZ


# Local X, Y, Z per-spin
x_localop = sum([nk.operator.spin.sigmax(hi, i) for i in range(hi.size)])/(2*NUM_SITES)
y_localop = sum([nk.operator.spin.sigmay(hi, i) for i in range(hi.size)])/(2*NUM_SITES)
z_localop = sum([nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)])/(2*NUM_SITES)

x_localop = x_localop.to_jax_operator()
y_localop = y_localop.to_jax_operator()
z_localop = z_localop.to_jax_operator()

print("------------Average (X,Y,Z)-Spin------------")
print("<X> = ", vstate_1.expect(x_localop))
print("<Y> = ", vstate_1.expect(y_localop))
print("<Z> = ", vstate_1.expect(z_localop))

print("------------Expected (X,Y,Z)-Spin on each site------------")
for site in range(hi.size):
        x_localop_site = nk.operator.spin.sigmax(hi, site)/2
        y_localop_site = nk.operator.spin.sigmay(hi, site)/2
        z_localop_site = nk.operator.spin.sigmaz(hi, site)/2

        print("<X> on site {} = ".format(site), vstate_1.expect(x_localop_site))
        print("<Y> on site {} = ".format(site), vstate_1.expect(y_localop_site))
        print("<Z> on site {} = ".format(site), vstate_1.expect(z_localop_site))
        print('\n')

# Number of X, Y, and Z type bonds for the purpose of averaging over each bond type
num_X_type_bonds = len(graph.edges(return_color=True, filter_color=1))
num_Y_type_bonds = len(graph.edges(return_color=True, filter_color=0))
num_Z_type_bonds = len(graph.edges(return_color=True, filter_color=2))

#print("Numberof X-bonds = ", num_X_type_bonds)
#print("Numberof Y-bonds = ", num_Y_type_bonds)
#print("Numberof Z-bonds = ", num_Z_type_bonds)

# XX, YY, and ZZ Correlations. Calculates SxSx, SySy, and SzSz across all edges in the
# lattice
XX = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                bond_ops=[SxSx/4.],\
                                bond_ops_colors=[],dtype=np.complex128)

YY = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                bond_ops=[SySy/4.],\
                                bond_ops_colors=[],dtype=np.complex128)

ZZ = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                bond_ops=[SzSz/4.],\
                                bond_ops_colors=[],dtype=np.complex128)

# Averaging over the total number of edges
XX = XX/graph.n_edges
YY = YY/graph.n_edges
ZZ = ZZ/graph.n_edges

XX = XX.to_jax_operator()
YY = YY.to_jax_operator()
ZZ = ZZ.to_jax_operator()

print("------------XX, YY, ZZ Correlations across all bonds------------")
print("<XX> = ", vstate_1.expect(XX))
print("<YY> = ", vstate_1.expect(YY))
print("<ZZ> = ", vstate_1.expect(ZZ))
print('\n')

bond_types = ['Y', 'X', 'Z']
num_type_bonds = [num_Y_type_bonds, num_X_type_bonds, num_Z_type_bonds]
for i in range(3):

        print("------------XX, YY, ZZ Correlations across {}-bonds------------".format(bond_types[i]))
        # XX, YY, and ZZ Correlations. Calculates SxSx, SySy, and SzSz across all edges in the
        # lattice
        XX = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                        bond_ops=[SxSx/4.],\
                                        bond_ops_colors=[i],dtype=np.complex128)

        YY = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                        bond_ops=[SySy/4.],\
                                        bond_ops_colors=[i],dtype=np.complex128)

        ZZ = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                        bond_ops=[SzSz/4.],\
                                        bond_ops_colors=[i],dtype=np.complex128)

        # Averaging over the total number of edges
        XX = XX/num_type_bonds[i]
        YY = YY/num_type_bonds[i]
        ZZ = ZZ/num_type_bonds[i]

        XX = XX.to_jax_operator()
        YY = YY.to_jax_operator()
        ZZ = ZZ.to_jax_operator()

        print("<XX> = ", vstate_1.expect(XX))
        print("<YY> = ", vstate_1.expect(YY))
        print("<ZZ> = ", vstate_1.expect(ZZ))

        # XY, XZ, YZ, YX, ZX, ZY. These are only calculated for the corresponding  bond-type
        # where AB is calculated across bond-type C for A,B,C in {X,Y,Z}
        XY = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                        bond_ops=[SxSy/4.],\
                                        bond_ops_colors=[i],dtype=np.complex128)

        XZ = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                        bond_ops=[SxSz/4.],\
                                        bond_ops_colors=[i],dtype=np.complex128)

        YZ = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                        bond_ops=[SySz/4.],\
                                        bond_ops_colors=[i],dtype=np.complex128)

        YX = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                        bond_ops=[SySx/4.],\
                                        bond_ops_colors=[i],dtype=np.complex128)

        ZX = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                        bond_ops=[SzSx/4.],\
                                        bond_ops_colors=[i],dtype=np.complex128)

        ZY = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                        bond_ops=[SzSy/4.],\
                                        bond_ops_colors=[i],dtype=np.complex128)

        # Averaged over the corresponding number of bonds
        XY = XY/num_type_bonds[i]
        XZ = XZ/num_type_bonds[i]
        YZ = YZ/num_type_bonds[i]
        YX = YX/num_type_bonds[i]
        ZX = ZX/num_type_bonds[i]
        ZY = ZY/num_type_bonds[i]

        XY = XY.to_jax_operator()
        XZ = XZ.to_jax_operator()
        YZ = YZ.to_jax_operator()
        YX = YX.to_jax_operator()
        ZX = ZX.to_jax_operator()
        ZY = ZY.to_jax_operator()


        print("------------XY, XZ, YZ, YX, ZX, ZY across {}-bonds------------".format(bond_types[i]))
        print("<XY> = ", vstate_1.expect(XY))
        print("<XZ> = ", vstate_1.expect(XZ))
        print("<YZ> = ", vstate_1.expect(YZ))
        print("<YX> = ", vstate_1.expect(YX))
        print("<ZX> = ", vstate_1.expect(ZX))
        print("<ZY> = ", vstate_1.expect(ZY))


        XY_minus_YX = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                        bond_ops=[(SxSy-SySx)/4.],\
                                        bond_ops_colors=[i],dtype=np.complex128)

        XZ_minus_ZX = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                        bond_ops=[(SxSz-SzSx)/4.],\
                                        bond_ops_colors=[i],dtype=np.complex128)

        ZY_minus_YZ = nk.operator.GraphOperator(hilbert=hi, graph=graph, site_ops=[],\
                                        bond_ops=[(SzSy-SySz)/4.],\
                                        bond_ops_colors=[i],dtype=np.complex128)

        XY_minus_YX = XY_minus_YX/num_type_bonds[i]
        XZ_minus_ZX = XZ_minus_ZX/num_type_bonds[i]
        ZY_minus_YZ = ZY_minus_YZ/num_type_bonds[i]

        XY_minus_YX = XY_minus_YX.to_jax_operator()
        XZ_minus_ZX = XZ_minus_ZX.to_jax_operator()
        ZY_minus_YZ = ZY_minus_YZ.to_jax_operator()

        print("------------XY - YX, XZ - ZX, YZ - ZY across {}-bonds------------".format(bond_types[i]))
        print("<XY - YX> = ", vstate_1.expect(XY_minus_YX))
        print("<XZ - ZX> = ", vstate_1.expect(XZ_minus_ZX))
        print("<YZ - ZY> = ", vstate_1.expect(ZY_minus_YZ))
        print("-----------------------------------------------------------------")
        print('\n')
'''
