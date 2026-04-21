import sys
import json
import os
from datetime import date
import netket as nk
import numpy as np
import matplotlib.pyplot as plt
from netket.operator.spin import sigmax,sigmaz, sigmay
from scipy.sparse.linalg import eigsh
import tracemalloc
import netket.experimental as nkx
os.environ["NETKET_DEBUG"] = "1"

PREV_VARIABLES = "out_GCNN_G=1.0_128-site_4_layers_20_width_1024_samples_run-4_2025-05-28_minSR.mpack"
NUM_BLOCKS = 4
WIDTH = 20
NUM_SAMPLES = 2**10
CHUNK_SIZE = None
NUM_SITES = 128
TODAY = date.today().isoformat()
NUM_ITERS = 20000
G = 1.0
RUN_NUM = 5
JOB_NUMBER = "G={}_{}-site_{}_layers_{}_width_{}_samples_run-{}_{}_minSR".format(G, NUM_SITES, NUM_BLOCKS,
                WIDTH, NUM_SAMPLES,RUN_NUM, TODAY)

EXTENT_STR = ""




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
        graph = nk.graph.KitaevHoneycomb(extent=[6, 6], pbc=True)
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
        graph = nk.graph.KitaevHoneycomb(extent=[8, 8], pbc=True)
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



#################################################################################
# GCNN Optimization 
#################################################################################

#Feature dimensions of hidden layers, from first to last
feature_dims = WIDTH

#Define the ResGCNN 
#ma = nk.models.ResGCNN(symmetries = symmetries, layers = NUM_BLOCKS, features = feature_dims, param_dtype=np.complex128)
ma = nk.models.GCNN(symmetries = symmetries, layers = NUM_BLOCKS, features = feature_dims, param_dtype=np.complex128)

#Metropolis-Hastings with two spins flipped that are at most second nearest neighbors 
sa = nk.sampler.MetropolisLocal(hilbert = hi)

#Stochastic reconfiguration   
sr = nk.optimizer.SR(nk.optimizer.qgt.QGTJacobianPyTree, diag_scale=0.01, diag_shift=0.001)

#Define a variational state so we can keep the parameters if we like
vstate_1 = nk.vqs.MCState(sampler=sa, model=ma, n_samples=NUM_SAMPLES, chunk_size=CHUNK_SIZE)
vars = nkx.vqs.variables_from_file(PREV_VARIABLES, vstate_1.variables)
vstate_1.variables = vars

#Stochastic reconfiguration
op_1 = nk.optimizer.Sgd(learning_rate=1e-3)

print("num layers: ", NUM_BLOCKS)
print("num nodes per layer: ", WIDTH)
print("num_samples: ", NUM_SAMPLES)
print("num params: ", vstate_1.n_parameters)
print("n_discard_per_chain: ", vstate_1.n_discard_per_chain)
print("chain_length: ", vstate_1.chain_length)
 
#Define a driver that performs VMC
gs = nkx.driver.VMC_SRt(hamiltonian=ha, optimizer=op_1, variational_state=vstate_1, diag_shift=0.001)

#Run the optimization
gs.run(n_iter=NUM_ITERS, out='out_GCNN_{}'.format(JOB_NUMBER), save_params_every=10)

print(vstate_1.expect(ha))
x_localop = sum([nk.operator.spin.sigmax(hi, i) for i in range(hi.size)])/NUM_SITES
y_localop = sum([nk.operator.spin.sigmay(hi, i) for i in range(hi.size)])/NUM_SITES
z_localop = sum([nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)])/NUM_SITES

print("Average X per-spin: ", vstate_1.expect(x_localop))
print("Average Y per-spin: ", vstate_1.expect(y_localop))
print("Average Z per-spin: ", vstate_1.expect(z_localop))


#Get data from log and 
energy = []
data=json.load(open("out_GCNN_{}.log".format(JOB_NUMBER)))
for en in data["Energy"]["Mean"]["real"]:
    print(en)
    energy.append(en)

with open("mean_energy_run_{}.txt".format(JOB_NUMBER), "w") as f:
    for item in energy:
        f.write("{}\n".format(item))
f.close()
