import sys
import json
import os

import jax
import netket as nk
import netket.experimental as nkx

from datetime import date
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import optax
import expectations

TODAY = date.today().isoformat()
NUM_SITES = 24
NUM_SAMPLES = 2**10
NUM_ITERS = 100000
NUM_BLOCKS = 4
CHUNK_SIZE = None
WIDTH = 24
EXTENT_STR = ""
K = 0.0
G = 1.0
JOB_NUMBER = "G={}_{}-site_{}_blocks_{}_width_{}_samples_{}_SRt".format(G, NUM_SITES, NUM_BLOCKS,
                WIDTH, NUM_SAMPLES, TODAY)


os.environ["NETKET_DEBUG"] = "1"

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
                 (19, 63, 20), (20, 86, 1), (20, 34, 0), (20, 21, 2), (21, 22, 1), (22, 36, 0),
                 (23, 24, 1), (22, 23, 2), (24, 38, 0), (24, 25, 2), (25, 26, 1), (26, 40, 0),
                 (26, 27, 2), (27, 28, 1), (28, 42, 0), (28, 29, 2), (29, 30, 1), (30, 44, 0),
                 (30, 31, 2), (31, 32, 1), (32, 46, 0), (32, 76, 2), (33, 95, 1), (33, 34, 2),
                 (33, 48, 0), (39, 54, 0), (39, 40, 2), (40, 41, 1), (41, 56, 0), (41, 42, 2),
                 (42, 43, 1), (43, 58, 0), (43, 44, 2), (44, 45, 1), (45, 60, 0), (45, 46, 2),
                 (46, 47, 1), (47, 62, 0), (47, 87, 2), (48, 49, 1), (49, 63, 0), (49, 50, 2),
                 (50, 51, 1), (51, 65, 0)]
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


#################################################################################
# Defining Hilbert Space
#################################################################################
hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes)


#################################################################################
# Defining Hamailtonian
#################################################################################

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
                [0, 0, 0, -2.j],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [2.j, 0, 0, 0]
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

# Checking Exact Diagonalization results for the hamiltonian, if this is 
# correctly reporting ~24-25 for a 32-site lattice, then we're good.
if NUM_SITES <= 18:
        sp_h = ha.to_sparse()

        eig_vals, eig_vecs = eigsh(sp_h, k=2, which="SA")
        E_gs = eig_vals[0]

        print("Ground State Energy = ", E_gs)


# Local X, Y, Z per-spin
x_localop = sum([nk.operator.spin.sigmax(hi, i) for i in range(hi.size)])/(2*NUM_SITES)
y_localop = sum([nk.operator.spin.sigmay(hi, i) for i in range(hi.size)])/(2*NUM_SITES)
z_localop = sum([nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)])/(2*NUM_SITES)


#################################################################################
# GCNN Optimization 
#################################################################################

#Feature dimensions of hidden layers, from first to last
feature_dims = WIDTH

#Define the ResGCNN 
#Define the ResGCNN
GCNN_hidden_mask = np.ones(len(symmetries))
mask_nums = []
if NUM_SITES == 32:
    mask_nums = [0, 31, 6, 25, 4, 27, 24, 7, 30, 1, 28, 3, 16, 15, 22, 9, 20, 11]
elif NUM_SITES == 72:
    mask_nums = [0, 71, 10, 61, 8, 63, 60, 11, 70, 1, 68, 3, 48, 23, 58, 13, 56, 15]

for i in mask_nums:
    GCNN_hidden_mask[i] = 0
print(GCNN_hidden_mask)

#ma = nk.models.GCNN(symmetries = symmetries, layers = NUM_BLOCKS, equal_amplitudes = True, features = feature_dims,\
#                        activation = nk.nn.activation.reim_selu, param_dtype=np.complex128)
ma = nk.models.GCNN(symmetries = symmetries, layers = NUM_BLOCKS, equal_amplitudes = True, features = feature_dims,\
                        activation = nk.nn.activation.reim_selu, hidden_mask = GCNN_hidden_mask, param_dtype=np.complex128)

#Metropolis-Hastings with two spins flipped that are at most second nearest neighbors 
#sa = nk.sampler.MetropolisExchange(hilbert = hi, graph=graph)
sa = nk.sampler.MetropolisLocal(hilbert = hi,  n_chains=NUM_SAMPLES)
#sa = nk.sampler.ParallelTemperingLocal(hilbert=hi, n_chains=NUM_SAMPLES/64)

#Stochastic reconfiguration
op_1 = nk.optimizer.Sgd(learning_rate=1e-2)

#Define a variational state so we can keep the parameters if we like
vstate_1 = nk.vqs.MCState(sampler=sa, model=ma, n_samples=NUM_SAMPLES, chunk_size=CHUNK_SIZE)
print("num_samples: ", NUM_SAMPLES)
print("num params: ", vstate_1.n_parameters)
print("n_discard_per_chain: ", vstate_1.n_discard_per_chain)
print("chain_length: ", vstate_1.chain_length)
 
#Define a driver that performs VMC
gs = nkx.driver.VMC_SRt(hamiltonian=ha, optimizer=op_1, variational_state=vstate_1, diag_shift=0.001)

#Run the optimization
gs.run(n_iter=100, out='out_{}'.format(JOB_NUMBER), save_params_every=10)

# Now running the optimization
#ma = nk.models.GCNN(symmetries = symmetries, layers = NUM_BLOCKS, equal_amplitudes = False, features = feature_dims,\
#                        activation = nk.nn.activation.reim_selu, param_dtype=np.complex128)
ma = nk.models.GCNN(symmetries = symmetries, layers = NUM_BLOCKS, equal_amplitudes = False, features = feature_dims,\
                        activation = nk.nn.activation.reim_selu, hidden_mask = GCNN_hidden_mask, param_dtype=np.complex128)
# Second Run
vstate_2 = nk.vqs.MCState(sampler=sa, model=ma, n_samples=NUM_SAMPLES, variables = vstate_1.variables, chunk_size=CHUNK_SIZE)
gs = nkx.driver.VMC_SRt(hamiltonian=ha, optimizer=op_1, variational_state=vstate_2, diag_shift=0.001)
gs.run(n_iter=NUM_ITERS, obs={"<X>": x_localop, "<Y>": y_localop, "<Z>": z_localop}, out='out_{}'.format(JOB_NUMBER))

#Get data from log and 
energy = []
data=json.load(open("out_{}.log".format(JOB_NUMBER)))
for en in data["Energy"]["Mean"]["real"]:
    print(en)
    energy.append(en)

with open("mean_energy_run_{}.txt".format(JOB_NUMBER), "w") as f:
    for item in energy:
        f.write("{}\n".format(item))
f.close()


# Calculated Expectations
obs = expectations.define_observables(NUM_SITES, hi)
data = expectations.calculate_expectations(gs, obs)
