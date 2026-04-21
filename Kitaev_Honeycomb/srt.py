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

TODAY = date.today().isoformat()
NUM_SITES = 18
NUM_SAMPLES = 2**11
NUM_ITERS = 100000
NUM_BLOCKS = 4
CHUNK_SIZE = 2**10
WIDTH = 18
EXTENT_STR = ""
K = -1.0
G = 0.0
JOB_NUMBER = "K={}_{}-site_{}_blocks_{}_width_{}_samples_{}_minSR".format(K, NUM_SITES, NUM_BLOCKS,
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
                [0, 0, 0, -0.5j],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0.5j, 0, 0, 0]
            ]
            )

SzSx_SxSz = np.array(
            [
             	[0, 0.25, 0.25, 0],
                [0.25, 0, 0, -0.25],
                [0.25, 0, 0, -0.25],
                [0, -0.25, -0.25, 0]
            ]
            )

SySz_SzSy = np.array(
            [
             	[0, -0.25j, -0.25j, 0],
                [0.25j, 0, 0, 0.25j],
                [0.25j, 0, 0, 0.25j],
                [0, -0.25j, -0.25j, 0]
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

#################################################################################
# GCNN Optimization 
#################################################################################

#Feature dimensions of hidden layers, from first to last
feature_dims = WIDTH

#Define the ResGCNN 
ma = nk.models.GCNN(symmetries = symmetries, layers = NUM_BLOCKS, features = feature_dims,\
                        activation = nk.nn.activation.reim_selu, param_dtype=np.complex128)

#print("model: ", ma)
#Metropolis-Hastings with two spins flipped that are at most second nearest neighbors 
sa = nk.sampler.MetropolisLocal(hilbert = hi, n_chains=NUM_SAMPLES/64)
#sa = nk.sampler.ParallelTemperingLocal(hilbert=hi, n_chains=16)
#sa = nk.sampler.MetropolisLocal(hilbert = hi, n_chains_per_rank= 256)
#Stochastic reconfiguration
op_1 = nk.optimizer.Sgd(learning_rate=1e-2)

#Define a variational state so we can keep the parameters if we like
vstate_1 = nk.vqs.MCState(sampler=sa, model=ma, n_samples=NUM_SAMPLES, chunk_size=CHUNK_SIZE)
#print(".is_probably_holomorphic: ",nk.utils.is_probably_holomorphic(vstate_1._apply_fun, vstate_1.parameters, vstate_1.samples, vstate_1.model_state))
print("num_samples: ", NUM_SAMPLES)
print("num params: ", vstate_1.n_parameters)
print("n_discard_per_chain: ", vstate_1.n_discard_per_chain)
print("chain_length: ", vstate_1.chain_length)

#Define a driver that performs VMC
gs = nkx.driver.VMC_SRt(hamiltonian=ha, optimizer=op_1, variational_state=vstate_1, diag_shift=0.001)

#Run the optimization
gs.run(n_iter=NUM_ITERS, out='out_GCNN_{}'.format(JOB_NUMBER), save_params_every=10)
#jax.profiler.save_device_memory_profile("post-run_{}.prof".format(JOB_NUMBER))

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


#plot the energy during the optimization
plt.xlabel("Number of Iterations")
plt.ylabel("Energy")
plt.title("Total Energy for {}-Site Kitaev Gamma Honeycomb Lattice".format(NUM_SITES))
plt.plot(energy)
fig = plt.gcf()
fig.set_size_inches(20, 15)
'''
if NUM_SITES <= 18:
        plt.axhline(y = E_gs, color = 'r', linestyle = '-')
'''
plt.savefig('kitaev_gamma_{}.png'.format(JOB_NUMBER))

#plot the energy during the optimization
plt.xlabel("Number of Iterations")
plt.ylabel("Energy")
plt.title("Total Energy for {}-Site Kitaev Gamma Honeycomb Lattice (Log Scale)".format(NUM_SITES))
ax = plt.gca()
ax.set_xscale('log')
plt.plot(energy)
fig = plt.gcf()
fig.set_size_inches(20, 15)
'''
if NUM_SITES <= 18:
        plt.axhline(y = E_gs, color = 'r', linestyle = '-')
'''
plt.savefig('kitaev_gamma_log_{}.png'.format(JOB_NUMBER))
#Use Netket to find symmetries of the graph
symmetries = graph.automorphisms()
print("Basic KitaevHoneycomb object with extent {}".format(EXTENT_STR))
print("___________________________________________________")
print("Number of Nodes = ", graph.n_nodes)
print("Number of Edges = ", graph.n_edges)
print("Number of Symmetries = ", len(symmetries))
print("Edges = ", graph.edges(return_color=True))
print('\n')

