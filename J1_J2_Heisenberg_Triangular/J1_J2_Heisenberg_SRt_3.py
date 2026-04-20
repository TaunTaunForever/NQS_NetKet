import sys
import json
import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
from datetime import date    
import netket as nk
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
import optax
import netket.experimental as nkx
import jax.profiler
import flax

TODAY = date.today().isoformat()
NUM_SITES = 64
NUM_SAMPLES = 2**12
NUM_ITERS = 10000
NUM_BLOCKS = 9
CHUNK_SIZE = 2**9
WIDTH = 6
EXTENT_STR = ""
J1 = 1.0
J2 = 0.07
JOB_NUMBER = "J1={}_J2={}_{}-site_{}_GCNN_layers_{}_width_{}_samples_{}".format(J1, J2, NUM_SITES, NUM_BLOCKS,
                WIDTH, NUM_SAMPLES, TODAY)

os.environ["NETKET_DEBUG"] = "1"

#################################################################################
# Defining Triangular Graph and Finding Symmetries
#################################################################################
print("Defining {}-site Triangular lattice".format(NUM_SITES))
print("___________________________________________________")
if NUM_SITES == 16:
    graph = nk.graph.Triangular(extent=[4, 4], max_neighbor_order=2, pbc=True)
    EXTENT_STR = "[4, 4]"
elif NUM_SITES == 24:
    graph = nk.graph.Triangular(extent=[5, 5], max_neighbor_order=2, pbc=True)
    EXTENT_STR = "[5, 5]"
elif NUM_SITES == 36:
    graph = nk.graph.Triangular(extent=[6, 6], max_neighbor_order=2, pbc=True)
    EXTENT_STR = "[6, 6]"
elif NUM_SITES == 64:
    graph = nk.graph.Triangular(extent=[8, 8], max_neighbor_order=2, pbc=True)
    EXTENT_STR = "[8, 8]"
elif NUM_SITES == 100:
    graph = nk.graph.Triangular(extent=[10, 10], max_neighbor_order=2, pbc=True)
    EXTENT_STR = "[10, 10]"

#Use Netket to find symmetries of the graph
symmetries = graph.automorphisms()
print("Basic Triangular object with extent {}".format(EXTENT_STR))
print("___________________________________________________")
print("Number of Nodes = ", graph.n_nodes)
print("Number of Edges = ", graph.n_edges)
print("Number of Symmetries = ", len(symmetries))
print("Size of Point Group = ", len(graph.point_group()))
print("Size of Space Group = ", len(graph.space_group()))
print("Size of Rotation Group = ", len(graph.rotation_group()))
print("Edges = ", graph.edges())
print('\n')


#################################################################################
# Defining Hilbert Space
#################################################################################
hi = nk.hilbert.Spin(s=1 / 2, N=graph.n_nodes, total_sz=0)


#################################################################################
# Defining Hamiltonian
#################################################################################
ha = nk.operator.Heisenberg(hilbert=hi, graph=graph, J=[J1,J2])


# Local X, Y, Z per-spin
x_localop = sum([nk.operator.spin.sigmax(hi, i) for i in range(hi.size)])/(2*NUM_SITES)
y_localop = sum([nk.operator.spin.sigmay(hi, i) for i in range(hi.size)])/(2*NUM_SITES)
z_localop = sum([nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)])/(2*NUM_SITES)

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

GCNN_hidden_mask = np.ones(len(symmetries))
mask_nums = []
if NUM_SITES == 36:
    mask_nums = [0, 5, 6, 3, 9, 10, 7, 2, 8, 11, 1, 4, 60, 360, 373, 13, 72, 133, 372, 12, 73, 132, 61, 361, 50, 297, 320, 25, 144, 195, 321, 24, 145, 194, 51, 296, 368, 374, 16, 78, 134, 68, 362, 380, 20, 74, 136, 66, 428, 311, 385, 88, 206, 123, 309, 392, 85, 204, 126, 423, 418, 239, 332, 100, 278, 183, 249, 404, 157, 266, 114, 358, 293, 323, 28, 150, 192, 58, 299, 317, 32, 146, 198, 54, 353, 251, 397, 160, 264, 111, 237, 329, 97, 276, 186, 411, 343, 179, 344, 172, 336, 171, 177, 341, 169, 338, 174, 346]

elif NUM_SITES == 64:
    mask_nums = [0, 5, 6, 3, 9, 10, 7, 2, 8, 11, 1, 4, 84, 672, 685, 13, 96, 181, 684, 12, 97, 180, 85, 673, 74, 585, 608, 25, 192, 267, 609, 24, 193, 266, 75, 584, 680, 686, 16, 102, 182, 92, 674, 692, 20, 98, 184, 90, 764, 599, 697, 112, 278, 171, 597, 704, 109, 276, 174, 759, 754, 503, 620, 124, 374, 255, 513, 716, 205, 362, 162, 670, 581, 611, 28, 198, 264, 82, 587, 605, 32, 194, 270, 78, 665, 515, 709, 208, 360, 159, 501, 617, 121, 372, 258, 747, 655, 419, 632, 220, 456, 243, 417, 629, 217, 458, 246, 658]

elif NUM_SITES == 100:
    mask_nums = [0, 5, 6, 3, 9, 10, 7, 2, 8, 11, 1, 4, 108, 1080, 1093, 13, 120, 229, 1092, 12, 121, 228, 109, 1081, 98, 969, 992, 25, 240, 339, 993, 24, 241, 338, 99, 968, 1088, 1094, 16, 126, 230, 116, 1082, 1100, 20, 122, 232, 114, 1196, 983, 1105, 136, 350, 219, 981, 1112, 133, 348, 222, 1191, 1186, 863, 1004, 148, 470, 327, 873, 1124, 253, 458, 210, 1078, 965, 995, 28, 246, 336, 106, 971, 989, 32, 242, 342, 102, 1073, 875, 1117, 256, 456, 207, 861, 1001, 145, 468, 330, 1179, 1063, 755, 1016, 268, 576, 315, 753, 1013, 265, 578, 318, 1066]

for i in mask_nums:
    GCNN_hidden_mask[i] = 0

#Feature dimensions of hidden layers, from first to last
feature_dims = WIDTH

#Define the Neural Network Model (First with 
ma = nk.models.GCNN(symmetries = symmetries, layers = NUM_BLOCKS, equal_amplitudes = True, features = feature_dims,\
                        activation = nk.nn.activation.reim_selu, hidden_mask = GCNN_hidden_mask, param_dtype=np.complex128)
#Metropolis-Hastings with two spins flipped that are at most second nearest neighbors 
#sa = nk.sampler.MetropolisLocal(hilbert = hi)
#sa = nk.sampler.ParallelTemperingLocal(hilbert=hi, n_chains=NUM_SAMPLES/24)
#sa = nk.sampler.MetropolisLocal(hilbert = hi, n_chains_per_rank= 256)
sa = nk.sampler.ParallelTemperingExchange(hi, graph=graph, d_max=2)
#sa = nk.sampler.MetropolisExchange(hi, graph=graph)
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

#In this run, we are pre-optimizing the phases by restricting all amplitudes to unity
gs.run(n_iter=100, out='out_GCNN_{}'.format(JOB_NUMBER), save_params_every=10)

# Now running the optimization
ma = nk.models.GCNN(symmetries = symmetries, layers = NUM_BLOCKS, equal_amplitudes = False, features = feature_dims,\
                        activation = nk.nn.activation.reim_selu, hidden_mask = GCNN_hidden_mask, param_dtype=np.complex128)
# Second Run
op_1 = nk.optimizer.Sgd(learning_rate=1e-2)
vstate_2 = nk.vqs.MCState(sampler=sa, model=ma, n_samples=NUM_SAMPLES, variables = vstate_1.variables, chunk_size=CHUNK_SIZE)
gs = nkx.driver.VMC_SRt(hamiltonian=ha, optimizer=op_1, variational_state=vstate_2, diag_shift=0.001)

gs.run(n_iter=NUM_ITERS, out='out_{}'.format(JOB_NUMBER))

#Get data from log and 
energy = []
data=json.load(open("out_{}.log".format(JOB_NUMBER)))
for en in data["Energy"]["Mean"]["real"]:
    # print(en)
    energy.append(en)

with open("mean_energy_run_{}.txt".format(JOB_NUMBER), "w") as f:
    for item in energy:
        f.write("{}\n".format(item))
f.close()



#plot the energy during the optimization
plt.xlabel("Number of Iterations")
plt.ylabel("Energy")
plt.title("Total Energy for J1-J2 Model on {}-Site Triangular Lattice".format(NUM_SITES))
plt.plot(energy)
fig = plt.gcf()
fig.set_size_inches(20, 15)
plt.savefig('J1-J2_triangular_{}.png'.format(JOB_NUMBER))

#plot the energy during the optimization
plt.xlabel("Number of Iterations")
plt.ylabel("Energy")
plt.title("Total Energy for J1-J2 Model on {}-Site Triangular Lattice (Log Scale)".format(NUM_SITES))
ax = plt.gca()
ax.set_xscale('log')
plt.plot(energy)
fig = plt.gcf()
fig.set_size_inches(20, 15)
plt.savefig('J1-J2_triangular_log_{}.png'.format(JOB_NUMBER))
