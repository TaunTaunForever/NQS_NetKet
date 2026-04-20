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
NUM_SITES = 36
NUM_SAMPLES = 2**11
NUM_ITERS = 10000
NUM_BLOCKS = 4
CHUNK_SIZE = 2**9
WIDTH = 8
EXTENT_STR = ""
J1 = 1.0
J2 = 0.20
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

if NUM_SITES == 16:
    GCNN_hidden_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,
     0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
     1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0,
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1,
     0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1,
     1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1,]
elif NUM_SITES == 36:
    GCNN_hidden_mask = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1,
     0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1,
     0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
     1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
     1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1,
     0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1,
     1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1,
     0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
     1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0,
     1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1,
     0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1,
     0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1,
     1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1,
     1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1,]
#Feature dimensions of hidden layers, from first to last
feature_dims = WIDTH

#Define the Neural Network Model (First with 
ma = nk.models.GCNN(symmetries = symmetries, layers = NUM_BLOCKS, equal_amplitudes = True, features = feature_dims,\
                        activation = nk.nn.activation.reim_selu,param_dtype=np.complex128) 
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
                        activation = nk.nn.activation.reim_selu, param_dtype=np.complex128)
# Second Run
op_1 = nk.optimizer.Sgd(learning_rate=1e-2)
vstate_2 = nk.vqs.MCState(sampler=sa, model=ma, n_samples=NUM_SAMPLES, variables = vstate_1.variables, chunk_size=CHUNK_SIZE)
gs = nkx.driver.VMC_SRt(hamiltonian=ha, optimizer=op_1, variational_state=vstate_2, diag_shift=0.001, \
                        obs={"<X>": x_localop, "<Y>" : y_localop, "<Z>": z_localop})

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
