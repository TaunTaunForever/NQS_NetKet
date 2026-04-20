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
import flax


PREV_VARIABLES = "out_J1=1.0_J2=0.05_36-site_4_GCNN_layers_20_width_4096_samples_2025-06-17_learning_rate=0.01.mpack"
TODAY = date.today().isoformat()
NUM_SITES = 36
NUM_SAMPLES = 2**12
NUM_ITERS = 10000
NUM_BLOCKS = 4
CHUNK_SIZE = 1024
WIDTH = 20
EXTENT_STR = ""
J1 = 1.0
J2 = 0.05
RUN_NUM = 2
JOB_NUMBER = "J1={}_J2={}_{}-site_{}_GCNN_layers_{}_width_{}_samples_{}_run_{}".format(J1, J2, NUM_SITES, NUM_BLOCKS,
                WIDTH, NUM_SAMPLES, TODAY, RUN_NUM)

os.environ["NETKET_DEBUG"] = "1"

#################################################################################
# Defining Triangular Graph and Finding Symmetries
#################################################################################
print("Defining {}-site Triangular lattice".format(NUM_SITES))
print("___________________________________________________")
graph = nk.graph.Triangular(extent=[6, 6], max_neighbor_order=2, pbc=True)
EXTENT_STR = "[6,6]"
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

#Feature dimensions of hidden layers, from first to last
feature_dims = WIDTH

#Define the ResGCNN
# For multivalued feature dimensions
feature_dims = WIDTH
#ma = nk.models.GCNN(symmetries = symmetries, layers = NUM_BLOCKS, features = feature_dims,
#                       activation = nk.nn.activation.reim_selu, param_dtype=np.complex128)
ma = nk.models.GCNN(symmetries = graph.point_group(), layers = NUM_BLOCKS, features = feature_dims,\
                       activation = nk.nn.activation.reim_selu, param_dtype=np.complex128)



sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=graph)

#Stochastic reconfiguration
op_1 = nk.optimizer.Sgd(learning_rate=1e-3)

#Define a variational state so we can keep the parameters if we like
vstate_1 = nk.vqs.MCState(sampler=sa, model=ma, n_samples=NUM_SAMPLES, chunk_size=CHUNK_SIZE)
vars = nkx.vqs.variables_from_file(PREV_VARIABLES, vstate_1.variables)
vstate_1.variables = vars
print("num_samples: ", NUM_SAMPLES)
print("num params: ", vstate_1.n_parameters)
print("n_discard_per_chain: ", vstate_1.n_discard_per_chain)
print("chain_length: ", vstate_1.chain_length)

#Define a driver that performs VMC
gs = nkx.driver.VMC_SRt(hamiltonian=ha, optimizer=op_1, variational_state=vstate_1, diag_shift=0.001)

#Run the optimization
gs.run(n_iter=NUM_ITERS, out='out_{}'.format(JOB_NUMBER), obs={"<X>": x_localop, "<Y>" : y_localop, "<Z>": z_localop})
print("-------------------------------------------------------------------------------")
print("Final Expected Value: ", vstate_1.expect(ha))

#with open("variational_state_{}.mpack".format(JOB_NUMBER), "wb") as file:
#        file.write(flax.serialization.to_bytes(vstate_1))


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

