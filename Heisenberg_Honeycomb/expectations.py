import os
os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
import netket as nk

def define_observables(num_sites, hi):
    obs = {}

    # Average system spin expectations
    x_localop = sum([nk.operator.spin.sigmax(hi, i) for i in range(hi.size)])/(2*num_sites)
    y_localop = sum([nk.operator.spin.sigmay(hi, i) for i in range(hi.size)])/(2*num_sites)
    z_localop = sum([nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)])/(2*num_sites)

    X_key = "<X>"
    Y_key = "<Y>"
    Z_key = "<Z>"


    obs.update({"<X>": x_localop})
    obs.update({"<Y>": y_localop})
    obs.update({"<Z>": z_localop})

    # On-site spin expectations
    for i in range(num_sites):
        X_i = nk.operator.spin.sigmax(hi, i)/2
        Y_i = nk.operator.spin.sigmay(hi, i)/2
        Z_i = nk.operator.spin.sigmaz(hi, i)/2

        X_i_label = "X_{}".format(i)
        Y_i_label = "Y_{}".format(i)
        Z_i_label = "Z_{}".format(i)

        obs.update({X_i_label: X_i})
        obs.update({Y_i_label: Y_i})
        obs.update({Z_i_label: Z_i})


    # Correlations Between site 0 and every other site in the lattice
    print("------------Correlations (XX,YY,Z) between site 0 and site i------------")
    for i in range(1, hi.size):
        # Defining XX, YY, and ZZ correlators between site 0 and site i
        XX = nk.operator.spin.sigmax(hi,0)*nk.operator.spin.sigmax(hi, i)/4
        YY = nk.operator.spin.sigmay(hi,0)*nk.operator.spin.sigmay(hi, i)/4
        ZZ = nk.operator.spin.sigmaz(hi,0)*nk.operator.spin.sigmaz(hi, i)/4

        XX_0i_label = f"XX_[0,{i}]"
        YY_0i_label = f"YY_[0,{i}]"
        ZZ_0i_label = f"ZZ_[0,{i}]"

        obs.update({XX_0i_label: XX})
        obs.update({YY_0i_label: YY})
        obs.update({ZZ_0i_label: ZZ})

    return obs



def calculate_expectations(vstate, hamiltonian, obs):
    energy_stats = vstate.expect(hamiltonian)
    print(f"Energy mean value = {energy_stats.mean}")

    expectation_values = {"Energy": energy_stats.mean}
    for key, operator in obs.items():
        stats = vstate.expect(operator)
        expectation_values[key] = stats.mean
        print(f"{key} mean value = {stats.mean}")

    return expectation_values
