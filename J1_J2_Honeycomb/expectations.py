import os

os.environ["JAX_PLATFORM_NAME"] = "gpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import netket as nk


def define_observables(num_sites, hi):
    obs = {}

    x_localop = sum(nk.operator.spin.sigmax(hi, i) for i in range(hi.size)) / (2 * num_sites)
    y_localop = sum(nk.operator.spin.sigmay(hi, i) for i in range(hi.size)) / (2 * num_sites)
    z_localop = sum(nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)) / (2 * num_sites)

    obs.update({"<X>": x_localop})
    obs.update({"<Y>": y_localop})
    obs.update({"<Z>": z_localop})

    for i in range(num_sites):
        obs[f"X_{i}"] = nk.operator.spin.sigmax(hi, i) / 2
        obs[f"Y_{i}"] = nk.operator.spin.sigmay(hi, i) / 2
        obs[f"Z_{i}"] = nk.operator.spin.sigmaz(hi, i) / 2

    print("------------Correlations (XX, YY, ZZ) between site 0 and site i------------")
    for i in range(1, hi.size):
        obs[f"XX_[0,{i}]"] = nk.operator.spin.sigmax(hi, 0) * nk.operator.spin.sigmax(hi, i) / 4
        obs[f"YY_[0,{i}]"] = nk.operator.spin.sigmay(hi, 0) * nk.operator.spin.sigmay(hi, i) / 4
        obs[f"ZZ_[0,{i}]"] = nk.operator.spin.sigmaz(hi, 0) * nk.operator.spin.sigmaz(hi, i) / 4

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
