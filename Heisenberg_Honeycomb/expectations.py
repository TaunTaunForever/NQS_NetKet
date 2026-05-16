from __future__ import annotations

import cmath
import math
from collections import defaultdict

import netket as nk
import numpy as np


def _spin_component_component_operator(hi, component: str, i: int, j: int):
    op = {
        "x": nk.operator.spin.sigmax,
        "y": nk.operator.spin.sigmay,
        "z": nk.operator.spin.sigmaz,
    }[component]
    return (op(hi, i) @ op(hi, j)) / 4.0


def _spin_dot_operator(hi, i: int, j: int):
    return (
        nk.operator.spin.sigmax(hi, i) @ nk.operator.spin.sigmax(hi, j)
        + nk.operator.spin.sigmay(hi, i) @ nk.operator.spin.sigmay(hi, j)
        + nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, j)
    ) / 4.0


def _graph_metadata(graph):
    basis_coords = np.asarray(graph.basis_coords, dtype=int)
    basis_vectors = np.asarray(graph.basis_vectors, dtype=float)
    site_offsets = np.asarray(graph.site_offsets, dtype=float)
    extent = np.asarray(graph.extent, dtype=int)
    positions = np.asarray(graph.positions, dtype=float)
    sublattice = basis_coords[:, 2]
    staggered_sign = 1 - 2 * sublattice
    return {
        "basis_coords": basis_coords,
        "basis_vectors": basis_vectors,
        "site_offsets": site_offsets,
        "extent": extent,
        "positions": positions,
        "sublattice": sublattice,
        "staggered_sign": staggered_sign,
    }


def _minimal_image_displacement(meta, i: int, j: int) -> np.ndarray:
    basis_coords = meta["basis_coords"]
    basis_vectors = meta["basis_vectors"]
    site_offsets = meta["site_offsets"]
    extent = meta["extent"]

    cell_delta = basis_coords[j, :2] - basis_coords[i, :2]
    offset_delta = site_offsets[basis_coords[j, 2]] - site_offsets[basis_coords[i, 2]]

    best_vec = None
    best_norm = None
    for s1 in (-extent[0], 0, extent[0]):
        for s2 in (-extent[1], 0, extent[1]):
            shifted = cell_delta + np.array([s1, s2], dtype=int)
            vec = shifted @ basis_vectors + offset_delta
            norm = float(np.dot(vec, vec))
            if best_norm is None or norm < best_norm - 1e-12:
                best_norm = norm
                best_vec = vec
    return best_vec


def _distance_shell(value: float, decimals: int = 8) -> float:
    return float(np.round(value, decimals=decimals))


def _canonical_bond_vector(vec: np.ndarray) -> tuple[float, float]:
    if vec[1] < -1e-10 or (abs(vec[1]) < 1e-10 and vec[0] < 0):
        vec = -vec
    return (float(np.round(vec[0], 8)), float(np.round(vec[1], 8)))


def _extract_bonds(graph):
    j1_bonds = sorted({tuple(sorted((i, j))) for i, j, c in graph.edges(return_color=True) if c == 0})
    j2_bonds = sorted({tuple(sorted((i, j))) for i, j, c in graph.edges(return_color=True) if c == 1})
    return j1_bonds, j2_bonds


def _find_hexagons(j1_bonds, n_sites: int):
    adjacency = defaultdict(set)
    for i, j in j1_bonds:
        adjacency[i].add(j)
        adjacency[j].add(i)

    def canonical_cycle(cycle):
        seqs = []
        for seq in (list(cycle), list(reversed(cycle))):
            for shift in range(len(seq)):
                rotated = tuple(seq[shift:] + seq[:shift])
                seqs.append(rotated)
        return min(seqs)

    cycles = set()

    def dfs(start, current, path):
        if len(path) == 6:
            if start in adjacency[current]:
                cycles.add(canonical_cycle(path))
            return
        for nxt in adjacency[current]:
            if nxt in path:
                continue
            dfs(start, nxt, path + [nxt])

    for start in range(n_sites):
        dfs(start, start, [start])

    return sorted(cycles)


def _reciprocal_basis(basis_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reciprocal_columns = 2.0 * np.pi * np.linalg.inv(basis_vectors)
    return reciprocal_columns[:, 0], reciprocal_columns[:, 1]


def _allowed_q_points(meta):
    extent = meta["extent"]
    b1, b2 = _reciprocal_basis(meta["basis_vectors"])
    q_points = []
    for n1 in range(extent[0]):
        for n2 in range(extent[1]):
            q_vec = n1 * b1 / extent[0] + n2 * b2 / extent[1]
            q_points.append(((int(n1), int(n2)), q_vec))
    return q_points


def _bond_center(meta, bond):
    i, j = bond
    return meta["positions"][i] + 0.5 * _minimal_image_displacement(meta, i, j)


def _group_values_by_distance(entries, key_name: str, value_name: str):
    grouped = defaultdict(list)
    for item in entries:
        grouped[_distance_shell(item[key_name])].append(item[value_name])
    return [
        {
            "distance": dist,
            "mean": float(np.mean(values)),
            "count": len(values),
        }
        for dist, values in sorted(grouped.items())
    ]


def define_observables(num_sites, hi, graph):
    basic_ops = {
        "<X>": sum(nk.operator.spin.sigmax(hi, i) for i in range(hi.size)) / (2 * num_sites),
        "<Y>": sum(nk.operator.spin.sigmay(hi, i) for i in range(hi.size)) / (2 * num_sites),
        "<Z>": sum(nk.operator.spin.sigmaz(hi, i) for i in range(hi.size)) / (2 * num_sites),
    }

    for i in range(num_sites):
        basic_ops[f"X_{i}"] = nk.operator.spin.sigmax(hi, i) / 2
        basic_ops[f"Y_{i}"] = nk.operator.spin.sigmay(hi, i) / 2
        basic_ops[f"Z_{i}"] = nk.operator.spin.sigmaz(hi, i) / 2

    return {
        "basic_ops": basic_ops,
        "hi": hi,
        "graph": graph,
        "num_sites": num_sites,
    }


def calculate_expectations(vstate, hamiltonian, observable_bundle):
    hi = observable_bundle["hi"]
    graph = observable_bundle["graph"]
    num_sites = observable_bundle["num_sites"]
    basic_ops = observable_bundle["basic_ops"]
    meta = _graph_metadata(graph)
    j1_bonds, _j2_bonds = _extract_bonds(graph)
    hexagons = _find_hexagons(j1_bonds, num_sites)

    energy_stats = vstate.expect(hamiltonian)
    energy_raw = complex(energy_stats.mean)
    energy_spin = energy_raw / 4.0
    print(f"Energy mean value (raw/Pauli convention) = {energy_raw}")
    print(f"Energy mean value per site (raw/Pauli convention) = {energy_raw / num_sites}")
    print(f"Energy mean value (spin convention) = {energy_spin}")
    print(f"Energy mean value per site (spin convention) = {energy_spin / num_sites}")

    expectation_values = {
        "Energy": energy_raw,
        "Energy_per_site": energy_raw / num_sites,
        "Energy_spin": energy_spin,
        "Energy_per_site_spin": energy_spin / num_sites,
    }
    for key, operator in basic_ops.items():
        stats = vstate.expect(operator)
        expectation_values[key] = complex(stats.mean)
        print(f"{key} mean value = {stats.mean}")

    spin_corr = np.zeros((num_sites, num_sites), dtype=np.complex128)
    for i in range(num_sites):
        for j in range(i, num_sites):
            stats = vstate.expect(_spin_dot_operator(hi, i, j))
            value = complex(stats.mean)
            spin_corr[i, j] = value
            spin_corr[j, i] = value

    pair_entries = []
    for i in range(num_sites):
        for j in range(num_sites):
            disp = _minimal_image_displacement(meta, i, j)
            pair_entries.append(
                {
                    "i": i,
                    "j": j,
                    "distance": float(np.linalg.norm(disp)),
                    "value": float(np.real(spin_corr[i, j])),
                }
            )

    site0_corr = [item for item in pair_entries if item["i"] == 0]
    corr_by_distance = _group_values_by_distance(pair_entries, "distance", "value")
    site0_component_corr = []
    for j in range(1, num_sites):
        xx = complex(vstate.expect(_spin_component_component_operator(hi, "x", 0, j)).mean)
        yy = complex(vstate.expect(_spin_component_component_operator(hi, "y", 0, j)).mean)
        zz = complex(vstate.expect(_spin_component_component_operator(hi, "z", 0, j)).mean)
        site0_component_corr.append(
            {
                "j": j,
                "XX": {"real": float(np.real(xx)), "imag": float(np.imag(xx))},
                "YY": {"real": float(np.real(yy)), "imag": float(np.imag(yy))},
                "ZZ": {"real": float(np.real(zz)), "imag": float(np.imag(zz))},
            }
        )

    eta = meta["staggered_sign"]
    neel_sf = float(np.real(np.sum((eta[:, None] * eta[None, :]) * spin_corr) / num_sites))

    q_entries = []
    q_points = _allowed_q_points(meta)
    for (n1, n2), q_vec in q_points:
        spin_sf = 0.0 + 0.0j
        neel_q = 0.0 + 0.0j
        for i in range(num_sites):
            for j in range(num_sites):
                phase = np.exp(1j * float(np.dot(q_vec, _minimal_image_displacement(meta, i, j))))
                spin_sf += phase * spin_corr[i, j]
                neel_q += eta[i] * eta[j] * phase * spin_corr[i, j]
        spin_sf = spin_sf / num_sites
        neel_q = neel_q / num_sites
        q_entries.append(
            {
                "q_index": [n1, n2],
                "q_vector": [float(q_vec[0]), float(q_vec[1])],
                "spin_structure_factor": float(np.real(spin_sf)),
                "neel_structure_factor": float(np.real(neel_q)),
            }
        )

    spin_peak = max(q_entries, key=lambda item: item["spin_structure_factor"])
    neel_peak = max(q_entries, key=lambda item: item["neel_structure_factor"])

    extent = meta["extent"]
    nonzero_q = [
        item
        for item in q_entries
        if item["q_index"] != [0, 0]
    ]
    q_min_entry = min(nonzero_q, key=lambda item: np.linalg.norm(item["q_vector"]))
    q_min_norm = float(np.linalg.norm(q_min_entry["q_vector"]))
    neel_corr_length = None
    if q_min_entry["neel_structure_factor"] > 0.0 and neel_sf > q_min_entry["neel_structure_factor"]:
        neel_corr_length = float(
            np.sqrt(neel_sf / q_min_entry["neel_structure_factor"] - 1.0)
            / (2.0 * np.sin(q_min_norm / 2.0))
        )

    bond_values = {}
    for bond in j1_bonds:
        stats = vstate.expect(_spin_dot_operator(hi, bond[0], bond[1]))
        bond_values[bond] = complex(stats.mean)

    orientation_vectors = sorted(
        {
            _canonical_bond_vector(_minimal_image_displacement(meta, i, j))
            for i, j in j1_bonds
        }
    )
    orientation_index = {vec: idx for idx, vec in enumerate(orientation_vectors)}

    local_complex_bond_order = []
    for site in range(num_sites):
        if meta["sublattice"][site] != 0:
            continue
        outgoing = []
        for bond in j1_bonds:
            if site not in bond:
                continue
            other = bond[1] if bond[0] == site else bond[0]
            vec = _canonical_bond_vector(_minimal_image_displacement(meta, site, other))
            outgoing.append((orientation_index[vec], bond_values[tuple(sorted((site, other)))]))
        outgoing.sort(key=lambda item: item[0])
        if len(outgoing) == 3:
            omega = cmath.exp(2j * np.pi / 3.0)
            psi = outgoing[0][1] + omega * outgoing[1][1] + (omega**2) * outgoing[2][1]
            local_complex_bond_order.append(psi)

    plaquette_order_parameter_abs_mean = None
    plaquette_order_parameter_complex_mean = None
    if local_complex_bond_order:
        plaquette_order_parameter_abs_mean = float(np.mean(np.abs(local_complex_bond_order)))
        plaquette_order_parameter_complex_mean = complex(np.mean(local_complex_bond_order))

    hexagon_energies = []
    for hexagon in hexagons:
        bonds = [tuple(sorted((hexagon[k], hexagon[(k + 1) % 6]))) for k in range(6)]
        value = sum(bond_values[bond] for bond in bonds)
        hexagon_energies.append(
            {
                "sites": list(hexagon),
                "value": float(np.real(value)),
            }
        )
    plaquette_energy_rms = None
    if hexagon_energies:
        plaquette_energy_rms = float(np.std([item["value"] for item in hexagon_energies]))

    reference_bond = j1_bonds[0] if j1_bonds else None
    dimer_entries = []
    dimer_by_distance = []
    if reference_bond is not None:
        ref_value = bond_values[reference_bond]
        ref_center = _bond_center(meta, reference_bond)
        ref_op = _spin_dot_operator(hi, reference_bond[0], reference_bond[1])
        for bond in j1_bonds:
            op = ref_op @ _spin_dot_operator(hi, bond[0], bond[1])
            stats = vstate.expect(op)
            full_value = complex(stats.mean)
            connected = full_value - ref_value * bond_values[bond]
            center = _bond_center(meta, bond)
            distance = float(np.linalg.norm(center - ref_center))
            dimer_entries.append(
                {
                    "bond": list(bond),
                    "distance": distance,
                    "value": float(np.real(connected)),
                }
            )
        dimer_by_distance = _group_values_by_distance(dimer_entries, "distance", "value")

    expectation_values.update(
        {
            "spin_spin_correlation_matrix_real": spin_corr.real.tolist(),
            "spin_spin_correlation_matrix_pauli_real": (4.0 * spin_corr.real).tolist(),
            "spin_spin_correlation_site0": site0_corr,
            "spin_spin_correlation_site0_components": site0_component_corr,
            "spin_spin_correlation_by_distance": corr_by_distance,
            "neel_structure_factor": neel_sf,
            "neel_structure_factor_pauli": 4.0 * neel_sf,
            "static_spin_structure_factor": q_entries,
            "static_spin_structure_factor_pauli": [
                {
                    "q_index": item["q_index"],
                    "q_vector": item["q_vector"],
                    "spin_structure_factor": 4.0 * item["spin_structure_factor"],
                    "neel_structure_factor": 4.0 * item["neel_structure_factor"],
                }
                for item in q_entries
            ],
            "static_spin_structure_factor_peak": spin_peak,
            "static_spin_structure_factor_peak_pauli": {
                "q_index": spin_peak["q_index"],
                "q_vector": spin_peak["q_vector"],
                "spin_structure_factor": 4.0 * spin_peak["spin_structure_factor"],
                "neel_structure_factor": 4.0 * spin_peak["neel_structure_factor"],
            },
            "neel_structure_factor_peak": neel_peak,
            "neel_structure_factor_peak_pauli": {
                "q_index": neel_peak["q_index"],
                "q_vector": neel_peak["q_vector"],
                "spin_structure_factor": 4.0 * neel_peak["spin_structure_factor"],
                "neel_structure_factor": 4.0 * neel_peak["neel_structure_factor"],
            },
            "correlation_length_estimates": {
                "neel": neel_corr_length,
                "q_min": {
                    "q_index": q_min_entry["q_index"],
                    "q_vector": q_min_entry["q_vector"],
                },
            },
            "nearest_neighbor_bond_energies": [
                {
                    "bond": list(bond),
                    "value": float(np.real(value)),
                }
                for bond, value in bond_values.items()
            ],
            "nearest_neighbor_bond_energies_pauli": [
                {
                    "bond": list(bond),
                    "value": 4.0 * float(np.real(value)),
                }
                for bond, value in bond_values.items()
            ],
            "plaquette_order_parameter_abs_mean": plaquette_order_parameter_abs_mean,
            "plaquette_order_parameter_complex_mean": plaquette_order_parameter_complex_mean,
            "plaquette_hexagon_energies": hexagon_energies,
            "plaquette_hexagon_energies_pauli": [
                {
                    "sites": item["sites"],
                    "value": 4.0 * item["value"],
                }
                for item in hexagon_energies
            ],
            "plaquette_energy_rms": plaquette_energy_rms,
            "plaquette_energy_rms_pauli": None if plaquette_energy_rms is None else 4.0 * plaquette_energy_rms,
            "dimer_dimer_reference_bond": list(reference_bond) if reference_bond is not None else None,
            "dimer_dimer_correlations": dimer_entries,
            "dimer_dimer_correlations_pauli": [
                {
                    "bond": item["bond"],
                    "distance": item["distance"],
                    "value": 16.0 * item["value"],
                }
                for item in dimer_entries
            ],
            "dimer_dimer_correlations_by_distance": dimer_by_distance,
            "dimer_dimer_correlations_by_distance_pauli": [
                {
                    "distance": item["distance"],
                    "mean": 16.0 * item["mean"],
                    "count": item["count"],
                }
                for item in dimer_by_distance
            ],
        }
    )

    print(f"Néel structure factor (spin convention) = {neel_sf}")
    print(f"Néel structure factor (Pauli convention) = {4.0 * neel_sf}")
    if neel_corr_length is not None:
        print(f"Néel correlation length estimate = {neel_corr_length}")
    if plaquette_order_parameter_abs_mean is not None:
        print(f"Plaquette/bond-order parameter |psi| mean = {plaquette_order_parameter_abs_mean}")

    return expectation_values
