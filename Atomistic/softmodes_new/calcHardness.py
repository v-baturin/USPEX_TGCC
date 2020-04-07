

import numpy as np

from scipy.stats import gmean
from itertools import chain

from USPEX.Common.Atomistic.AtomisticConfig import AtomisticConfig
from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure
from USPEX.Common.Atomistic.Element import Element
#from .BondHardness import BondHardness
from .BondHardness_new import BondHardness_new
#from .AtomTypeCounter import atomTypeCounter

_MAX_CELL_LENGTH = 4   # max length of any cell length


# def calcHardness(CONFIG : AtomisticConfig, _system : AtomicStructure) -> float:
#     '''
#     Calculate hardness for a given structure from bond hardness model.
#     See http://han.ess.sunysb.edu/hardness/ for details.
#
#     :param config:
#     :param system:
#     :return H: hardness (GPa).
#     '''
#
#     system = AtomicStructure(symbols=_system.chemicalSymbols, positions=_system.coordinates, cell=_system.cell)
#
#     atomTypes, atom_type_seq = atomTypeCounter(system.chemicalSymbols)
#     R_val = [Element(type).covalent_radius for type in atomTypes]
#
#     bonds = BondHardness(system, CONFIG.goodBonds)
#     bond_group = bonds.all_types()
#     N_group = len(bond_group)  # number of bond group
#     N_atoms = len(system)  # number of atoms
#
#
#     # nu_factor should be normalized to satisfy sum rule.
#     nu_factor = np.zeros(N_atoms)
#
#     for k in range(N_atoms):
#         nu_full = 0.0
#
#         for bond in bonds:  # how many type of bonds
#             a, b = bond.atoms()
#             if a == k:
#                 nu_full += np.exp(-bond.delta / 0.37)
#             if b == k:
#                 nu_full += np.exp(-bond.delta / 0.37)
#         nu_factor[k] = CONFIG.valences[atom_type_seq[k]] / nu_full
#
#     '''
#     Apply the bond hardness model here. Two for loops here:
#         - outer loop goes through all different bond groups;
#         - inner loop goes though all individual bonds in a given group.
#     Inner loop can take arithmetic/geometric average.
#     Outer loop must use geometric average.
#     '''
#
#     H = 1.0
#
#     for bondtype in bond_group:
#         tmp_bonds = bonds.getType(bondtype)
#         n = len(tmp_bonds)
#         h_tmp = 1.0
#
#         if n > 0:
#             h_tmp1 = []
#             for bond in tmp_bonds:
#                 a1, b1 = bond.atoms()
#                 a = atom_type_seq[a1]  # atom1 in the bond
#                 b = atom_type_seq[b1]  # atom2 in the bond
#
#                 R_a = R_val[a] + bond.delta / 2
#                 R_b = R_val[b] + bond.delta / 2
#                 nu = np.exp(-bond.delta / 0.37)
#                 EN_a = 0.481 * CONFIG.valenceElectrons[a] / R_a  # electronegativity
#                 EN_b = 0.481 * CONFIG.valenceElectrons[b] / R_b
#
#                 # Effective CN that describes the atomic valence:
#                 CN_a = CONFIG.valences[a] / (nu * nu_factor[a1])
#                 CN_b = CONFIG.valences[b] / (nu * nu_factor[b1])
#
#                 f_ab = 0.25 * abs(EN_a - EN_b) / np.sqrt(EN_a * EN_b)  # ionicity indicator
#                 X_ab = np.sqrt(EN_a * EN_b / (CN_a * CN_b))  # electron-holding energy
#                 h_tmp1.append(X_ab * np.exp(-2.7 * f_ab))
#             h_tmp = n * gmean(h_tmp1)  # geometric average
#         H = H * h_tmp
#
#     # Final equation:
#     H = 423.8 * N_group * (H ** (1.0 / N_group)) / system.volume - 3.4
#     return H

def calcHardness_new(CONFIG : AtomisticConfig, system : AtomicStructure) -> float:
    '''
    Calculate hardness for a given structure from bond hardness model.
    See http://han.ess.sunysb.edu/hardness/ for details.

    :param CONFIG:
    :param system:
    :return H: hardness (GPa).
    '''

    # TODO this is hardcode. Really, it's better to check whether we have very shrink lattice parameters (at least 1)
    # and then make supercell in a right direction.
    # a, b, c = _system.cell_lengths_and_angles[:3]
    # m = np.ones(3, dtype=int)
    # if a < MAX_CELL_LENGTH:
    #     m[0] = 2
    # if b < MAX_CELL_LENGTH:
    #     m[1] = 2
    # if c < MAX_CELL_LENGTH:
    #     m[2] = 2
    # coor, lat = optLattice(_system.coordinates, _system.cell)
    # system = AtomicStructure(symbols=_system.chemicalSymbols, positions=coor, cell=lat)
    # system *= m

    bonds = BondHardness_new(system, CONFIG.goodBonds)

    # Calculate bond valence using classical Brown's bond valence model.
    # nu_factor should be normalized to satisfy sum rule.
    nu_factor = []

    for k, symbol in enumerate(system.chemicalSymbols):
        nu_full = 0.0

        for bond in chain(*bonds):  # how many type of bonds
            a, b = bond.indicies
            if a == k:
                nu_full += np.exp(-bond.delta / 0.37)
            if b == k:
                nu_full += np.exp(-bond.delta / 0.37)
        nu_factor.append(CONFIG.valences[symbol] / nu_full)

    '''
    Apply the bond hardness model here. Two for loops here:
        - outer loop goes through all different bond groups;
        - inner loop goes though all individual bonds in a given group.
    Inner loop can take arithmetic/geometric average.
    Outer loop must use geometric average.
    '''

    H = 1.0

    for tmp_bonds in bonds:
        h_tmp = 1.0

        if tmp_bonds:
            h_tmp1 = []
            for bond in tmp_bonds:
                a, b = bond.symbols

                R_a = Element(a).covalent_radius + bond.delta / 2
                R_b = Element(b).covalent_radius + bond.delta / 2
                nu = np.exp(-bond.delta / 0.37)
                EN_a = 0.481 * CONFIG.valenceElectrons[a] / R_a  # electronegativity
                EN_b = 0.481 * CONFIG.valenceElectrons[b] / R_b

                # Effective CN that describes the atomic valence:
                a1, b1 = bond.indicies
                CN_a = CONFIG.valences[a] / (nu * nu_factor[a1])
                CN_b = CONFIG.valences[b] / (nu * nu_factor[b1])

                f_ab = 0.25 * abs(EN_a - EN_b) / np.sqrt(EN_a * EN_b)  # ionicity indicator
                X_ab = np.sqrt(EN_a * EN_b / (CN_a * CN_b))  # electron-holding energy
                h_tmp1.append(X_ab * np.exp(-2.7 * f_ab))
            h_tmp = len(tmp_bonds) * gmean(h_tmp1)  # geometric average
        H = H * h_tmp

    # Final equation:
    H = 423.8 * len(bonds) * (H ** (1.0 / len(bonds))) / system.volume - 3.4

    return H
