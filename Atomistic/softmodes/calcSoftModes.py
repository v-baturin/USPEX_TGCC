from __future__ import division

from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure
from .getMinimalGraphBonds import getMinimalGraphBonds

from USPEX.Common.Atomistic.Element import Element



import numpy as np
from itertools import chain


R_val = lambda symbol: Element(symbol).covalent_radius


def AddDynMatSelf(D, a, H, Cos, phase):
    C = H * np.array([
        [Cos[0] * Cos[0], Cos[0] * Cos[1], Cos[0] * Cos[2]],
        [Cos[1] * Cos[0], Cos[1] * Cos[1], Cos[1] * Cos[2]],
        [Cos[2] * Cos[0], Cos[2] * Cos[1], Cos[2] * Cos[2]],
    ])

    D[a * 3: (a + 1) * 3, a * 3: (a + 1) * 3] += C * (1.0 - phase)

    return D


def AddDynMat(D, a, b, H, Cos, phase1, phase2):
    C = H * np.array([
        [Cos[0] * Cos[0], Cos[0] * Cos[1], Cos[0] * Cos[2]],
        [Cos[1] * Cos[0], Cos[1] * Cos[1], Cos[1] * Cos[2]],
        [Cos[2] * Cos[0], Cos[2] * Cos[1], Cos[2] * Cos[2]],
    ])

    D[a * 3:(a + 1) * 3, a * 3:(a + 1) * 3] += C
    D[a * 3:(a + 1) * 3, b * 3:(b + 1) * 3] -= (C * phase1)
    D[b * 3:(b + 1) * 3, a * 3:(a + 1) * 3] -= (C * phase2)
    D[b * 3:(b + 1) * 3, b * 3:(b + 1) * 3] += C

    return D


# R_val : dict, N_val : dict, val : dict,
def calcSoftModes(system : AtomicStructure, config, kVector0=np.zeros(3)):
    '''
    The function calculates vibrational modes based on the dynamic matrix (D) constructed from bond hardness model.

    K-vector should be in A^-1, very important!
        reciprocal_lat_x = 2pi*(lat_y X lat_z)/V
        k_abs = k*reciprocal_lattice

    :param system: 
    :param kVector0: K-vector.
    :return freq: frequencies of all modes.
    :return eigvector: eigenvector of all modes.
    '''

    bonds = getMinimalGraphBonds(system, config.goodBonds)

    # assert isinstance(system.bonds, Bonds)

    # Convert everything to ndarray:
    lat = system.cell
    coords = system.scaled_coordinates

    rec_lat = np.zeros((3, 3))
    det_lat = np.linalg.det(lat)
    rec_lat[0, :] = 2.0 * np.pi * np.cross(lat[1, :], lat[2, :]) / det_lat
    rec_lat[1, :] = 2.0 * np.pi * np.cross(lat[2, :], lat[0, :]) / det_lat
    rec_lat[2, :] = 2.0 * np.pi * np.cross(lat[0, :], lat[1, :]) / det_lat
    kVector = np.dot(kVector0, rec_lat)

    # Obtain the bond information:
    N_atom = len(system)  # number of atoms

    # Initiallization of Dynamic matrix (3N*3N):
    D = np.zeros((3 * N_atom, 3 * N_atom))

    # Calculate bond valence using classical Brown's bond valence model.
    # nu_factor should be normalized to satisfy sum rule.
    nu_factor = []

    # atomTypes, atom_type_seq = atomTypeCounter(system.chemicalSymbols)
    for k, symbol in enumerate(system.chemicalSymbols):
        nu_full = 0.0

        for bond in chain(*bonds):  # how many type of bonds
            a, b = bond.indicies
            if a == k:
                nu_full += np.exp(-bond.delta / 0.37)
            if b == k:
                nu_full += np.exp(-bond.delta / 0.37)
        nu_factor.append(config.valences[symbol] / nu_full)

    for bond_group in bonds:
        for bond in bond_group:
            s1, s2 = bond.symbols
            i1, i2 = bond.indicies
            # a = atom_type_seq[ID1]
            # b = atom_type_seq[ID2]
            R = R_val(s1) + R_val(s2) + bond.delta
            R_val_sum = R_val(s1) + R_val(s2)
            R_a = R_val(s1)/R_val_sum * R
            R_b = R_val(s2)/R_val_sum * R
            nu = np.exp(-bond.delta / 0.37)
            EN_a = 0.481 * config.valenceElectrons[s1] / R_a
            EN_b = 0.481 * config.valenceElectrons[s2] / R_b
            CN_a = config.valences[s1] / (nu * nu_factor[i1])
            CN_b = config.valences[s2] / (nu * nu_factor[i2])

            f_ab = 0.25 * np.abs(EN_a - EN_b) / np.sqrt(EN_a * EN_b)
            X_ab = np.sqrt(EN_a * EN_b / (CN_a * CN_b))

            C = np.round(bond.vector / np.linalg.norm(bond.vector) * 1000000) / 1000000
            phase_k1 = np.real(np.exp(1j * np.dot(kVector, bond.vector)))
            phase_k2 = np.real(np.exp(1j * np.dot(kVector, -bond.vector)))
            H = X_ab * np.exp(-2.7 * f_ab)

            if i1 == i2:
                if not np.allclose(bond.direction, [0,0,0]):
                    D = AddDynMatSelf(D, i1, H, C, phase_k1)
            else:
                D = AddDynMat(D, i1, i2, H, C, phase_k1, phase_k2)

    # Compare D from Matlab and Python:
    '''
    from lib.mat2dict import loadmat
    D_matlab = loadmat('test_SoftModes2/D_POSCAR_1_supercell=2.mat')['D']

    for i in range(D.shape[0]):
        print '%4i: %12.8f' % (i, np.max(np.abs(D[i, :] - D_matlab[i, :])))
    '''

    # Eigenvectors have returned as columns.
    freq, eigvector = np.linalg.eig(D)
    freq = np.real(freq)
    IX = freq.argsort()
    freq, eigvector = freq[IX], np.real(eigvector[:, IX])

    return freq, eigvector
