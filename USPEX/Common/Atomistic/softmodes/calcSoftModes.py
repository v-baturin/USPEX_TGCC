from __future__ import division

from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure
from ..Bonds import Bonds

import numpy as np


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


def calcSoftModes(system : AtomicStructure, R_val, N_val, val, kVector0=np.zeros(3)):
    '''
    The function calculates vibrational modes based on the dynamic matrix (D) constructed from bond hardness model.

    K-vector should be in A^-1, very important!
        reciprocal_lat_x = 2pi*(lat_y X lat_z)/V
        k_abs = k*reciprocal_lattice

    :param system: 
    :param R_val: radii.
    :param N_val: number of valence electrons.
    :param val: valence.
    :param kVector0: K-vector.
    :return freq: frequencies of all modes.
    :return eigvector: eigenvector of all modes.
    '''

    assert isinstance(system.bonds, Bonds)

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
    bond_groups = system.bonds.all_types()    # list of bond group
    N_group = len(bond_groups)           # number of bond group
    N_atom = len(system)  # number of atoms

    # Initiallization of Dynamic matrix (3N*3N):
    D = np.zeros((3 * N_atom, 3 * N_atom))

    # Calculate bond valence using classical Brown's bond valence model.
    # nu_factor should be normalized to satisfy sum rule.
    nu_factor = np.zeros(N_atom)

    for k in range(N_atom):
        nu_full = 0.0
        for bond in system.bonds:
            a, b = bond.atoms()
            if a == k:
                nu_full += np.exp(-bond.delta / 0.37)
            if b == k:
                nu_full += np.exp(-bond.delta / 0.37)
        nu_factor[k] = val[system.atom_type_seq[k]] / nu_full

    for bondtype in bond_groups:
        bonds = system.bonds.getType(bondtype)

        for bond in bonds:
            ID1, ID2 = bond.atoms()
            a = system.atom_type_seq[ID1]
            b = system.atom_type_seq[ID2]
            R = R_val[a] + R_val[b] + bond.delta
            R_val_sum = R_val[a] + R_val[b]
            R_a = R_val[a]/R_val_sum * R
            R_b = R_val[b]/R_val_sum * R
            nu = np.exp(-bond.delta / 0.37)
            EN_a = 0.481 * N_val[a] / R_a
            EN_b = 0.481 * N_val[b] / R_b
            CN_a = val[a] / (nu * nu_factor[ID1])
            CN_b = val[b] / (nu * nu_factor[ID2])

            f_ab = 0.25 * np.abs(EN_a - EN_b) / np.sqrt(EN_a * EN_b)
            X_ab = np.sqrt(EN_a * EN_b / (CN_a * CN_b))

            vect = coords[ID1, :] - coords[ID2, :] - bond.direction
            dR_ab = np.dot(vect, lat)

            C = np.round(dR_ab / np.linalg.norm(dR_ab) * 1000000) / 1000000
            phase_k1 = np.real(np.exp(1j * np.dot(kVector, dR_ab)))
            phase_k2 = np.real(np.exp(1j * np.dot(kVector, -dR_ab)))
            H = X_ab * np.exp(-2.7 * f_ab)

            if ID1 == ID2:
                if not np.isclose(bond.direction, np.zeros(3)).all():
                    D = AddDynMatSelf(D, ID1, H, C, phase_k1)
            else:
                D = AddDynMat(D, ID1, ID2, H, C, phase_k1, phase_k2)

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
