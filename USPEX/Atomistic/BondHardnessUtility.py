"""
USPEX.Atomistic.BondHardnessUtility
===================================
"""

import logging
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import gmean
from itertools import chain

logger = logging.getLogger(__name__)

closest = np.array(
    [[[0, 0, 0]], [[-1, 0, 0]], [[-1, 0, -1]], [[-1, -1, -1]], [[-1, -1, 0]], [[0, -1, 0]], [[0, -1, -1]],
     [[0, 0, -1]], [[-1, -1, 1]], [[-1, 0, 1]], [[-1, 1, 1]], [[-1, 1, 0]], [[-1, 1, -1]], [[0, -1, 1]],
     [[0, 0, 1]], [[0, 1, 1]], [[0, 1, 0]], [[0, 1, -1]], [[1, 0, 0]], [[1, 0, -1]], [[1, -1, -1]],
     [[1, -1, 0]], [[1, -1, 1]], [[1, 0, 1]], [[1, 1, 1]], [[1, 1, 0]], [[1, 1, -1]]])


class BondHardnessUtility:

    atomType = None

    @classmethod
    def registerTypes(cls, atomType):
        """
        Register types used by this utility.

        :param structureType: type representing atomic structure.
        :param atomType: type representing chemical element.
        :param cellType: type representing unit cell.
        :param atomicDisassemblerType: type representing utility used for disassembling structure into molecules.
        """
        cls.atomType = atomType

    def __init__(self):
        pass

    def calcHardness(self, structure, bonds):
        '''
        Calculate hardness for a given structure from bond hardness model.
        See http://han.ess.sunysb.edu/hardness/ for details.

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

        # Calculate bond valence using classical Brown's bond valence model.
        # nu_factor should be normalized to satisfy sum rule.
        nu_factor = []

        for k, symbol in enumerate(structure.getAtomTypes()):
            nu_full = 0.0

            for bond in chain(*bonds):  # how many type of bonds
                a, b = bond.indicies
                if a == k:
                    nu_full += np.exp(-bond.delta / 0.37)
                if b == k:
                    nu_full += np.exp(-bond.delta / 0.37)
            nu_factor.append(symbol.valence / nu_full)

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

                    R_a = self.atomType(a).covalent_radius + bond.delta / 2
                    R_b = self.atomType(b).covalent_radius + bond.delta / 2
                    nu = np.exp(-bond.delta / 0.37)
                    EN_a = 0.481 * self.atomType(a).valence_electrons / R_a  # electronegativity
                    EN_b = 0.481 * self.atomType(b).valence_electrons / R_b

                    # Effective CN that describes the atomic valence:
                    a1, b1 = bond.indicies
                    CN_a = self.atomType(a).valence / (nu * nu_factor[a1])
                    CN_b = self.atomType(b).valence / (nu * nu_factor[b1])

                    f_ab = 0.25 * abs(EN_a - EN_b) / np.sqrt(EN_a * EN_b)  # ionicity indicator
                    X_ab = np.sqrt(EN_a * EN_b / (CN_a * CN_b))  # electron-holding energy
                    h_tmp1.append(X_ab * np.exp(-2.7 * f_ab))
                h_tmp = len(tmp_bonds) * gmean(h_tmp1)  # geometric average
            H = H * h_tmp

        # Final equation:
        H = 423.8 * len(bonds) * (H ** (1.0 / len(bonds))) / structure.getCell().getVolume() - 3.4

        return H

    def calcSoftModes(self, system, bonds, kVector0=np.zeros(3)):
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

        # assert isinstance(system.bonds, Bonds)

        # Convert everything to ndarray:
        lat = system.getCell().getCellVectors()
        # coords = system.getFractionalCoordinates()

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
        for k, symbol in enumerate(system.getAtomTypes()):
            nu_full = 0.0

            for bond in chain(*bonds):  # how many type of bonds
                a, b = bond.indicies
                if a == k:
                    nu_full += np.exp(-bond.delta / 0.37)
                if b == k:
                    nu_full += np.exp(-bond.delta / 0.37)
            nu_factor.append(symbol.valence / nu_full)

        for bond_group in bonds:
            for bond in bond_group:
                s1, s2 = bond.symbols
                i1, i2 = bond.indicies
                e1, e2 = self.atomType(s1), self.atomType(s2)
                # a = atom_type_seq[ID1]
                # b = atom_type_seq[ID2]
                R_val_sum = e1.covalent_radius + e2.covalent_radius
                R = R_val_sum + bond.delta
                R_a = e1.covalent_radius / R_val_sum * R
                R_b = e2.covalent_radius / R_val_sum * R
                nu = np.exp(-bond.delta / 0.37)
                EN_a = 0.481 * self.atomType(s1).valence_electrons / R_a
                EN_b = 0.481 * self.atomType(s2).valence_electrons / R_b
                CN_a = self.atomType(s1).valence / (nu * nu_factor[i1])
                CN_b = self.atomType(s2).valence / (nu * nu_factor[i2])

                f_ab = 0.25 * np.abs(EN_a - EN_b) / np.sqrt(EN_a * EN_b)
                X_ab = np.sqrt(EN_a * EN_b / (CN_a * CN_b))

                C = np.round(bond.vector / np.linalg.norm(bond.vector) * 1000000) / 1000000
                phase_k1 = np.real(np.exp(1j * np.dot(kVector, bond.vector)))
                phase_k2 = np.real(np.exp(1j * np.dot(kVector, -bond.vector)))
                H = X_ab * np.exp(-2.7 * f_ab)

                if i1 == i2:
                    if not np.allclose(bond.direction, [0, 0, 0]):
                        D = _AddDynMatSelf(D, i1, H, C, phase_k1)
                else:
                    D = _AddDynMat(D, i1, i2, H, C, phase_k1, phase_k2)

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
        freq, eigvector = list(freq[IX]), list(np.real(eigvector[:, IX]).T)

        return freq, eigvector

    @staticmethod
    def calcCoordinationNumbers(structure):
        radii = np.tile([element.covalent_radius for element in structure.getAtomTypes()], len(closest))
        base = radii.reshape(1, radii.size) + radii.reshape(radii.size, 1)
        vertices = np.dot(np.concatenate((closest + structure.getFractionalCoordinates()), axis=0),
                          structure.getCell().getCellVectors())
        order = np.exp(-(cdist(vertices, vertices) - base) / 0.23)
        order = np.delete(np.triu(order, 1), 0, 1) + np.delete(np.tril(order, -1), len(vertices) - 1, 1)
        return order.sum(axis=1) / order.max(axis=1)


def _AddDynMatSelf(D, a, H, Cos, phase):
    C = H * np.array([
        [Cos[0] * Cos[0], Cos[0] * Cos[1], Cos[0] * Cos[2]],
        [Cos[1] * Cos[0], Cos[1] * Cos[1], Cos[1] * Cos[2]],
        [Cos[2] * Cos[0], Cos[2] * Cos[1], Cos[2] * Cos[2]],
    ])

    D[a * 3: (a + 1) * 3, a * 3: (a + 1) * 3] += C * (1.0 - phase)

    return D


def _AddDynMat(D, a, b, H, Cos, phase1, phase2):
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

# TODO in case of nonzero kVector return eigenVectors for supercel
'''
function [eigVector, coords, lat, numIons] = calcEigenvectorK(eigVectorK, supercell, lat0, coords0, numIons0)

% creates the proper eigenvector out of 'k' one for varcomp

N = size(coords0, 1);
k1 = supercell(1);
k2 = supercell(2);
k3 = supercell(3);
kVector = zeros(1,3);
if k1 > 1
    kVector(1) = 1/k1;
end
if k2 > 1
    kVector(2) = 1/k2;
end
if k3 > 1
    kVector(3) = 1/k3;
end

numIons = numIons0*k1*k2*k3;
lat(1,:) = lat0(1,:)*k1;
lat(2,:) = lat0(2,:)*k2;
lat(3,:) = lat0(3,:)*k3;
coords1 = coords0; % build first mini-cell
for a = 1 : N
    coords1(a,1) = coords0(a,1)/k1;
    coords1(a,2) = coords0(a,2)/k2;
    coords1(a,3) = coords0(a,3)/k3;
end
coords = zeros(k1*k2*k3*N,3);
for i = 1 : k1
    for j = 1 : k2
        for k = 1 : k3
            for a = 1 : N
                ind = (i-1)*k2*k3*N + (j-1)*k3*N + (k-1)*N + a;
                coords(ind, 1) = coords1(a,1) + (i-1)/k1;
                coords(ind, 2) = coords1(a,2) + (j-1)/k2;
                coords(ind, 3) = coords1(a,3) + (k-1)/k3;
            end
        end
    end
end

% make real eigenvectors (or rather displacements) out of 'k' ones
% it looks like we can simply take the sign of the real part
rec_lat = zeros(3,3);
rec_lat(1,:) = 2*pi*cross(lat0(2,:), lat0(3,:))/det(lat0);
rec_lat(2,:) = 2*pi*cross(lat0(3,:), lat0(1,:))/det(lat0);
rec_lat(3,:) = 2*pi*cross(lat0(1,:), lat0(2,:))/det(lat0);
kVectorA = kVector*rec_lat;
coordsA = coords*lat;
eigVector = zeros(3*N*k1*k2*k3, 3*N);
col = 1;
for eVec = 1 : size(eigVectorK,2)
    eigenV = eigVectorK(:,eVec);
    coord_ind = 1;
    ind = 1;
    for i = 1 : 3*N*k1*k2*k3
        eigVector(i, col) = real(eigenV(ind)*(cos(dot(kVectorA, coordsA(coord_ind,:)))+sqrt(-1)*sin(dot(kVectorA, coordsA(coord_ind,:)))));
        ind = ind + 1;
        if i/3 == round(i/3)
            coord_ind = coord_ind + 1;
        end
        if ind > 3*N
            ind = 1;
        end
    end
    norm_factor = norm(eigVector(:,col));
    for i = 1 : 3*N*k1*k2*k3
        eigVector(i,col) = eigVector(i,col)/norm_factor;
    end
    col = col + 1;
end
'''