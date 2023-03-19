"""
USPEX.Atomistic.Bonds
============================

Objects and methods for handling chemical bonds

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
import numpy as np
from typing import Dict, List, Tuple, Union
from ase.atoms import Atom, Atoms
from ase.neighborlist import primitive_neighbor_list
from itertools import chain, combinations_with_replacement
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist
from scipy.stats import gmean
from itertools import chain

from .VolumeEstimator import VolumeEstimator


logger = logging.getLogger(__name__)

closest = np.array(
    [[[0, 0, 0]], [[-1, 0, 0]], [[-1, 0, -1]], [[-1, -1, -1]], [[-1, -1, 0]], [[0, -1, 0]], [[0, -1, -1]],
     [[0, 0, -1]], [[-1, -1, 1]], [[-1, 0, 1]], [[-1, 1, 1]], [[-1, 1, 0]], [[-1, 1, -1]], [[0, -1, 1]],
     [[0, 0, 1]], [[0, 1, 1]], [[0, 1, 0]], [[0, 1, -1]], [[1, 0, 0]], [[1, 0, -1]], [[1, -1, -1]],
     [[1, -1, 0]], [[1, -1, 1]], [[1, 0, 1]], [[1, 1, 1]], [[1, 1, 0]], [[1, 1, -1]]])

SAME_BOND_THRESHOLD = 0.05  # (Angstroms) same bond within this distance.
MAX_BOND = 5.0  # (Angstroms) maximum distance deviation for bonds search.
LOWER_BOND = 0.5  # (Angstroms) lower bound for the distance of atoms in a bond.


class Bond(object):
    """
    Class for the chemical bond description. It has the follow parameters:

    * delta (float): distance between atoms - (R_val_1 + R_val_2), in Angstroms
    """

    def __init__(self, atom1: Atom, atom2: Atom, dir1: Tuple[int] = (0, 0, 0), dir2: Tuple[int] = (0, 0, 0)):
        """
        :type atom1: Atom
        :param atom1: reference to the first atom in bond.
        :type atom2: Atom
        :param atom2: reference to the second atom in bond.
        :type direction: list
        :param direction: bond direction with respect to cell parameters.
        """
        assert atom1.atoms == atom2.atoms
        assert 3 == len(dir1) == len(dir2)
        self._cell = atom1.atoms.get_cell()
        self._atom1, self._atom2 = atom1, atom2
        self._dir1 = np.asarray(dir1, dtype=int)
        self._dir2 = np.asarray(dir2, dtype=int)

    @property
    def indicies(self) -> Tuple[int, int]:
        return self._atom1.index, self._atom2.index

    @property
    def direction(self):
        return self._dir2 - self._dir1

    @property
    def symbols(self) -> Tuple[str, str]:
        return self._atom1.symbol, self._atom2.symbol

    @property
    def vector(self):
        return self._atom2.position - self._atom1.position + np.dot(self._dir2 - self._dir1, self._cell)

    @property
    def distance(self) -> float:
        return np.linalg.norm(self.vector)

    @property
    def delta(self):
        R_val = lambda symbol: BondUtility.atomType(symbol).covalent_radius
        return self.distance - R_val(self._atom1.symbol) - R_val(self._atom2.symbol)

    @property
    def atoms(self):
        """
        :rtype: tuple
        :return: indices of origin and end atoms in bond.
        """
        return self._atom1, self._atom2

    def isClose(self, other, threshold):
        isEqualSymbols = self.symbols == other.symbols or self.symbols == reversed(other.symbols)
        isEqualDistance = np.abs(self.distance - other.distance) < threshold
        return isEqualSymbols and isEqualDistance


class BondUtility:

    atomType = None
    disassemblerType = None

    @classmethod
    def registerTypes(cls, atomType, disassemblerType):
        """
        Register types used by this utility.

        :param structureType: type representing atomic structure.
        :param atomType: type representing chemical element.
        :param cellType: type representing unit cell.
        :param atomicDisassemblerType: type representing utility used for disassembling structure into molecules.
        """
        cls.atomType = atomType
        cls.disassemblerType = disassemblerType

    def __init__(self, sameBond: float = None, maxBond: float = None, lowerBond: float = None, goodBonds: dict = None,
                 cutoff: Union[str, Dict, float, int] = 'strong', volumeType=0, ionDistances=None):
        self.sameBond = sameBond if sameBond is not None else SAME_BOND_THRESHOLD
        self.maxBond = maxBond if maxBond is not None else MAX_BOND
        self.lowerBond = lowerBond if lowerBond is not None else LOWER_BOND
        if goodBonds is not None:
            self.goodBonds = {}
            for key, value in goodBonds.items():
                self.goodBonds[frozenset(key)] = value
        else:
            self.goodBonds = None
        self.cutoff = cutoff
        self.volumeEstimator = VolumeEstimator(volumeType)

        self._distances = {}
        ionDistances = ionDistances if ionDistances is not None else {}
        for key, value in ionDistances.items():
            assert isinstance(key, str)
            assert np.isfinite(value)
            s1, s2 = key.split(' ')
            self._distances[(s1, s2)] = value

    def isConnected(self, structure, cutoff=None):
        """
        checks if structure is connected, with bonds graph based on thresholds based on atom valence radii
        Rcutoff(type_i, type_j) = checkConnectivityCutoffFactor * (Rval(type_i) + Rval(type_j))

        @param structure: AtomicStructure instance
        @param cutoff: str, dict, float, int
        @return: bool
        """
        cutoff = self.buildCutoffDict(structure, cutoff)
        strongBonds, weakBonds = self.getAllBondsInCutoff(structure, cutoff)
        pbc = structure.getCell().getPBC()
        # TODO: Make cutoffparameter an input parameter
        return self._howmanyConnectedComponents(len(structure), strongBonds + weakBonds, pbc) == 1

    def _strongBondsMaxDelta(self, structure):
        """
        A bond ij is considered strong if
        |r_ij| - (Rcov(Type_i) + Rcov(Type_j)) <= goodBondsDelta(Type_i,Type_j))
        @param structure: AtomicStructure instance
        @return:
        """
        goodBonds = {frozenset((s1.short_name, s2.short_name)): np.power(s1.good_bonds * s2.good_bonds, 0.5)
                     for s1, s2 in combinations_with_replacement(structure.getAtomTypes(), 2)} \
            if self.goodBonds is None else self.goodBonds

        return {key: - 0.37 * np.log(val) for key, val in goodBonds.items()}

    def buildCutoffDict(self, structure, cutoff=None):
        """
        Builds dictionary of cutoffs compatible with ase.neighborlist.primitive_neighbor_list. Each dict item
        corresponds to a pair of elements with values of threshold bond distances
        Cutoffs are build as:
            1. 'strong' cutoff based on classic USPEX checkConnectivity
            2. 'vdw' cutoff as sum of van der Waals radii
        @param structure: AtomicStructure instance
        @param cutoffParameter: float or int, factor or increment depending on cutoffType
        @return: dict of cutoffs consistent with  cutoff dict parameter
        """
        if cutoff == None:
            cutoff = self.cutoff

        if isinstance(cutoff, (dict, float, int)):
            return cutoff

        covalentLengths = {frozenset((s1.short_name, s2.short_name)): self.atomType(s1.short_name).covalent_radius +
                                                                      self.atomType(s2.short_name).covalent_radius
                           for s1, s2 in combinations_with_replacement(structure.getAtomTypes(), 2)}

        if cutoff == 'strong':
            strongMaxDelta = self._strongBondsMaxDelta(structure)
            cutoff = {key: val + strongMaxDelta[key] for key, val in covalentLengths.items()}
        elif cutoff == 'vdw':
            cutoff = {frozenset((s1.short_name, s2.short_name)): self.atomType(s1.short_name).vanderWaals_radius +
                                                                 self.atomType(s2.short_name).vanderWaals_radius
                      for s1, s2 in combinations_with_replacement(structure.getAtomTypes(), 2)}
        else:
            raise ValueError('Unsupported cutoffType')
        return {tuple(key) * (3 - len(key)): val for key, val in cutoff.items()}

    def getAllBondsInCutoff(self, structure, cutoff):
        """
        Gets all bonds in structure, whose lengths do not exceed cut-off
        Cut-off parameter is passed to ase.neighborlist.primitive_neighbor_list, hence its format
        @param structure: AtomicStructure instance
        @param cutoff: float or dict or list or array
                Cutoff for neighbor search. It can be:

                    * A single float: This is a global cutoff for all elements.
                    * A dictionary: This specifies cutoff values for element
                      pairs. Specification accepts element numbers of symbols.
                      Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
                    * A list/array with a per atom value: This specifies the radius of
                      an atomic sphere for each atom. If spheres overlap, atoms are
                      within each other's neighborhood. See :func:`~ase.neighborlist.natural_cutoffs`
                      for an example on how to get such a list.
        @return: (strongBonds: List, weakBonds: List) (separated according to goodBonds-based criteria)
        """

        atoms = Atoms(symbols=[s.short_name for s in structure.getAtomTypes()],
                          positions=structure.getCartesianCoordinates(),
                          cell=structure.getCell().getCellVectors(),
                          pbc=structure.getCell().getPBC())

        # 1. Calculate bonds within cutoff.
        bonds = []
        i_init, j_init, dists, vecs, dirs = primitive_neighbor_list(quantities='ijdDS', pbc=atoms.pbc,
                                                                    cell=atoms.get_cell(complete=True),
                                                                    positions=atoms.get_scaled_positions(),
                                                                    cutoff=cutoff, numbers=atoms.numbers,
                                                                    use_scaled_positions=True)

        for i, j, dist, vec, dir in zip(i_init, j_init, dists, vecs, dirs):
            # TODO Why we had this less 0.5A and not more than 5A (usually)
            if dist < self.lowerBond or j < i:
                continue
            bonds.append(Bond(atom1=atoms[i], atom2=atoms[j], dir2=dir))

        tmp_bonds = sorted(bonds, key=lambda x: x.delta)

        # 2. Group similar bonds and distribute them between strong and weak bonds
        #    according to goodBonds-based criterion
        strongBonds = []
        weakBonds = []
        strongBondMaxDelta = self._strongBondsMaxDelta(structure)
        while tmp_bonds:
            bond = tmp_bonds.pop(0)
            bonds_one_type = [bond]
            bonds_remain = []
            # Obtain all bonds with the same type by distance:
            for b in tmp_bonds:
                if b.isClose(bond, self.sameBond):
                    bonds_one_type.append(b)
                else:
                    bonds_remain.append(b)
            tmp_bonds = bonds_remain
            a, b = bonds_one_type[0].symbols
            if min([bond.delta for bond in bonds_one_type]) < strongBondMaxDelta[frozenset((a, b))]:
                strongBonds.append(bonds_one_type)  # Add by group
            else:
                weakBonds.append(bonds_one_type)

        return strongBonds, weakBonds

    def getMinimalGraphBonds(self, structure) -> list:
        '''
        Calculates bond graph minimal for the structure to be 3D connected.

        :param structure: AtomicStructure instance
        :return: bond graph
        '''

        N_atom = len(structure)
        pbc = structure.getCell().getPBC()

        # 1. getting bonds that are shorter than cutoff (self.maxBond)
        cutoff = self.maxBond
        bondIn, weakBonds = self.getAllBondsInCutoff(structure, cutoff)

        # 2. check 3D connectivity, if not satisfied, add more bonds of increasing lengths,
        #    until connectivity is acheived
        N_components = self._howmanyConnectedComponents(N_atom, bondIn, pbc=pbc)

        while N_components > 1:
            bond_tmp = bondIn + [weakBonds.pop(0)]  # The stuture is not fully connected, adding more bonds
            N_components_new = self._howmanyConnectedComponents(N_atom, bond_tmp, pbc=pbc)
            if N_components_new < N_components:
                N_components = N_components_new  # Connectivity increased, accept adding more bonds
                bondIn = bond_tmp

        # 3. Remove double count of bond like [i,i] pair;
        for i, bonds_tmp in enumerate(bondIn):
            indicies = []
            for j, bond in enumerate(bonds_tmp):
                a, b = bond.indicies
                if a == b:
                    indicies.append(j)
            for j in sorted(indicies[::2], reverse=True):
                del bondIn[i][j]

        return bondIn

    def _howmanyConnectedComponents(self, N, bonds, pbc):
        """
        Calculate number of connected components.
        :param N: number of atoms
        :param bonds: bond graph
        :param pbc: pbc
        :return: Number of connected components
        """
        pbc = np.array(pbc, dtype=bool)

        supercell_size = 2

        supercell_dims = pbc * (supercell_size - 1) + 1
        supercell_ranges = np.array([[0, 1]] * 3) * supercell_dims.reshape(3, -1)
        all_cells_in_super = np.array(np.meshgrid(*[range(*x) for x in supercell_ranges])).T.reshape(-1, 3)
        total_cells = len(all_cells_in_super)

        graph = np.zeros((total_cells * N, total_cells * N))

        pwrs = np.zeros(3, dtype=int)
        pwrs[pbc] = np.array([2, 1, 0])[np.sort(pbc)]
        for bond in chain(*bonds):
            i, j = bond.indicies
            for klm in all_cells_in_super:
                i_super = i + np.sum(supercell_dims ** pwrs * klm * N)
                j_super = j + np.sum(supercell_dims ** pwrs * ((klm + bond.direction) % supercell_dims) * N)
                graph[i_super, j_super] = 1

        N_components, labels = connected_components(graph)
        return N_components

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

    def hardness(self, system):
        if 'bondUtility.hardness' not in system:
            structure, disassembler = self.disassemblerType.assembe(**system)
            bonds = self.getMinimalGraphBonds(structure)
            system['bondUtility.hardness'] = self.calcHardness(structure, bonds)
        return system['bondUtility.hardness']

    @staticmethod
    def calcCoordinationNumbers(structure):
        radii = np.tile([element.covalent_radius for element in structure.getAtomTypes()], len(closest))
        base = radii.reshape(1, radii.size) + radii.reshape(radii.size, 1)
        vertices = np.dot(np.concatenate((closest + structure.getFractionalCoordinates()), axis=0),
                          structure.getCell().getCellVectors())
        order = np.exp(-(cdist(vertices, vertices) - base) / 0.23)
        order = np.delete(np.triu(order, 1), 0, 1) + np.delete(np.tril(order, -1), len(vertices) - 1, 1)
        return order.sum(axis=1) / order.max(axis=1)

    def getDistances(self, symbols, pressure):
        """
        For given array of symbols generates matrix of minimal distances.
        If minimal distance for pair of symbols is not predefined calculates it using volumeUtility.

        :param symbols: N array of symbols
        :param pressure: external pressure.

        :return: N*N array of minimal distances.
        """
        symbols = [symbol.short_name for symbol in symbols]
        uniqueSimbols = np.unique(symbols)
        minDistMatrix = {}
        radii = {symbol: self.volumeEstimator.calcAtomVolume(symbol, pressure) ** (1.0 / 3.0)
                 for symbol in uniqueSimbols}
        for s1, s2 in combinations_with_replacement(uniqueSimbols, 2):
            if (s1, s2) in self._distances:
                minDistMatrix[(s2, s1)] = minDistMatrix[(s1, s2)] = self._distances[(s1, s2)]
            elif (s2, s1) in self._distances:
                minDistMatrix[(s1, s2)] = minDistMatrix[(s2, s1)] = self._distances[(s2, s1)]
            elif self.volumeEstimator.volumeType == 0:
                minDistMatrix[(s1, s2)] = minDistMatrix[(s2, s1)] = min(0.22 * (radii[s1] + radii[s2]), 1.2)
            else:
                minDistMatrix[(s1, s2)] = minDistMatrix[(s2, s1)] = 0.45 * (radii[s1] + radii[s2])

        N = len(symbols)
        mDM = np.zeros((N, N), dtype=float)
        for i, j in combinations_with_replacement(range(N), 2):
            s1 = symbols[i]
            s2 = symbols[j]
            mDM[i, j] = mDM[j, i] = minDistMatrix[(s1, s2)]
        return mDM


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
