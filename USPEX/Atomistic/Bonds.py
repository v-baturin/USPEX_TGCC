"""
USPEX.Atomistic.Bonds
============================

Objects and methods for handling chemical bonds

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import numpy as np
from typing import Dict, List, Tuple
from ase.atoms import Atom, Atoms
from ase.neighborlist import primitive_neighbor_list
from itertools import chain, combinations_with_replacement
from scipy.sparse.csgraph import connected_components

from ..Atomistic.Element import Element

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
        return self._atom2.position - self._atom1.position + np.dot(self._dir2-self._dir1, self._cell)

    @property
    def distance(self) -> float:
        return np.linalg.norm(self.vector)

    @property
    def delta(self):
        R_val = lambda symbol: Element(symbol).covalent_radius
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


class Bonds:

    def __init__(self, sameBond: float = None, maxBond: float = None, lowerBond: float = None, goodBonds: dict = None):
        self.sameBond = sameBond if sameBond is not None else SAME_BOND_THRESHOLD
        self.maxBond = maxBond if maxBond is not None else MAX_BOND
        self.lowerBond = lowerBond if lowerBond is not None else LOWER_BOND
        if goodBonds is not None:
            self.goodBonds = {}
            for key, value in goodBonds.items():
                self.goodBonds[frozenset(key)] = value
        else:
            self.goodBonds = None

    def isConnected(self, SYSTEM):
        strongBonds, weakBonds = self.getAllBondsInCutoff(SYSTEM, cutoffType='RcovTimes', cutoffParameter=2)
        pbc = SYSTEM.getCell().getPBC()
        # TODO: Make cutoffparameter an input parameter
        # try:
        #     self.getMinimalGraphBonds(structure)
        #     res = True
        # except:
        #     res = False
        return self._howmanyConnectedComponents(len(SYSTEM), strongBonds + weakBonds, pbc) == 1

    def getAllBondsInCutoff(self, SYSTEM, cutoffType='Rmax', cutoffParameter=None):
        """
        Gets all bonds in SYSTEM, whose lengths do not exceed
        cutoffs of one of the following types:
            1. 'Rmax' (default), cutoff Parameter is a simple bond threshold (default: MAX_BOND)
            2. 'RcovTimes' covalent radii times given factor (default: 1)
            3. 'RcovPlus' covalent radii plus increments (default: 0)
            4. 'RcovPlusGoodBonds' covalent radii plus goodBonds-based increments (default)
        @param SYSTEM: AtomicStructure instance
        @param cutoffFactor: float or int
        @param cutoffRadius: float or int
        @return:
        """
        defaultParameters = {'Rmax': self.maxBond, 'RcovTimes': 1, 'RcovPlus': 0}
        if cutoffParameter is None and cutoffType != 'RcovPlusGoodBonds':
            cutoffParameter = defaultParameters[cutoffType]

        goodBonds = {frozenset((s1.short_name, s2.short_name)): np.power(s1.good_bonds * s2.good_bonds, 0.5)
                     for s1, s2 in combinations_with_replacement(SYSTEM.getAtomTypes(), 2)} \
            if self.goodBonds is None else self.goodBonds

        strongBondThresholds = {key: - 0.37 * np.log(goodBonds[key])
                                for key in goodBonds.keys()}

        if cutoffType == 'Rmax':
            cutoff = cutoffParameter
        else:
            covalentLengths = {frozenset((s1.short_name, s2.short_name)): Element(s1.short_name).covalent_radius +
                                                                          Element(s1.short_name).covalent_radius
                               for s1, s2 in combinations_with_replacement(SYSTEM.getAtomTypes(), 2)}
            if cutoffType == 'RcovTimes':
                cutoff = {key: val * cutoffParameter for key, val in covalentLengths.items()}
            elif cutoffType == 'RcovPlus':
                cutoff = {key: val + cutoffParameter for key, val in covalentLengths.items()}
            elif cutoffType == 'RcovPlusGoodBonds':
                cutoff = {key: val + strongBondThresholds[key] for key, val in covalentLengths.items()}
            else:
                raise ValueError('Unsupported cutoffType')
            cutoff = {tuple(key) * (3 - len(key)): val for key, val in cutoff.items()}

        structure = Atoms(symbols=[s.short_name for s in SYSTEM.getAtomTypes()],
                          positions=SYSTEM.getCartesianCoordinates(),
                          cell=SYSTEM.getCell().getCellVectors(),
                          pbc=SYSTEM.getCell().getPBC())

        # 1) Calculate bonds within upper bound to max_bond.
        # 2) Group bonds by using same_bond criterion.
        bonds = []
        i_init, j_init, dists, vecs, dirs = primitive_neighbor_list(quantities='ijdDS', pbc=structure.pbc,
                                                                    cell=structure.get_cell(complete=True),
                                                                    positions=structure.get_scaled_positions(),
                                                                    cutoff=cutoff, numbers=structure.numbers,
                                                                    use_scaled_positions=True)

        for i, j, dist, vec, dir in zip(i_init, j_init, dists, vecs, dirs):
            # TODO Why we had this less 0.5A and not more than 5A (usually)
            # if np.abs(dist - tmp_Rval) > cutoff or dist < 0.5:
            if dist < self.lowerBond or j < i:
                continue
            bonds.append(Bond(atom1=structure[i], atom2=structure[j], dir2=dir))

        tmp_bonds = sorted(bonds, key=lambda x: x.delta)

        strongBonds = []
        weakBonds = []

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
            if min([bond.delta for bond in bonds_one_type]) < strongBondThresholds[frozenset((a, b))]:
                strongBonds.append(bonds_one_type)  # Add by group
            else:
                weakBonds.append(bonds_one_type)

        return [strongBonds, weakBonds]

    def getMinimalGraphBonds(self, SYSTEM) -> list:
        '''
        Calculates bond graph minimal for the structure to be 3D connected.

        :param SYSTEM:
        :return:
        '''

        N_atom = len(SYSTEM)
        pbc = SYSTEM.getCell().getPBC()
        # goodBonds = {frozenset((s1.short_name, s2.short_name)): np.power(s1.good_bonds * s2.good_bonds, 0.5)
        #              for s1, s2 in combinations_with_replacement(SYSTEM.getAtomTypes(), 2)} \
        #     if self.goodBonds is None else self.goodBonds
        # covalentLengths = {(s1.short_name, s2.short_name): Element(s1.short_name).covalent_radius +
        #                                                    Element(s1.short_name).covalent_radius
        #                    for s1, s2 in combinations_with_replacement(SYSTEM.getAtomTypes(), 2)}
        # structure = Atoms(symbols=[s.short_name for s in SYSTEM.getAtomTypes()],
        #                   positions=SYSTEM.getCartesianCoordinates(),
        #                   cell=SYSTEM.getCell().getCellVectors(),
        #                   pbc=SYSTEM.getCell().getPBC())
        #
        # # 1) Calculate bonds within upper bound to max_bond.
        # # 2) Group bonds by using same_bond criterion.
        # bonds = []
        # i_init, j_init, dists, vecs, dirs = primitive_neighbor_list(quantities='ijdDS', pbc=structure.pbc,
        #                                                             cell=structure.get_cell(complete=True),
        #                                                             positions=structure.get_scaled_positions(),
        #                                                             cutoff=self.maxBond, numbers=structure.numbers,
        #                                                             use_scaled_positions=True)
        #
        # for i, j, dist, vec, dir in zip(i_init, j_init, dists, vecs, dirs):
        #     # TODO Why we had this less 0.5A and not more than 5A (usually)
        #     # if np.abs(dist - tmp_Rval) > cutoff or dist < 0.5:
        #     if dist < self.lowerBond or j < i:
        #         continue
        #     bonds.append(Bond(atom1=structure[i], atom2=structure[j], dir2=dir))

        bondIn, weakBonds = self.getAllBondsInCutoff(SYSTEM, cutoffType='Rmax', cutoffParameter=self.maxBond)


        bond_total = []
        # while tmp_bonds:
        #     bond = tmp_bonds.pop(0)
        #     bonds_one_type = [bond]
        #     bonds_remain = []
        #     # Obtain all bonds with the same type by distance:
        #     for b in tmp_bonds:
        #         if b.isClose(bond, self.sameBond):
        #             bonds_one_type.append(b)
        #         else:
        #             bonds_remain.append(b)
        #     tmp_bonds = bonds_remain
        #     bond_total.append(bonds_one_type)
        #
        # # 3) Add bonds by group.
        # bond_in = []
        # bond_left = []
        #
        # # delete short bonds
        # for bond_group in bond_total:
        #     a, b = bond_group[0].symbols
        #     small_bond = -0.37 * np.log(goodBonds[frozenset((a, b))])
        #     if min([bond.delta for bond in bond_group]) < small_bond:
        #         bond_in.append(bond_group)  # Add by group
        #     else:
        #         bond_left.append(bond_group)
        # del bond_group[0]

        # 5, check 3D connectivity, if not satisfied, add more bonds,
        #   but we only include those bonds which could increase connectivity
        # ---Looks like we have to include all bonds before the connectivity changes
        #   otherwise, we won't add them

        N_components = self._howmanyConnectedComponents(N_atom, bondIn, pbc=pbc)
        # List = connectList(chain(*bond_in))

        while N_components > 1:
            # disp('The stuture is not fully connected, adding more bonds');
            bond_tmp = bondIn + [weakBonds.pop(0)]
            # List_new = connectList(chain(*bond_tmp))
            N_components_new = self._howmanyConnectedComponents(N_atom, bond_tmp, pbc=pbc)
            # if len(List_new) > len(List) or len(List) == 1: # increase connectivity accept
            if N_components_new < N_components:
                # disp('The connectivity is increased, accept adding more bonds');
                # List = List_new
                N_components = N_components_new
                bondIn = bond_tmp
                # else
                # disp('The connectivity is not increased, reject adding more bonds');

        # 6, Remove double count of bond like [i,i] pair;
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


def defaultGoodBonds(symbols: List[str]) -> Dict[Tuple[str, str], float]:
    """
    The function provides default good bonds values.

    :type symbols: list
    :param symbols: a list with atomic symbols.
    :rtype: numpy array
    :return gBmatrix: upper-triangular N*N matrix with good bonds values.
    """

    assert len(set(symbols)) == len(symbols)
    goodBond = lambda symbol: Element(symbol).good_bonds

    gB_dict = {}
    for s1, s2 in combinations_with_replacement(symbols, r=2):
        gB_dict[(s1, s2)] = np.power(goodBond(s1) * goodBond(s2), 0.5)
    return gB_dict

    # numSpecies = len(symbols)
    # gB = np.zeros(numSpecies, dtype=float)
    # gBmatrix = np.zeros((numSpecies, numSpecies), dtype=float)
    #
    # for i in range(numSpecies):
    #     gB[i] = Element(symbols[i]).good_bonds
    #
    # for i in range(numSpecies):
    #     for j in range(i, numSpecies):
    #         gBmatrix[i, j] = gBmatrix[j, i] = np.power(gB[i] * gB[j], 0.5)
    #
