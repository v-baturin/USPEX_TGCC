"""
USPEX.Atomistic.Bonds
============================

Objects and methods for handling chemical bonds

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import numpy as np
from ase.atom import Atom
# from ase.neighborlist import primitive_neighbor_list
from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Dict, List, Tuple

from ..Atomistic.Element import Element


@dataclass
class Bond(object):
    """
    Class for the chemical bond description. It has the follow parameters:

    * _atom1 (Atom): reference to the first atom in bond
    * _atom2 (Atom): reference to the second atom in bond
    * type (int): just a number. If bond1.type == bond2.type then the two bonds have
                  the same distance and type (like C-C and C-C are the same)
    * delta (float): distance between atoms - (R_val_1 + R_val_2), in Angstroms

    Some constant parameters that can be used from outside:

    * SAME_BOND_THRESHOLD = 0.05  # (Angstroms) same bond within this distance.
    * MAX_BOND = 5.0              # (Angstroms) maximum distance deviation for bonds search.
    * LOWER_BOUND = 0.5           # (Angstroms) lower bound for the distance of atoms in a bond.
    """

    _atom1 : Atom
    _atom2 : Atom
    enable : bool = False

    SAME_BOND_THRESHOLD = 0.05  # (Angstroms) same bond within this distance.
    MAX_BOND = 5.0              # (Angstroms) maximum distance deviation for bonds search.
    LOWER_BOUND = 0.5           # (Angstroms) lower bound for the distance of atoms in a bond.

    def __init__(self, atom1 : Atom, atom2 : Atom, dir1 : List[int] = [0,0,0], dir2 : List[int] = [0,0,0]):
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
        self._dir1 = np.array(dir1, dtype=int)
        self._dir2 = np.array(dir2, dtype=int)

    @property
    def indicies(self) -> Tuple[int, int]:
        return self._atom1.index, self._atom2.index

    @property
    def direction(self) -> List[int]:
        return self._dir2 - self._dir1

    @property
    def symbols(self) -> Tuple[str, str]:
        return self._atom1.symbol, self._atom2.symbol

    @property
    def vector(self) -> List[float]:
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

    def __eq__(self, other) -> bool:
        """
        Special method for supporting '==' operator.

        :type other: :class:`Bond`
        :param other: another bond for comparison.
        :rtype: bool
        :return: True if the two bond lengths are closer than SAME_BOND_THRESHOLD, False otherwise.
        """

        isEqual_Symbols = self.symbols == other.symbols or self.symbols == reversed(other.symbols)
        isEqualDistance = np.abs(self.distance - other.distance) < self.SAME_BOND_THRESHOLD
        return isEqual_Symbols and isEqualDistance

    def __ne__(self, other) -> bool:
        """
        Special method for supporting '!=' operator.

        :type other: :class:`Bond`
        :param other: another bond for comparison.
        :rtype: bool
        :return: False if the two bond lengths are closer than SAME_BOND_THRESHOLD, True otherwise.
        """
        isEqual_Symbols = self.symbols == other.symbols or self.symbols == reversed(other.symbols)
        isEqualDistance = np.abs(self.distance - other.distance) < self.SAME_BOND_THRESHOLD
        return not (isEqual_Symbols and isEqualDistance)



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
    for s1,s2 in combinations_with_replacement(symbols, r=2):
        gB_dict[(s1,s2)] = np.power(goodBond(s1) * goodBond(s2), 0.5)
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
