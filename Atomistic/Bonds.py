"""
USPEX.Common.Atomistic.Bonds
============================

Objects and methods for handling chemical bonds

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import numpy as np

from ase.atom import Atom
from ase.neighborlist import primitive_neighbor_list
from dataclasses import dataclass
from typing import List, Union, Tuple

from USPEX.Common.Atomistic.Element import Element


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
    distance : float
    delta : float
    enable : bool = False

    SAME_BOND_THRESHOLD = 0.05  # (Angstroms) same bond within this distance.
    MAX_BOND = 5.0              # (Angstroms) maximum distance deviation for bonds search.
    LOWER_BOUND = 0.5           # (Angstroms) lower bound for the distance of atoms in a bond.

    def __init__(self, atom1 : Atom, atom2 : Atom, distance : float,
                 direction : List[int] = [0,0,0], vector : List[float] = None):
        """
        :type atom1: Atom
        :param atom1:
            reference to the first atom in bond.
        :type atom2: Atom
        :param atom2:
            reference to the second atom in bond.
        :type distance: float
        :param distance:
            distance between atoms - (R_val_1 + R_val_2), in Angstroms.
        :type direction: list
        :param direction: bond direction with respect to cell parameters.
        """
        assert 3 == len(direction)
        assert distance > 0.0

        R_val = lambda symbol: Element(symbol).covalent_radius

        self._atom1, self._atom2 = atom1, atom2
        self.distance = distance
        self.delta = distance - R_val(atom1.symbol) - R_val(atom2.symbol)
        self.direction = direction
        self.vector = vector

    @property
    def indicies(self) -> Tuple[int, int]:
        return self._atom1.index, self._atom2.index

    @property
    def symbols(self) -> Tuple[str, str]:
        return self._atom1.symbol, self._atom2.symbol

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



def defaultGoodBonds(symbols: list):
    """
    The function provides default good bonds values.

    :type symbols: list
    :param symbols: a list with atomic symbols.
    :rtype: numpy array
    :return gBmatrix: upper-triangular N*N matrix with good bonds values.
    """

    numSpecies = len(symbols)
    gB = np.zeros(numSpecies, dtype=float)
    gBmatrix = np.zeros((numSpecies, numSpecies), dtype=float)

    for i in range(numSpecies):
        gB[i] = Element(symbols[i]).good_bonds

    for i in range(numSpecies):
        for j in range(i, numSpecies):
            gBmatrix[i, j] = gBmatrix[j, i] = np.power(gB[i] * gB[j], 0.5)

    return gBmatrix
