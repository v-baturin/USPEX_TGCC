import numpy as np

from ase.atom import Atom
from ase.neighborlist import primitive_neighbor_list
from dataclasses import dataclass
from typing import List, Union, Tuple

from USPEX.Common.Atomistic.Element import Element


def dfs1(graph : dict, root : int):
    '''
    Depth-first search
    :Link: https://en.wikipedia.org/wiki/Depth-first_search

    :param graph: 
    :param root: 
    :return: 
    '''


    visited, stack = set(), [root]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            stack.extend(graph[vertex] - visited)
    return visited


# @dataclass
# class B_Atom:
#     name : str
#     index : int
#
#     def __init__(self, name:str, index:int):
#         assert index >= 0
#         self.index = index
#         self.name = name


@dataclass
class Bond(object):
    '''
    Class for the chemical bond description. It has follow parameters:

    * _atom1 (int): ID of the fisrt atom in bond
    * _atom2 (int): ID of the second atom in bond
    * type (int): just number. If bond1.type == bond2.type then they are the same distance and type (like C-C and C-C are the same)
    * delta (float): distance between atoms - (R_val_1 + R_val_2), in Angstrems

    Some constant parameters that can be used from outside:

    * SAME_BOND_THRESHOLD = 0.05  # (Angstrems) same bond within this distance.
    * MAX_BOND = 5.0              # (Angstrems) maximum distance deviation for bonds search.
    * LOWER_BOUND = 0.5           # (Angstrems) lower bound for the distance of atoms in a bond.
    '''

    _atom1 : Atom
    _atom2 : Atom
    distance : float
    delta : float
    type : Union[int, str, None]
    enable : bool = False

    SAME_BOND_THRESHOLD : float = 0.05  # (Angstrems) same bond within this distance.
    MAX_BOND : float = 5.0              # (Angstrems) maximum distance deviation for bonds search.
    LOWER_BOUND : float = 0.5           # (Angstrems) lower bound for the distance of atoms in a bond.

    def __init__(self, atom1 : Atom, atom2 : Atom, distance : float, type=None,
                 direction : List[int] = [0,0,0], vector : List[float] = None):
        '''
        :param atom1: 
        :param atom2: 
        :param distance: 
        :param type: 
        :param direction: 
        '''
        assert 3 == len(direction)
        assert distance > 0.0

        R_val = lambda symbol: Element(symbol).covalent_radius

        self._atom1, self._atom2 = atom1, atom2
        self.type = type
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
        '''
        :return: indices of origin and end atoms in bond.
        '''
        return self._atom1, self._atom2

    def __eq__(self, other) -> bool:
        isEqual_Symbols = self.symbols == other.symbols or self.symbols == reversed(other.symbols)
        isEqualDistance = np.abs(self.distance - other.distance) < self.SAME_BOND_THRESHOLD
        return isEqual_Symbols and isEqualDistance

    def __ne__(self, other) -> bool:
        isEqual_Symbols = self.symbols == other.symbols or self.symbols == reversed(other.symbols)
        isEqualDistance = np.abs(self.distance - other.distance) < self.SAME_BOND_THRESHOLD
        return not (isEqual_Symbols and isEqualDistance)


# class Bonds(List[Bond]):
#     '''
#     Class for the list of the bonds.
#     '''
#
#     def append(self, object : Bond):
#         super(Bonds, self).append(object)
#
#     def getType(self, type):
#         '''
#         :param type (int or None): type number.
#         :return: all bonds of _type_.
#         '''
#         return Bonds([x for x in self if x.type == type])
#
#     def all_types(self):
#         '''
#         :return: all types of the bonds in this list.
#         '''
#         return set([x.type for x in self])
#
#     # TODO do it
#     def _is3Dconnected(self) -> bool:
#         X = np.max([np.abs(bond.direction[0]) for bond in self])
#         Y = np.max([np.abs(bond.direction[1]) for bond in self])
#         Z = np.max([np.abs(bond.direction[2]) for bond in self])
#         return X > 0 and Y > 0 and Z > 0
#
#     # TODO fix it
#     def connectList(self):
#         '''
#         IMPORTANT NOTE: this implementation checks connectivity of the atoms INSIDE unit cell.
#         It skips bonds that have and end in neighboring cells.
#         :return: the connectivity list in this set of bonds.
#         '''
#
#         # First, lets check whether we are not consider the case, when we have only 1 atom in unit cell.
#         uniqueAtoms = np.unique([bond.indicies for bond in self])
#         if 1 == len(uniqueAtoms):
#             return [uniqueAtoms[0]]
#
#         # bonds inside the unit cell
#         bonds = [bond.indicies for bond in self]
#         uniqueAtoms = np.unique(bonds)
#         pairs = {}
#
#         for atom in uniqueAtoms:
#             ToAdd = np.zeros((0, 1), dtype=int)
#             for eachBond in bonds:
#                 if eachBond[0] == atom:
#                     ToAdd = np.append(ToAdd, eachBond[1])
#                 elif eachBond[1] == atom:
#                     ToAdd = np.append(ToAdd, eachBond[0])
#             if len(ToAdd):
#                 ToAdd = np.array([x for x in ToAdd if x != atom], dtype=int)
#                 pairs[atom] = set(np.unique(ToAdd))
#
#         visited = dfs1(pairs, uniqueAtoms[0])
#         return list(visited)
#
#     def __add__(self, other):
#         return Bonds(super(Bonds,self).__add__(other))

def connectList(bondsGraph):
    '''
    IMPORTANT NOTE: this implementation checks connectivity of the atoms INSIDE unit cell.
    It skips bonds that have and end in neighboring cells.
    :return: the connectivity list in this set of bonds.
    '''
    bonds = [bond.indicies for bond in bondsGraph]

    # First, lets check whether we are not consider the case, when we have only 1 atom in unit cell.
    uniqueAtoms = np.unique(bonds)
    if 1 == len(uniqueAtoms):
        return [uniqueAtoms[0]]

    # bonds inside the unit cell
    uniqueAtoms = np.unique(bonds)
    pairs = {}

    for atom in uniqueAtoms:
        ToAdd = np.zeros((0, 1), dtype=int)
        for eachBond in bonds:
            if eachBond[0] == atom:
                ToAdd = np.append(ToAdd, eachBond[1])
            elif eachBond[1] == atom:
                ToAdd = np.append(ToAdd, eachBond[0])
        if len(ToAdd):
            ToAdd = np.array([x for x in ToAdd if x != atom], dtype=int)
            pairs[atom] = set(np.unique(ToAdd))

    visited = dfs1(pairs, uniqueAtoms[0])
    return list(visited)



def defaultGoodBonds(symbols):
    '''
    The function provides default good bonds values.

    :param atomType: a list with numeric representation of atom types.
    :return gBmatrix: upper-triangular N*N matrix with good bonds values.
    '''

    numSpecies = len(symbols)
    gB = np.zeros(numSpecies, dtype=float)
    gBmatrix = np.zeros((numSpecies, numSpecies), dtype=float)

    for i in range(numSpecies):
        gB[i] = Element(symbols[i]).good_bonds

    for i in range(numSpecies):
        for j in range(i, numSpecies):
            gBmatrix[i,j] = gBmatrix[j,i] = np.power(gB[i] * gB[j], 0.5)

    return gBmatrix
