"""
USPEX.Common.SpaceGroups.SymmetricStructure
===========================================

Collection of objects for working with topologies

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import numpy as np
from collections.abc import Sequence
from copy import deepcopy
from pymatgen.symmetry.groups import in_array_list
from itertools import chain


class SymmetricStructure(object):
    """
    Class representing topological net.
    """

    def __init__(self, name, group, sites):
        """
        Initialize topological net object.

        :type name: str
        :param name: Net name from TOPOS database.
        :type group: :class:`~USPEX.Common.SpaceGroups.SpaceGroups3D.Group`
        :param group: Space group corresponding to the net.
        :type sites: list
        :param sites: Symmetry inequivalent nodes.
        """
        self.name = name
        self.group = group
        self.sites = np.asarray(sites)
        self._multiplicities = None
        self._operations = None
        self._orbits = None

    def flavours(self, supercell: tuple = (1, 1, 1)):
        """
        Returns a list-like object enumerating all possible nets with same bond structures but different colourings
        of nodes and preserving the given supercell.

        :type supercell: tuple
        :param supercell: 3-tuple defining supercell.
        :rtype: :class:`SymmetricFlavours`
        :return: sequence of topological nets.
        """
        a, b, c = supercell
        shifts = np.array([np.array((i, j, k), dtype=float) for i in range(a) for j in range(b) for k in range(c)])
        positions = (np.stack(list(chain(*self.getOrbits()))).reshape((-1, 1, 3))
                     + shifts.reshape((1, -1, 3))).reshape((-1, 3)) / supercell
        return SymmetricFlavours(self.name, positions, self.group.getSupercellGroup(supercell).getAllSubgroups())

    def getOrbits(self):
        """
        :return: List of orbits (numpy arrays) for each site.
        """
        if self._orbits is None:
            self._orbits = self.group(self.sites)
        return deepcopy(self._orbits)

    @property
    def multiplicities(self):
        if self._multiplicities is None:
            self._multiplicities = np.asarray([len(orbit) for orbit in self.getOrbits()])
        return self._multiplicities

    @property
    def operations(self):
        if self._operations is None:
            self._operations = []
            for node, positions in zip(self.sites, self.getOrbits()):
                nodeOperationsVariants = []
                for group in self.group.getNotPositionInvariantSubgroups(node):
                    operations = []
                    for operation, position in zip(group.operators, positions):
                        operation = np.copy(operation)
                        operation[0:3, 3] = position
                        operations.append(operation)
                    nodeOperationsVariants.append(operations)
                self._operations.append(nodeOperationsVariants)
        return self._operations


class SymmetricFlavours(Sequence):
    """
    Class representing sequence of symmetric flavours of given structure.
    Creating a list of flavours is an expensive operation, so we simulate such list
    and generate requested flavour on the fly instead.
    A flavour is a symmetric structure with some chosen sites colouring.
    """

    def __init__(self, name, positions, subgroups):
        """
        :param name: Name of base structre.
        :param orbits: list of orbits (numpy arrays) for each site.
        :param subgroups: sequence of subgroups corresponding site colouring.
        """
        self._name = name
        self._subgroups = subgroups
        self._positions = positions

    def __getitem__(self, i):
        """
        Get topological flavour with index i.

        :type i: int
        :param i: Index
        :rtype: :class:`SymmetricStructure`
        :return: Symmetric structure object describing obtained flavour.
        """
        subgroup = self._subgroups[i]
        nodeCoordinates = []
        for subOrbit in subgroup(self._positions):
            for subNode in subOrbit:
                if in_array_list(nodeCoordinates, subNode):
                    break
            else:
                nodeCoordinates.append(subOrbit[0])
        return SymmetricStructure(name=self._name, group=subgroup, sites=nodeCoordinates)

    def __len__(self):
        """
        Special method which returns the number of flavours.
        """
        return len(self._subgroups)
