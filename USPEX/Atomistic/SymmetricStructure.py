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
        return SymmetricFlavours(self.name, self.sites, self.group.getAllSubgroups(supercell))

    def getOrbits(self):
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
    Class representing sequence of topological net flavours.
    Creating a list of flavours is an expensive operation, so we simulate such list
    and generate requested flavour on the fly instead.
    A flavour is a topological net with some chosen nodes colouring.
    """

    def __init__(self, name, sites, subgroups):
        """
        """
        self._name = name
        self._sites = sites
        self._subgroups = subgroups

    def __getitem__(self, i):
        """
        Get topological flavour with index i.

        :type i: int
        :param i: Index
        :rtype: :class:`SymmetricStructure`
        :return: Topological net object describing obtained flavour.
        """
        subgroup = self._subgroups[i]
        nodeCoordinates = []
        for remOrbit in self._subgroups.calcOrbits(self._sites):
            for subOrbit in subgroup(remOrbit):
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
