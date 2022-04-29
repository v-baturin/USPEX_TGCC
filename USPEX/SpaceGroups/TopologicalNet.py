"""
USPEX.Common.SpaceGroups.TopologicalNet
=======================================

Collection of objects for working with topologies

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import numpy as np
from collections.abc import Sequence

from .SpaceGroups3D import in_array_list


class TopologicalNet(object):
    """
    Class representing topological net.
    """

    def __init__(self, name, group, nodes, bonds):
        """
        Initialize topological net object.

        :type name: str
        :param name: Net name from TOPOS database.
        :type group: :class:`~USPEX.Common.SpaceGroups.SpaceGroups3D.Group`
        :param group: Space group corresponding to the net.
        :type nodes: list
        :param nodes: Symmetry inequivalent nodes.
        :type bonds: list or None
        :param bonds: Symmetry inequivalent bonds.
        """
        self.name = name
        self.group = group
        self.nodes = np.asarray(nodes)
        self._multiplicities = None
        self._operations = None

    def flavours(self, supercell: tuple = (1, 1, 1)):
        """
        Returns a list-like object enumerating all possible nets with same bond structures but different colourings
        of nodes and preserving the given supercell.

        :type supercell: tuple
        :param supercell: 3-tuple defining supercell.
        :rtype: :class:`TopologicalFlavours`
        :return: sequence of topological nets.
        """
        return TopologicalFlavours(self, supercell)

    @property
    def multiplicities(self):
        if self._multiplicities is None:
            self._multiplicities = np.asarray([len(orbit) for orbit in self.group(self.nodes)])
        return self._multiplicities

    @property
    def operations(self):
        if self._operations is None:
            self._operations = []
            for node, positions in zip(self.nodes, self.group(self.nodes)):
                nodeOperationsVariants = []
                for variant in self.group.getNotPositionInvariantSubgroups(node):
                    operations = []
                    for operation, position in zip(variant.operators, positions):
                        operation = np.copy(operation)
                        operation[0:3, 3] = position
                        operations.append(operation)
                    nodeOperationsVariants.append(operations)
                self._operations.append(nodeOperationsVariants)
        return self._operations


class TopologicalFlavours(Sequence):
    """
    Class representing sequence of topological net flavours.
    Creating a list of flavours is an expensive operation, so we simulate such list
    and generate requested flavour on the fly instead.
    A flavour is a topological net with some chosen nodes colouring.
    """

    def __init__(self, net,  supercell: tuple = (1, 1, 1)):
        """
        Initialize topological flavour.

        :type net: :class:`TopologicalNet`
        :param net: Parent topological net.
        :type supercell: tuple
        :param supercell: 3-tuple defining supercell.
        """
        self._net = net
        self._subgroups = net.group.getAllSubgroups(supercell)

    def __getitem__(self, i):
        """
        Get topological flavour with index i.

        :type i: int
        :param i: Index
        :rtype: :class:`TopologicalNet`
        :return: Topological net object describing obtained flavour.
        """
        subgroup = self._subgroups[i]
        nodeCoordinates = []
        for remOrbit in self._subgroups.calcOrbits(self._net.nodes):
            for subOrbit in subgroup(remOrbit):
                nodeIsUnique = True
                for subNode in subOrbit:
                    if in_array_list(nodeCoordinates, subNode):
                        nodeIsUnique = False
                        break
                if nodeIsUnique:
                    nodeCoordinates.append(subOrbit[0])
        # TODO Redefine bonds
        return TopologicalNet(name=self._net.name, group=subgroup, nodes=nodeCoordinates, bonds=None)

    def __len__(self):
        """
        Special method which returns the number of flavours.
        """
        return len(self._subgroups)
