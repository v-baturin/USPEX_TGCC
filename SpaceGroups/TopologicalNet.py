import numpy as np
from collections import Sequence

from .SpaceGroups3D import in_array_list


class TopologicalNet(object):
    """
    Class representing topological net.
    """

    def __init__(self, name, group, nodes, bonds):
        """
        Initialize topological net object.
        :param name: Net name from TOPOS database.
        :param group: Space group corresponding the net.
        :param nodes: Symmetry inequivalent nodes.
        :param bonds: Symmetry inequivalent bonds.
        """
        self.name = name
        self.group = group
        self.nodes = np.asarray(nodes)
        self.operations = NodeOperations(self.nodes, self.group)
        # for node in self.nodes:
        #     nodeOperationsVariants = self.group.getNotPositionInvariantSubgroups(node)
        #     self.operations.append(nodeOperationsVariants)
        self.multiplicities = np.asarray([len(orbit) for orbit in self.group(self.nodes)])
        self.bonds = bonds
        self.coordinationNumbers = []
        self.coordinationNumbers = np.asarray(self.coordinationNumbers)


    def flavours(self, supercell : tuple = (1, 1, 1)):
        """
        Reterns list-like object enumerating all possible nets with same bond structures but different colourings of nodes
        and preserving given supercell.
        :param supercell: 3-tuple defining supercell.
        :return:
        """
        return TopologicalFlavours(self, supercell)


class NodeOperations(Sequence):
    """

    """

    def __init__(self, nodes, group):
        """

        :param nodes:
        :param group:
        """
        self.nodes = nodes
        self.group = group
        self._operations = [None] * len(self.nodes)

    def __getitem__(self, item):
        """

        :param item:
        :return:
        """
        if self._operations[item] is None:
            self._operations[item] = self.group.getNotPositionInvariantSubgroups(self.nodes[item])
        return self._operations[item]

    def __len__(self):
        """

        :return:
        """
        return len(self.nodes)


class TopologicalFlavours(Sequence):
    """
    Class representing sequence of topological net flavours.
    Creating list of flavours is an expensive operation so we simulate such list
    and generate requested flavour on the fly instead.
    Flavour is a topological net with some chosen nodes colouring.
    """

    def __init__(self, net,  supercell : tuple = (1,1,1)):
        """
        Initialize topological flavour.
        :param net: Parent topological net.
        :param supercell: 3-tuple defining supercell.
        """
        self._net = net
        self._subgroups = net.group.getAllSubgroups(supercell)

    def __getitem__(self, i):
        """
        Get topological flavour with index i.
        :param i: Index
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
        #TODO Redefine bonds
        return TopologicalNet(name = self._net.name, group = subgroup, nodes = np.asarray(nodeCoordinates), bonds = None)

    def __len__(self):
        """
        Returns number of flavours.
        :return: Number of flavours.
        """
        return len(self._subgroups)
