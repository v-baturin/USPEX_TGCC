import unittest
import numpy as np

from ..TopologicalNet import TopologicalNet
from ..SpaceGroups3D import Group

net_def_1 = {"337-topos_btu_btu": {"totalAtomNumber": 3, "groupName": "Pmmm", "nods": [[0.0, 0.0, 0.25], [0.0, 0.5, 0.0]], "bonds": [[[0.0, 0.0, 0.25], [-1.0, 0.0, 0.25]], [[0.0, 0.0, 0.25], [0.0, 0.0, 0.75]], [[0.0, 0.0, 0.25], [0.0, 0.0, -0.25]], [[0.0, 0.0, 0.25], [0.0, -0.5, 0.0]], [[0.0, 0.5, 0.0], [-1.0, 0.5, 0.0]]]}}
net_def_2 = {"735-topos_jse_jse": {"totalAtomNumber": 28, "groupName": "I4_1/a", "nods": [[1.6653345369377348e-16, 2.7755575615628914e-17, 0.5], [0.5000000000000001, 2.7755575615628914e-17, 0.3964], [0.10390000000000021, 0.26770000000000016, 0.4268]], "bonds": [[[1.6653345369377348e-16, 2.7755575615628914e-17, 0.5], [1.2677, -0.10389999999999994, 0.5732]], [[0.5000000000000001, 2.7755575615628914e-17, 0.3964], [0.3961000000000002, -0.26769999999999994, 0.3232]], [[0.5000000000000001, 2.7755575615628914e-17, 0.3964], [0.8961000000000001, -0.26769999999999994, 0.4268]], [[0.10390000000000021, -0.7322999999999998, 0.4268], [-0.10389999999999966, -0.26769999999999994, 0.4268]]]}}
net_def_3 = {"2177-topos_mmm_mmm": {"totalAtomNumber": 14, "cell": [[3.4641, 0.0, 0.0], [0.0, 3.4641, 0.0], [0.0, 0.0, 2.44949]], "groupName": "P4/nmm", "nods": [[0.5, 0.5, 0.0], [0.5, 0.3333300000000001, 0.33333000000000007], [0.25, 0.75, 0.5000000000000001]], "bonds": [[[0.5, 1.5, 0.0], [0.3333300000000001, 1.5, -0.33333000000000007]], [[0.5, 1.3333300000000001, 0.33333000000000007], [0.25, 1.25, 0.5000000000000001]]]}}
net_def_4 = {"1964-topos_urk_urk": {"totalAtomNumber": 7, "cell": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], "groupName": "Pm-3m", "nods": [[0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5]], "bonds": [[[0.0, 0.0, 0.5], [-0.5, 0.0, 0.5]], [[0.0, 0.5, 0.5], [-0.5, 0.5, 0.5]]]}}
net_def_5 = {"1904-topos_ith-d_ith-d": {"totalAtomNumber": 14, "cell": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "groupName": "Pm-3n", "nods": [[0.25, 0.25, 0.25], [0.25, 0.0, 0.5]], "bonds": [[[0.25, 0.25, 0.25], [0.0, 0.5, 0.25]], [[0.25, 0.0, 0.5], [-0.25, 0.0, 0.5]]]}}


class RandTopTest1(unittest.TestCase):
    def setUp(self):
        name, params = list(net_def_1.items())[0]
        self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])

    def test_flavours(self):
        flavours = self.net.getFlavours((2,3,1))
        for i, flavour in enumerate(flavours):
            self.assertTrue(np.sum(flavour.multiplicities) == 18,
                            msg = 'Error with {}th flavour. Multiplicities are {}.'.format(i,flavour.multiplicities))


class RandTopTest2(unittest.TestCase):
    def setUp(self):
        name, params = list(net_def_2.items())[0]
        self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])

    def test_isMultiplicities(self):
        self.assertTrue(np.all(self.net.multiplicities == [4,8,16]))


class RandTopTest3(unittest.TestCase):
    def setUp(self):
        name, params = list(net_def_3.items())[0]
        self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])

    def test_isMultiplicities(self):
        self.assertTrue(np.all(self.net.multiplicities == [2,8,4]))

    def test_flavours(self):
        flavours = self.net.getFlavours((2,1,1))
        for i, flavour in enumerate(flavours):
            self.assertTrue(np.sum(flavour.multiplicities) == 28,
                            msg = 'Error with {}th flavour. Multiplicities are {}.'.format(i,flavour.multiplicities))


class RandTopTest4(unittest.TestCase):
    def setUp(self):
        name, params = list(net_def_4.items())[0]
        self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])

    def test_isMultiplicities(self):
        self.assertTrue(np.all(self.net.multiplicities == [3,3,1]))

    def test_flavours(self):
        flavours = self.net.getFlavours((1,4,1))
        for i, flavour in enumerate(flavours):
            self.assertTrue(np.sum(flavour.multiplicities) == 28,
                            msg = 'Error with {}th flavour. Multiplicities are {}.'.format(i,flavour.multiplicities))


class RandTopTest5(unittest.TestCase):
    def setUp(self):
        name, params = list(net_def_5.items())[0]
        self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])

    def test_isMultiplicities(self):
        self.assertTrue(np.all(self.net.multiplicities == [8,6]))

    def test_flavours(self):
        flavours = self.net.getFlavours((1,1,2))
        for i, flavour in enumerate(flavours):
            self.assertTrue(np.sum(flavour.multiplicities) == 28,
                        msg = 'Error with {}th flavour. Multiplicities are {}.'.format(i,flavour.multiplicities))
