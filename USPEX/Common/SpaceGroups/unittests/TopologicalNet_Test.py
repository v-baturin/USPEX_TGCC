import unittest
import numpy as np

from ..TopologicalNet import TopologicalNet
from ..SpaceGroups3D import Group

net_def_1 = {"337-topos_btu_btu": {"totalAtomNumber": 3, "groupName": "Pmmm", "nods": [[0.0, 0.0, 0.25], [0.0, 0.5, 0.0]], "bonds": [[[0.0, 0.0, 0.25], [-1.0, 0.0, 0.25]], [[0.0, 0.0, 0.25], [0.0, 0.0, 0.75]], [[0.0, 0.0, 0.25], [0.0, 0.0, -0.25]], [[0.0, 0.0, 0.25], [0.0, -0.5, 0.0]], [[0.0, 0.5, 0.0], [-1.0, 0.5, 0.0]]]}}
net_def_2 = {"735-topos_jse_jse": {"totalAtomNumber": 28, "groupName": "I4_1/a", "nods": [[1.6653345369377348e-16, 2.7755575615628914e-17, 0.5], [0.5000000000000001, 2.7755575615628914e-17, 0.3964], [0.10390000000000021, 0.26770000000000016, 0.4268]], "bonds": [[[1.6653345369377348e-16, 2.7755575615628914e-17, 0.5], [1.2677, -0.10389999999999994, 0.5732]], [[0.5000000000000001, 2.7755575615628914e-17, 0.3964], [0.3961000000000002, -0.26769999999999994, 0.3232]], [[0.5000000000000001, 2.7755575615628914e-17, 0.3964], [0.8961000000000001, -0.26769999999999994, 0.4268]], [[0.10390000000000021, -0.7322999999999998, 0.4268], [-0.10389999999999966, -0.26769999999999994, 0.4268]]]}}
net_def_3 = {"2177-topos_mmm_mmm": {"totalAtomNumber": 14, "cell": [[3.4641, 0.0, 0.0], [0.0, 3.4641, 0.0], [0.0, 0.0, 2.44949]], "groupName": "P4/nmm", "nods": [[0.5, 0.5, 0.0], [0.5, 0.3333300000000001, 0.33333000000000007], [0.25, 0.75, 0.5000000000000001]], "bonds": [[[0.5, 1.5, 0.0], [0.3333300000000001, 1.5, -0.33333000000000007]], [[0.5, 1.3333300000000001, 0.33333000000000007], [0.25, 1.25, 0.5000000000000001]]]}}
net_def_4 = {"1964-topos_urk_urk": {"totalAtomNumber": 7, "cell": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], "groupName": "Pm-3m", "nods": [[0.0, 0.0, 0.5], [0.0, 0.5, 0.5], [0.5, 0.5, 0.5]], "bonds": [[[0.0, 0.0, 0.5], [-0.5, 0.0, 0.5]], [[0.0, 0.5, 0.5], [-0.5, 0.5, 0.5]]]}}
net_def_5 = {"1904-topos_ith-d_ith-d": {"totalAtomNumber": 14, "cell": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "groupName": "Pm-3n", "nods": [[0.25, 0.25, 0.25], [0.25, 0.0, 0.5]], "bonds": [[[0.25, 0.25, 0.25], [0.0, 0.5, 0.25]], [[0.25, 0.0, 0.5], [-0.25, 0.0, 0.5]]]}}
net_def_6 = {"443-topos_ecn_ecn/polar": {"totalAtomNumber": 9, "cell": [[2.8844, 0.0, 0.0], [-1.4422, 2.4979636746758347, 0.0], [0.0, 0.0, 1.0]], "groupName": "R3m", "nods": [[0.7688, 0.8844, 0]], "bonds": [[[-0.2312, -0.1156, 0], [-0.55107, -0.44893, -0.33333]], [[-0.2312, -0.1156, 0], [-0.2312, -0.1156, -1.0]], [[-0.2312, -0.1156, 0], [0.1156, 0.2312, 0]]]}}
net_def_7 = {"1934-topos_xal_xal": {"totalAtomNumber": 2, "cell": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], "groupName": "P4/mmm", "nods": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]], "bonds": [[[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]], [[0.0, 0.0, 0.0], [-0.5, -0.5, -0.5]]]}}
net_def_8 = {"1940-topos_xaq_xaq": {"totalAtomNumber": 11, "cell": [[3.3094, 0.0, 0.0], [0.0, 3.3094, 0.0], [0.0, 0.0, 3.3094]], "groupName": "P-43m", "nods": [[0.3489, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1745, 0.1745, 0.1745]], "bonds": [[[0.3489, 0.0, 0.0], [0.1745, -0.1745, -0.1745]], [[0.3489, 0.0, 0.0], [0.6511, 0.0, 0.0]], [[0.0, 0.0, 0.0], [-0.1745, 0.1745, -0.1745]]]}}
net_def_9 = {"858-topos_nbo_nbo": {"totalAtomNumber": 6, "cell": [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]], "groupName": "Im-3m", "nods": [[0.5, 0.5, 0.0]], "bonds": [[[-0.5, -0.5, 0.0], [0.0, -0.5, 0.0]]]}}

# class RandTop_Test1(unittest.TestCase):
#     def setUp(self):
#         name, params = list(net_def_1.items())[0]
#         self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])
#
#     def test_flavours(self):
#         flavours = self.net.flavours((2, 3, 1))
#         # for i, flavour in enumerate(flavours):
#         i = 0
#         allGood = True
#         while i < len(flavours):
#             flavour = flavours[i]
#             self.assertTrue(np.sum(flavour.multiplicities) == 18,
#                             msg = 'Error with {}th flavour. Multiplicities are {}.'.format(i,flavour.multiplicities))
#             for operations, mult, node, n in zip(flavour.operations, flavour.multiplicities, flavour.nodes, range(len(flavour.nodes))):
#                 goodVariantExist = False
#                 for j, variant in enumerate(operations):
#                     if len(variant.operators) == mult:
#                         goodVariantExist = True
#                     # else:
#                     #     print(i, n, len(variant.operators), mult, j, len(operations))
#                 if not goodVariantExist:
#                     print(i, n, mult, len(operations))
#                     allGood = False
#                     # self.assertEqual(len(variant.operators), mult,
#                     #                  msg = 'In {}th flavour operations and multiplicities are inconsistent.'.format(i))
#             i += 1
#         self.assertTrue(allGood)
#         # flavour = flavours[10]
#         # operations, mult, node = flavour.operations[2], flavour.multiplicities[2], flavour.nodes[2]
#         # variant = operations[1]
#         # print(len(variant.operators), mult)
#         # print(variant.operators, node)



class RandTop_Test2(unittest.TestCase):
    def setUp(self):
        name, params = list(net_def_2.items())[0]
        self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])

    def test_isMultiplicities(self):
        self.assertTrue(np.all(self.net.multiplicities == [4,8,16]))


class RandTop_Test3(unittest.TestCase):
    def setUp(self):
        name, params = list(net_def_3.items())[0]
        self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])

    def test_isMultiplicities(self):
        self.assertTrue(np.all(self.net.multiplicities == [2,8,4]))

    def test_flavours(self):
        flavours = self.net.flavours((2, 1, 1))
        allGood = True
        for i, flavour in enumerate(flavours):
            self.assertTrue(np.sum(flavour.multiplicities) == 28,
                            msg = 'Error with {}th flavour. Multiplicities are {}.'.format(i,flavour.multiplicities))
            for operations, mult, node, n in zip(flavour.operations, flavour.multiplicities, flavour.nodes, range(len(flavour.nodes))):
                goodVariantExist = False
                for j, variant in enumerate(operations):
                    if len(variant.operators) == mult:
                        goodVariantExist = True
                    # else:
                    #     print(i, n, len(variant.operators), mult, j, len(operations))
                if not goodVariantExist:
                    print(i, n, mult, len(operations))
                    allGood = False
        self.assertTrue(allGood)


class RandTop_Test4(unittest.TestCase):
    def setUp(self):
        name, params = list(net_def_4.items())[0]
        self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])

    def test_isMultiplicities(self):
        self.assertTrue(np.all(self.net.multiplicities == [3,3,1]))

    def test_flavours(self):
        flavours = self.net.flavours((1, 4, 1))
        allGood = True
        for i, flavour in enumerate(flavours):
            self.assertTrue(np.sum(flavour.multiplicities) == 28,
                            msg = 'Error with {}th flavour. Multiplicities are {}.'.format(i,flavour.multiplicities))
            for operations, mult, node, n in zip(flavour.operations, flavour.multiplicities, flavour.nodes, range(len(flavour.nodes))):
                goodVariantExist = False
                for j, variant in enumerate(operations):
                    if len(variant.operators) == mult:
                        goodVariantExist = True
                    # else:
                    #     print(i, n, len(variant.operators), mult, j, len(operations))
                if not goodVariantExist:
                    print(i, n, mult, len(operations))
                    allGood = False
        self.assertTrue(allGood)


class RandTop_Test5(unittest.TestCase):
    def setUp(self):
        name, params = list(net_def_5.items())[0]
        self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])

    def test_isMultiplicities(self):
        self.assertTrue(np.all(self.net.multiplicities == [8,6]))

    def test_flavours(self):
        flavours = self.net.flavours((1, 1, 2))
        allGood = True
        for i, flavour in enumerate(flavours):
            self.assertTrue(np.sum(flavour.multiplicities) == 28,
                        msg = 'Error with {}th flavour. Multiplicities are {}.'.format(i,flavour.multiplicities))
            for operations, mult, node, n in zip(flavour.operations, flavour.multiplicities, flavour.nodes, range(len(flavour.nodes))):
                goodVariantExist = False
                for j, variant in enumerate(operations):
                    if len(variant.operators) == mult:
                        goodVariantExist = True
                    # else:
                    #     print(i, n, len(variant.operators), mult, j, len(operations))
                if not goodVariantExist:
                    print(i, n, mult, len(operations))
                    allGood = False
        self.assertTrue(allGood)

class RandTopTest6(unittest.TestCase):
    def setUp(self):
        name, params = list(net_def_6.items())[0]
        self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])

    def test_isMultiplicities(self):
        self.assertTrue(np.all(self.net.multiplicities == [9]))

    def test_flavours(self):
        flavours = self.net.flavours((1, 2, 1))
        allGood = True
        for i, flavour in enumerate(flavours):
            self.assertTrue(np.sum(flavour.multiplicities) == 18,
                        msg = 'Error with {}th flavour. Multiplicities are {}.'.format(i,flavour.multiplicities))
            for operations, mult, node, n in zip(flavour.operations, flavour.multiplicities, flavour.nodes, range(len(flavour.nodes))):
                goodVariantExist = False
                for j, variant in enumerate(operations):
                    if len(variant.operators) == mult:
                        goodVariantExist = True
                    # else:
                    #     print(i, n, len(variant.operators), mult, j, len(operations))
                if not goodVariantExist:
                    print(i, n, mult, len(operations))
                    allGood = False
        self.assertTrue(allGood)

class RandTopTest7(unittest.TestCase):
    def setUp(self):
        name, params = list(net_def_7.items())[0]
        self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])

    def test_isMultiplicities(self):
        self.assertTrue(np.all(self.net.multiplicities == [1,1]))

    def test_flavours(self):
        flavours = self.net.flavours((1, 1, 2))
        allGood = True
        for i, flavour in enumerate(flavours):
            self.assertTrue(np.sum(flavour.multiplicities) == 4,
                        msg = 'Error with {}th flavour. Multiplicities are {}.'.format(i,flavour.multiplicities))
            for operations, mult, node, n in zip(flavour.operations, flavour.multiplicities, flavour.nodes, range(len(flavour.nodes))):
                goodVariantExist = False
                for j, variant in enumerate(operations):
                    if len(variant.operators) == mult:
                        goodVariantExist = True
                    # else:
                    #     print(i, n, len(variant.operators), mult, j, len(operations))
                if not goodVariantExist:
                    #print(i, n, mult, len(operations))
                    allGood = False
        self.assertFalse(allGood)


class RandTopTest8(unittest.TestCase):
    def setUp(self):
        name, params = list(net_def_8.items())[0]
        self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])

    def test_flavours(self):
        flavours = self.net.flavours((1, 1, 1))
        for i, flavour in enumerate(flavours):
            self.assertTrue(np.sum(flavour.multiplicities) == 11,
                        msg = 'Error with {}th flavour. Multiplicities are {}.'.format(i,flavour.multiplicities))


# class RandTopTest9(unittest.TestCase):
#     def setUp(self):
#         name, params = list(net_def_9.items())[0]
#         self.net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])
#
#     def test_isMultiplicities(self):
#         self.assertTrue(np.all(self.net.multiplicities == [6]))
#
#     def test_flavours(self):
#         flavours = self.net.flavours((1, 1, 1))
#         allGood = True
#         for i, flavour in enumerate(flavours):
#             self.assertTrue(np.sum(flavour.multiplicities) == 6,
#                         msg = 'Error with {}th flavour. Multiplicities are {}.'.format(i,flavour.multiplicities))
#             for operations, mult, node, n in zip(flavour.operations, flavour.multiplicities, flavour.nodes, range(len(flavour.nodes))):
#                 goodVariantExist = False
#                 for j, variant in enumerate(operations):
#                     if len(variant.operators) == mult:
#                         goodVariantExist = True
#                     # else:
#                     #     print(i, n, len(variant.operators), mult, j, len(operations))
#                 if not goodVariantExist:
#                     #print(i, n, mult, len(operations))
#                     allGood = False
#         self.assertFalse(allGood)
