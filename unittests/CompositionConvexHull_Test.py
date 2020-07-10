"""
USPEX.Common.Atomistic.unittests.ConvexHull_Test
================================================

Class for ConvexHull testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import unittest
import numpy as np
import os


from ..Fitness import Fintness


# class System(object):
#
#     def __init__(self, composition, enthalpy):
#         self.composition = composition
#         self.enthalpy = enthalpy


class CompositionConvexHull_Test(unittest.TestCase):

    def test_unocomponent(self):
        filename = 'tmp_u.dump'
        if os.path.exists(filename):
            os.remove(filename)

        heights = Fintness.convexHullHeight(np.empty((1,0), dtype=float), np.array([-2.0]))
        self.assertTrue(np.allclose(heights, np.array([0])))

        heights = Fintness.convexHullHeight(np.empty((2,0), dtype=float), np.array([-2.0, -4.0]))
        self.assertTrue(np.allclose(heights, np.array([2.0,0])))

        heights = Fintness.convexHullHeight(np.empty((3,0), dtype=float), np.array([-2.0,-4.0,-1.0]))
        self.assertTrue(np.allclose(heights, np.array([2.0,0,3.0])))

        # compositionSpace = CompositionSpace(symbols=['Mo'], blocks=[[1]], range=[[1, 18]])
        # self.convexHull = CompositionConvexHull(compositionSpace, saved_data=filename)
        # system1 = System({'Mo': 4}, -8.0)
        # system2 = System({'Mo': 4}, -16.0)
        # system3 = System({'Mo': 8}, -8.0)
        #
        # # ans = self.convexHull.height[0]
        # # self.assertTrue(np.isinf(ans) and ans > 0)
        # self.convexHull.extend([system1])
        # self.assertEqual(len(self.convexHull.lower_bound), 1)
        #
        # self.convexHull.extend([system2])
        # self.assertEqual(len(self.convexHull.lower_bound), 1)
        # self.assertEqual(set(self.convexHull.lower_bound), {system2})
        # self.assertAlmostEqual(self.convexHull.height[0], 2.0)
        # self.assertEqual(set(self.convexHull.upper_bound), {system1})
        # self.assertAlmostEqual(self.convexHull.depth[1], -2.0)
        #
        # self.convexHull.extend([system3])
        # self.assertEqual(len(self.convexHull.lower_bound), 1)
        # self.assertEqual(set(self.convexHull.lower_bound), {system2})
        # self.assertAlmostEqual(self.convexHull.height[0], 2.0)
        # self.assertAlmostEqual(self.convexHull.height[2], 3.0)
        # self.assertEqual(set(self.convexHull.upper_bound), {system3})
        # self.assertAlmostEqual(self.convexHull.depth[0], -1.0)
        # self.assertAlmostEqual(self.convexHull.depth[1], -3.0)

    def test_bicomponent(self):
        filename = 'tmp.dump'
        if os.path.exists(filename):
            os.remove(filename)

        heights = Fintness.convexHullHeight(np.array([[2.0/7.0]]),
                                            np.array([-5.0/14.0]))
        self.assertTrue(np.allclose(heights, np.array([0])))

        heights = Fintness.convexHullHeight(np.array([[2.0/7.0], [2.0/7.0]]),
                                            np.array([-5.0/14.0, -1.0]))
        self.assertTrue(np.allclose(heights, np.array([9.0/14.0, 0.0])))

        heights = Fintness.convexHullHeight(np.array([[2.0/7.0], [2.0/7.0], [0.5]]),
                                            np.array([-5.0/14.0, -1.0, -0.5]))
        self.assertTrue(np.allclose(heights, np.array([0.642857, 0.0, 0.0])))

        heights = Fintness.convexHullHeight(np.array([[2.0/7.0], [2.0/7.0], [0.5], [1.0]]),
                                            np.array([-5.0/14.0, -1.0, -0.5, -0.25]))
        self.assertTrue(np.allclose(heights, np.array([0.642857, 0.0, 0.275, 0.0])))

        heights = Fintness.convexHullHeight(np.array([[2.0/7.0], [2.0/7.0], [0.5], [1.0], [1.0]]),
                                            np.array([-5.0/14.0, -1.0, -0.5, -0.25, -0.5]))
        self.assertTrue(np.allclose(heights, np.array([0.642857, 0.0, 0.35, 0.25, 0.0])))

        heights = Fintness.convexHullHeight(np.array([[2.0/7.0], [2.0/7.0], [0.5], [1.0], [1.0], [0.0]]),
                                            np.array([-5.0/14.0, -1.0, -0.5, -0.25, -0.5, -1.2]))
        self.assertTrue(np.allclose(heights, np.array([0.642857, 0.0, 0.35, 0.25, 0.0, 0.0])))

        heights = Fintness.convexHullHeight(np.array([[2.0/7.0], [2.0/7.0], [0.5], [1.0], [1.0], [0.0], [0.6]]),
                                            np.array([-5.0/14.0, -1.0, -0.5, -0.25, -0.5, -1.2, -0.2]))
        self.assertTrue(np.allclose(heights, np.array([0.642857, 0.0, 0.35, 0.25, 0.0, 0.0, 0.58])))

        heights = Fintness.convexHullHeight(np.array([[2.0/7.0], [2.0/7.0], [0.5], [1.0], [1.0], [0.0], [0.6], [0.4]]),
                                            np.array([-5.0/14.0, -1.0, -0.5, -0.25, -0.5, -1.2, -0.2, -1.6]))
        self.assertTrue(np.allclose(heights, np.array([1.128571, 0.485714, 0.916667, 0.25, 0.0, 0.0, 1.033333, 0.0])))

        # compositionSpace = CompositionSpace(symbols=['Mo', 'B'], blocks=[[1, 0], [0, 1]], range=[[0, 18], [0, 18]],
        #                           minAt=8, maxAt=18)
        # self.convexHull = CompositionConvexHull(compositionSpace, saved_data=filename)
        # system1 = System({'Mo': 4, 'B': 10}, -5.0)
        # system2 = System({'Mo': 4, 'B': 10}, -14.0)
        # system3 = System({'Mo': 8, 'B': 8}, -8.0)
        # system4 = System({'Mo': 8}, -2.0)
        # system5 = System({'Mo': 8}, -4.0)
        # system6 = System({'B': 10}, -12.0)
        # system7 = System({'Mo': 6, 'B': 4}, -2.0)
        # system8 = System({'Mo': 4, 'B': 6}, -16.0)
        #
        #
        # # ans = self.convexHull.height[0]
        # # self.assertTrue(np.isinf(ans) and ans > 0)
        # self.convexHull.extend([system1])
        # self.assertEqual(len(self.convexHull.lower_bound), 1)
        #
        # # ans = self.convexHull.height[1]
        # # self.assertTrue(np.isinf(ans) and ans > 0)
        # self.convexHull.extend([system2])
        # self.assertEqual(len(self.convexHull.lower_bound), 1)
        # # self.assertEqual(self.convexHull.elements[0].enthalpy, -14.0)
        #
        # ans = self.convexHull.height[0]
        # self.assertAlmostEqual(ans, 9/14)
        # # self.convexHull.extend([system1])
        # # self.assertEqual(len(self.convexHull.lower_bound), 2)
        # # self.assertEqual(self.convexHull.elements[0].enthalpy, -14.0)
        #
        # # ans = self.convexHull.height[2]
        # # self.assertTrue(np.isinf(ans) and ans > 0)
        # self.convexHull.extend([system3])
        # self.assertEqual(len(self.convexHull.lower_bound), 2)
        # self.assertEqual(set(self.convexHull.lower_bound), {system2, system3})
        # self.assertAlmostEqual(self.convexHull.height[0], 0.642857)
        #
        # # ans = self.convexHull.height[3]
        # # self.assertTrue(np.isinf(ans) and ans > 0)
        # self.convexHull.extend([system4])
        # self.assertEqual(len(self.convexHull.lower_bound), 2)
        # self.assertEqual(set(self.convexHull.lower_bound), {system2,system4})
        # self.assertAlmostEqual(self.convexHull.height[0], 0.642857)
        # self.assertAlmostEqual(self.convexHull.height[2], 0.275)
        #
        # # ans = self.convexHull.height[4]
        # # self.assertTrue(np.isinf(ans) and ans > 0)
        # self.convexHull.extend([system5])
        # self.assertEqual(len(self.convexHull.lower_bound), 2)
        # self.assertEqual(set(self.convexHull.lower_bound), {system2, system5})
        # self.assertAlmostEqual(self.convexHull.height[0], 0.642857)
        # self.assertAlmostEqual(self.convexHull.height[2], 0.35)
        # self.assertAlmostEqual(self.convexHull.height[3], 0.25)
        #
        # # self.convexHull.extend([system4])
        # # self.assertEqual(len(self.convexHull.lower_bound), 2)
        #
        # # ans = self.convexHull.height[5]
        # # self.assertTrue(np.isinf(ans) and ans > 0)
        # self.convexHull.extend([system6])
        # self.assertEqual(len(self.convexHull.lower_bound), 3)
        # self.assertEqual(set(self.convexHull.lower_bound), {system2, system5, system6})
        # self.assertAlmostEqual(self.convexHull.height[0], 0.642857)
        # self.assertAlmostEqual(self.convexHull.height[2], 0.35)
        # self.assertAlmostEqual(self.convexHull.height[3], 0.25)
        #
        # # self.convexHull.extend([system2])
        # # self.assertEqual(len(self.convexHull.lower_bound), 3)
        #
        # self.convexHull.extend([system7])
        # self.assertEqual(len(self.convexHull.lower_bound), 3)
        # self.assertEqual(set(self.convexHull.lower_bound), {system2, system5, system6})
        # self.assertAlmostEqual(self.convexHull.height[0], 0.642857)
        # self.assertAlmostEqual(self.convexHull.height[2], 0.35)
        # self.assertAlmostEqual(self.convexHull.height[3], 0.25)
        # self.assertAlmostEqual(self.convexHull.height[6], 0.58)
        #
        # self.convexHull.extend([system8])
        # self.assertEqual(len(self.convexHull.lower_bound), 3)
        # self.assertEqual(set(self.convexHull.lower_bound), {system5, system6, system8})
        # self.assertAlmostEqual(self.convexHull.height[0], 1.128571)
        # self.assertAlmostEqual(self.convexHull.height[1], 0.485714)
        # self.assertAlmostEqual(self.convexHull.height[2], 0.916667)
        # self.assertAlmostEqual(self.convexHull.height[3], 0.25)
        # self.assertAlmostEqual(self.convexHull.height[6], 1.033333)

    # def test_bicomponent1(self):
    #     filename = 'tmp1.dump'
    #     if os.path.exists(filename):
    #         os.remove(filename)
    #
    #     compositionSpace = CompositionSpace(symbols=['Mo', 'B'], blocks=[[1, 0], [0, 1]], range=[[0, 18], [0, 18]],
    #                               minAt=8, maxAt=18)
    #     self.convexHull = CompositionConvexHull(compositionSpace, saved_data=filename)
    #     system1 = System({'Mo': 4, 'B': 10}, -5.0)
    #     system2 = System({'Mo': 4, 'B': 10}, -14.0)
    #     system3 = System({'Mo': 8, 'B': 8}, -8.0)
    #     system4 = System({'Mo': 8}, -2.0)
    #     system5 = System({'Mo': 8}, -4.0)
    #     system6 = System({'B': 10}, -12.0)
    #     system7 = System({'Mo': 6, 'B': 4}, -2.0)
    #     system8 = System({'Mo': 4, 'B': 6}, -16.0)
    #
    #     systems = [system1, system2, system3, system4, system5, system6, system7, system8]
    #     self.convexHull.extend(systems)
    #
    #     self.assertEqual(set(self.convexHull.lower_bound), {system5, system6, system8})
    #     self.assertEqual(set(self.convexHull.upper_bound), {system1, system4, system6, system7})
    #     self.convexHull.clean()

        # ans = self.convexHull[system1]
        # self.assertTrue(np.isinf(ans) and ans < 0)
        # self.convexHull.add(system1)
        # self.assertEqual(len(self.convexHull.elements), 1)
        # self.assertEqual(self.convexHull.elements[0].enthalpy, -5.0)

        # ans = self.convexHull[system2]
        # self.assertAlmostEqual(ans, -9/14)
        # self.convexHull.add(system2)
        # self.assertEqual(len(self.convexHull.elements), 1)
        # self.assertEqual(self.convexHull.elements[0].enthalpy, -14.0)
        #
        # ans = self.convexHull[system1]
        # self.assertAlmostEqual(ans, 9/14)
        # self.convexHull.add(system1)
        # self.assertEqual(len(self.convexHull.elements), 1)
        # self.assertEqual(self.convexHull.elements[0].enthalpy, -14.0)
        #
        # ans = self.convexHull[system3]
        # self.assertTrue(np.isinf(ans) and ans < 0)
        # self.convexHull.add(system3)
        # self.assertEqual(len(self.convexHull.elements), 2)
        # enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        # self.assertEqual(enthalpies, {-14.0, -8.0})
        #
        # ans = self.convexHull[system4]
        # self.assertTrue(np.isinf(ans) and ans < 0)
        # self.convexHull.add(system4)
        # self.assertEqual(len(self.convexHull.elements), 2)
        # enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        # self.assertEqual(enthalpies, {-14.0, -2.0})
        #
        # ans = self.convexHull[system5]
        # self.assertAlmostEqual(ans, -2.0/8)
        # self.convexHull.add(system5)
        # self.assertEqual(len(self.convexHull.elements), 2)
        # enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        # self.assertEqual(enthalpies, {-14.0, -4.0})
        #
        # ans = self.convexHull[system4]
        # self.assertAlmostEqual(ans, 2.0/8)
        # self.convexHull.add(system4)
        # self.assertEqual(len(self.convexHull.elements), 2)
        # enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        # self.assertEqual(enthalpies, {-14.0, -4.0})
        #
        # ans = self.convexHull[system6]
        # self.assertTrue(np.isinf(ans) and ans < 0)
        # self.convexHull.add(system6)
        # self.assertEqual(len(self.convexHull.elements), 3)
        # enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        # self.assertEqual(enthalpies, {-14.0, -4.0, -12.0})
        #
        # ans = self.convexHull[system2]
        # self.assertAlmostEqual(ans, 0.0)
        # self.convexHull.add(system2)
        # self.assertEqual(len(self.convexHull.elements), 3)
        # enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        # self.assertEqual(enthalpies, {-14.0, -4.0, -12.0})
        #
        # ans = self.convexHull[system7]
        # self.assertGreater(ans, 0.0)
        # self.convexHull.add(system7)
        # self.assertEqual(len(self.convexHull.elements), 3)
        # enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        # self.assertEqual(enthalpies, {-14.0, -4.0, -12.0})
        #
        # ans = self.convexHull[system8]
        # self.assertLess(ans, 0.0)
        # self.convexHull.add(system8)
        # self.assertEqual(len(self.convexHull.elements), 3)
        # enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        # self.assertEqual(enthalpies, {-16.0, -4.0, -12.0})

    def test_bicomponent2(self):
        filename = 'tmp2.dump'
        if os.path.exists(filename):
            os.remove(filename)

        heights = Fintness.convexHullHeight(np.array([[0.3], [8.0/13.0], [0.25], [0.3], [13.0/18.0], [0.9]]),
                                            np.array([-8.94225, -8.65781, -8.13805, -8.8125, -8.37417, -7.7661]))
        self.assertTrue(np.allclose(np.flatnonzero(heights), np.array([3])))

        # compositionSpace = CompositionSpace(symbols=['Mo', 'B'], blocks=[[1, 0], [0, 1]], range=[[0, 18], [0, 18]],
        #                          minAt=8, maxAt=18)
        # self.convexHull = CompositionConvexHull(compositionSpace, saved_data=filename)
        # system1 = System({'Mo': 6, 'B': 14}, -178.845)
        # system2 = System({'Mo': 16, 'B': 10}, -225.103)
        # system3 = System({'Mo': 5, 'B': 14}, -162.761)
        # system4 = System({'Mo': 6, 'B': 14}, -176.250)
        # system5 = System({'Mo': 13, 'B': 5}, -150.735)
        # system6 = System({'Mo': 9, 'B': 1}, -77.661)
        # systems = [system1, system2, system3, system4, system5, system6]
        #
        # self.convexHull.extend(systems)
        #
        # self.assertEqual(set(self.convexHull.lower_bound), {system1,system2,system3,system5,system6})
        # self.assertEqual(set(self.convexHull.upper_bound), {system3,system6})
        # self.convexHull.clean()
        # ans = self.convexHull[system1]
        # self.assertTrue(np.isinf(ans) and ans < 0)
        # self.convexHull.add(system1)
        # self.assertEqual(len(self.convexHull.elements), 1)
        #
        # ans = self.convexHull[system2]
        # self.assertTrue(np.isinf(ans) and ans < 0)
        # self.convexHull.add(system2)
        # self.assertEqual(len(self.convexHull.elements), 2)
        #
        # ans = self.convexHull[system3]
        # self.assertTrue(np.isinf(ans) and ans < 0)
        # self.convexHull.add(system3)
        # self.assertEqual(len(self.convexHull.elements), 3)
        #
        # ans = self.convexHull[system4]
        # self.assertTrue(ans > 0)
        # self.convexHull.add(system4)
        # self.assertEqual(len(self.convexHull.elements), 3)
        #
        # ans = self.convexHull[system5]
        # self.assertTrue(ans < 0)
        # self.convexHull.add(system5)
        # self.assertEqual(len(self.convexHull.elements), 4)
        #
        # ans = self.convexHull[system6]
        # self.assertTrue(np.isinf(ans) and ans < 0)
        # self.convexHull.add(system6)
        # self.assertEqual(len(self.convexHull.elements), 5)

    def test_bicomponent3(self):
        filename = 'tmp3.dump'
        if os.path.exists(filename):
            os.remove(filename)

        heights = Fintness.convexHullHeight(np.array([[0.25], [1.0], [0.25], [0.3], [1.0]]),
                                            np.array([-8.66625, -8.5888, -8.6701, -8.87155, -7.6904]))
        self.assertTrue(np.allclose(np.flatnonzero(heights), np.array([0, 4])))

        # compositionSpace = CompositionSpace(symbols=['Mo', 'B'], blocks=[[1, 0], [0, 1]], range=[[0, 18], [0, 18]],
        #                          minAt=8, maxAt=18)
        # self.convexHull = CompositionConvexHull(compositionSpace, saved_data=filename)
        # system1 = System({'Mo': 5, 'B': 15}, -173.325)
        # system2 = System({'Mo': 5}, -42.944)
        # system3 = System({'Mo': 3, 'B': 9}, -104.041)
        # system4 = System({'Mo': 6, 'B': 14}, -177.431)
        # system5 = System({'Mo': 9}, -69.214)
        # systems = [system1, system2, system3, system4, system5]
        #
        # self.convexHull.extend(systems)
        #
        # self.assertEqual(set(self.convexHull.lower_bound), {system2,system3,system4})
        # self.assertEqual(set(self.convexHull.upper_bound), {system1,system5})
        # self.convexHull.clean()

        # ans = self.convexHull[system1]
        # self.assertTrue(np.isinf(ans) and ans < 0)
        # self.convexHull.add(system1)
        # self.assertEqual(len(self.convexHull.elements), 1)
        #
        # ans = self.convexHull[system2]
        # self.assertTrue(np.isinf(ans) and ans < 0)
        # self.convexHull.add(system2)
        # self.assertEqual(len(self.convexHull.elements), 2)
        #
        # ans = self.convexHull[system3]
        # self.assertTrue(ans < 0)
        # self.convexHull.add(system3)
        # self.assertEqual(len(self.convexHull.elements), 2)
        #
        # ans = self.convexHull[system4]
        # self.assertTrue(ans < 0)
        # self.convexHull.add(system4)
        # self.assertEqual(len(self.convexHull.elements), 3)
        #
        # ans = self.convexHull[system5]
        # self.assertTrue(ans > 0)
        # self.convexHull.add(system5)
        # self.assertEqual(len(self.convexHull.elements), 3)
