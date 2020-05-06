"""
USPEX.Common.Atomistic.unittests.ConvexHull_Test
================================================

Class for ConvexHull testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import unittest
import numpy as np

from ..ConvexHull import ConvexHull
from ..CompositionSpace import CompositionSpace

class System(object):

    def __init__(self, composition, enthalpy):
        self.composition = composition
        self.enthalpy = enthalpy


class ConvexHull_Test(unittest.TestCase):

    def test_bicomponent1(self):
        compositionSpace = CompositionSpace(symbols=['Mo', 'B'], blocks=[[1, 0], [0, 1]], range=[[0, 18], [0, 18]],
                                  minAt=8, maxAt=18)
        self.convexHull = ConvexHull(compositionSpace)
        system1 = System({'Mo': 4, 'B': 10}, -5.0)
        system2 = System({'Mo': 4, 'B': 10}, -14.0)
        system3 = System({'Mo': 8, 'B': 8}, -8.0)
        system4 = System({'Mo': 8}, -2.0)
        system5 = System({'Mo': 8}, -4.0)
        system6 = System({'B': 10}, -12.0)
        system7 = System({'Mo': 6, 'B': 4}, -2.0)
        system8 = System({'Mo': 4, 'B': 6}, -16.0)

        ans = self.convexHull[system1]
        self.assertTrue(np.isinf(ans) and ans < 0)
        self.convexHull.add(system1)
        self.assertEqual(len(self.convexHull.elements), 1)
        self.assertEqual(self.convexHull.elements[0].enthalpy, -5.0)

        ans = self.convexHull[system2]
        self.assertAlmostEqual(ans, -9/14)
        self.convexHull.add(system2)
        self.assertEqual(len(self.convexHull.elements), 1)
        self.assertEqual(self.convexHull.elements[0].enthalpy, -14.0)

        ans = self.convexHull[system1]
        self.assertAlmostEqual(ans, 9/14)
        self.convexHull.add(system1)
        self.assertEqual(len(self.convexHull.elements), 1)
        self.assertEqual(self.convexHull.elements[0].enthalpy, -14.0)

        ans = self.convexHull[system3]
        self.assertTrue(np.isinf(ans) and ans < 0)
        self.convexHull.add(system3)
        self.assertEqual(len(self.convexHull.elements), 2)
        enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        self.assertEqual(enthalpies, {-14.0, -8.0})

        ans = self.convexHull[system4]
        self.assertTrue(np.isinf(ans) and ans < 0)
        self.convexHull.add(system4)
        self.assertEqual(len(self.convexHull.elements), 2)
        enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        self.assertEqual(enthalpies, {-14.0, -2.0})

        ans = self.convexHull[system5]
        self.assertAlmostEqual(ans, -2.0/8)
        self.convexHull.add(system5)
        self.assertEqual(len(self.convexHull.elements), 2)
        enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        self.assertEqual(enthalpies, {-14.0, -4.0})

        ans = self.convexHull[system4]
        self.assertAlmostEqual(ans, 2.0/8)
        self.convexHull.add(system4)
        self.assertEqual(len(self.convexHull.elements), 2)
        enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        self.assertEqual(enthalpies, {-14.0, -4.0})

        ans = self.convexHull[system6]
        self.assertTrue(np.isinf(ans) and ans < 0)
        self.convexHull.add(system6)
        self.assertEqual(len(self.convexHull.elements), 3)
        enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        self.assertEqual(enthalpies, {-14.0, -4.0, -12.0})

        ans = self.convexHull[system2]
        self.assertAlmostEqual(ans, 0.0)
        self.convexHull.add(system2)
        self.assertEqual(len(self.convexHull.elements), 3)
        enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        self.assertEqual(enthalpies, {-14.0, -4.0, -12.0})

        ans = self.convexHull[system7]
        self.assertGreater(ans, 0.0)
        self.convexHull.add(system7)
        self.assertEqual(len(self.convexHull.elements), 3)
        enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        self.assertEqual(enthalpies, {-14.0, -4.0, -12.0})

        ans = self.convexHull[system8]
        self.assertLess(ans, 0.0)
        self.convexHull.add(system8)
        self.assertEqual(len(self.convexHull.elements), 3)
        enthalpies = set(sys.enthalpy for sys in self.convexHull.elements)
        self.assertEqual(enthalpies, {-16.0, -4.0, -12.0})

    def test_bicomponent2(self):
        compositionSpace = CompositionSpace(symbols=['Mo', 'B'], blocks=[[1, 0], [0, 1]], range=[[0, 18], [0, 18]],
                                 minAt=8, maxAt=18)
        self.convexHull = ConvexHull(compositionSpace)
        system1 = System({'Mo': 6, 'B': 14}, -178.845)
        system2 = System({'Mo': 16, 'B': 10}, -225.103)
        system3 = System({'Mo': 5, 'B': 14}, -162.761)
        system4 = System({'Mo': 6, 'B': 14}, -176.250)
        system5 = System({'Mo': 13, 'B': 5}, -150.735)
        system6 = System({'Mo': 9, 'B': 1}, -77.661)

        ans = self.convexHull[system1]
        self.assertTrue(np.isinf(ans) and ans < 0)
        self.convexHull.add(system1)
        self.assertEqual(len(self.convexHull.elements), 1)

        ans = self.convexHull[system2]
        self.assertTrue(np.isinf(ans) and ans < 0)
        self.convexHull.add(system2)
        self.assertEqual(len(self.convexHull.elements), 2)

        ans = self.convexHull[system3]
        self.assertTrue(np.isinf(ans) and ans < 0)
        self.convexHull.add(system3)
        self.assertEqual(len(self.convexHull.elements), 3)

        ans = self.convexHull[system4]
        self.assertTrue(ans > 0)
        self.convexHull.add(system4)
        self.assertEqual(len(self.convexHull.elements), 3)

        ans = self.convexHull[system5]
        self.assertTrue(ans < 0)
        self.convexHull.add(system5)
        self.assertEqual(len(self.convexHull.elements), 4)

        ans = self.convexHull[system6]
        self.assertTrue(np.isinf(ans) and ans < 0)
        self.convexHull.add(system6)
        self.assertEqual(len(self.convexHull.elements), 5)

    def test_bicomponent3(self):
        compositionSpace = CompositionSpace(symbols=['Mo', 'B'], blocks=[[1, 0], [0, 1]], range=[[0, 18], [0, 18]],
                                 minAt=8, maxAt=18)
        self.convexHull = ConvexHull(compositionSpace)
        system1 = System({'Mo': 5, 'B': 15}, -173.325)
        system2 = System({'Mo': 5}, -42.944)
        system3 = System({'Mo': 3, 'B': 9}, -104.041)
        system4 = System({'Mo': 6, 'B': 14}, -177.431)
        system5 = System({'Mo': 9}, -69.214)

        ans = self.convexHull[system1]
        self.assertTrue(np.isinf(ans) and ans < 0)
        self.convexHull.add(system1)
        self.assertEqual(len(self.convexHull.elements), 1)

        ans = self.convexHull[system2]
        self.assertTrue(np.isinf(ans) and ans < 0)
        self.convexHull.add(system2)
        self.assertEqual(len(self.convexHull.elements), 2)

        ans = self.convexHull[system3]
        self.assertTrue(ans < 0)
        self.convexHull.add(system3)
        self.assertEqual(len(self.convexHull.elements), 2)

        ans = self.convexHull[system4]
        self.assertTrue(ans < 0)
        self.convexHull.add(system4)
        self.assertEqual(len(self.convexHull.elements), 3)

        ans = self.convexHull[system5]
        self.assertTrue(ans > 0)
        self.convexHull.add(system5)
        self.assertEqual(len(self.convexHull.elements), 3)
