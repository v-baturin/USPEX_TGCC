import unittest
import numpy as np

from ..CompositionCH import CompositionCH
from USPEX.Atomistic.Primitives.AtomicStructure import AtomicStructure
from ...Optimizers.PoolEntry import EntryFlavour, PoolEntry
from ...components import CompositionSpace, SimpleMoleculeUtility


class CompostionCH_Test(unittest.TestCase):

    def setUp(self) -> None:
        PoolEntry.createEngine(':memory:')

    def test_unocomponent(self):
        compositionSpace = CompositionSpace(symbols=['Mo'], blocks=[[1]], range=[[1, 18]])
        simpleMoleculeUtility = SimpleMoleculeUtility()
        extensions = dict(
            simpleMoleculeUtility=simpleMoleculeUtility.propertyExtension(simpleMoleculeUtility)
        )

        system0 = {'ID': 0, 'isBad': False, '.enthalpy': -2.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] ]}
        system0 = PoolEntry(0, EntryFlavour(extensions=extensions, **system0))
        system1 = {'ID': 1, 'isBad': False, '.enthalpy': -8.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 4]}
        system1 = PoolEntry(1, EntryFlavour(extensions=extensions, **system1))
        system2 = {'ID': 2, 'isBad': False, '.enthalpy': -16.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 4]}
        system2 = PoolEntry(2, EntryFlavour(extensions=extensions, **system2))
        system3 = {'ID': 3, 'isBad': False, '.enthalpy': -8.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 8]}
        system3 = PoolEntry(3, EntryFlavour(extensions=extensions, **system3))
        self.convexHull = CompositionCH([system0], compositionSpace)

        # ans = self.convexHull.height[0]
        # self.assertTrue(np.isinf(ans) and ans > 0)
        self.assertEqual(len(self.convexHull.lower_bound), 1)

        self.convexHull.extend([system2])
        self.assertEqual(len(self.convexHull.lower_bound), 1)
        self.assertEqual(set(s['ID'] for s in self.convexHull.lower_bound), {2})
        self.assertAlmostEqual(self.convexHull.height[0], 2.0)
        self.assertEqual(set(s['ID'] for s in self.convexHull.upper_bound), {0})
        self.assertAlmostEqual(self.convexHull.depth[1], -2.0)

        self.convexHull.extend([system3])
        self.assertEqual(len(self.convexHull.lower_bound), 1)
        self.assertEqual(set(s['ID'] for s in self.convexHull.lower_bound), {2})
        self.assertAlmostEqual(self.convexHull.height[0], 2.0)
        self.assertAlmostEqual(self.convexHull.height[2], 3.0)
        self.assertEqual(set(s['ID'] for s in self.convexHull.upper_bound), {3})
        self.assertAlmostEqual(self.convexHull.depth[0], -1.0)
        self.assertAlmostEqual(self.convexHull.depth[1], -3.0)

    def test_bicomponent(self):
        compositionSpace = CompositionSpace(symbols=['Mo', 'B'], blocks=[[1, 0], [0, 1]], range=[[0, 18], [0, 18]],
                                            minAt=8, maxAt=18)
        simpleMoleculeUtility = SimpleMoleculeUtility()
        extensions = dict(
            simpleMoleculeUtility=simpleMoleculeUtility.propertyExtension(simpleMoleculeUtility)
        )
        system1 = {'ID': 0, 'isBad': False, '.enthalpy': -5.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 4 + ['B'] * 10]}
        system1 = PoolEntry(0, EntryFlavour(extensions=extensions, **system1))
        system2 = {'ID': 1, 'isBad': False, '.enthalpy': -14.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 4 + ['B'] * 10]}
        system2 = PoolEntry(1, EntryFlavour(extensions=extensions, **system2))
        system3 = {'ID': 2, 'isBad': False, '.enthalpy': -8.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 8 + ['B'] * 8]}
        system3 = PoolEntry(2, EntryFlavour(extensions=extensions, **system3))
        system4 = {'ID': 3, 'isBad': False, '.enthalpy': -2.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 8]}
        system4 = PoolEntry(3, EntryFlavour(extensions=extensions, **system4))
        system5 = {'ID': 4, 'isBad': False, '.enthalpy': -4.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 8]}
        system5 = PoolEntry(4, EntryFlavour(extensions=extensions, **system5))
        system6 = {'ID': 5, 'isBad': False, '.enthalpy': -12.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['B'] * 10]}
        system6 = PoolEntry(5, EntryFlavour(extensions=extensions, **system6))
        system7 = {'ID': 6, 'isBad': False, '.enthalpy': -2.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 6 + ['B'] * 4]}
        system7 = PoolEntry(6, EntryFlavour(extensions=extensions, **system7))
        system8 = {'ID': 7, 'isBad': False, '.enthalpy': -16.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 4 + ['B'] * 6]}
        system8 = PoolEntry(7, EntryFlavour(extensions=extensions, **system8))



        # ans = self.convexHull.height[0]
        # self.assertTrue(np.isinf(ans) and ans > 0)
        self.convexHull = CompositionCH([system1], compositionSpace)
        self.assertEqual(len(self.convexHull.lower_bound), 1)

        # ans = self.convexHull.height[1]
        # self.assertTrue(np.isinf(ans) and ans > 0)
        self.convexHull.extend([system2])
        self.assertEqual(len(self.convexHull.lower_bound), 1)
        # self.assertEqual(self.convexHull.elements[0].enthalpy, -14.0)

        ans = self.convexHull.height[0]
        self.assertAlmostEqual(ans, 9/14)
        # self.convexHull.extend([system1])
        # self.assertEqual(len(self.convexHull.lower_bound), 2)
        # self.assertEqual(self.convexHull.elements[0].enthalpy, -14.0)

        # ans = self.convexHull.height[2]
        # self.assertTrue(np.isinf(ans) and ans > 0)
        self.convexHull.extend([system3])
        self.assertEqual(len(self.convexHull.lower_bound), 2)
        self.assertEqual(set(s['ID'] for s in self.convexHull.lower_bound), {1, 2})
        self.assertAlmostEqual(self.convexHull.height[0], 0.642857)

        # ans = self.convexHull.height[3]
        # self.assertTrue(np.isinf(ans) and ans > 0)
        self.convexHull.extend([system4])
        self.assertEqual(len(self.convexHull.lower_bound), 2)
        self.assertEqual(set(s['ID'] for s in self.convexHull.lower_bound), {1,3})
        self.assertAlmostEqual(self.convexHull.height[0], 0.642857)
        self.assertAlmostEqual(self.convexHull.height[2], 0.275)

        # ans = self.convexHull.height[4]
        # self.assertTrue(np.isinf(ans) and ans > 0)
        self.convexHull.extend([system5])
        self.assertEqual(len(self.convexHull.lower_bound), 2)
        self.assertEqual(set(s['ID'] for s in self.convexHull.lower_bound), {1, 4})
        self.assertAlmostEqual(self.convexHull.height[0], 0.642857)
        self.assertAlmostEqual(self.convexHull.height[2], 0.35)
        self.assertAlmostEqual(self.convexHull.height[3], 0.25)

        # self.convexHull.extend([system4])
        # self.assertEqual(len(self.convexHull.lower_bound), 2)

        # ans = self.convexHull.height[5]
        # self.assertTrue(np.isinf(ans) and ans > 0)
        self.convexHull.extend([system6])
        self.assertEqual(len(self.convexHull.lower_bound), 3)
        self.assertEqual(set(s['ID'] for s in self.convexHull.lower_bound), {1, 4, 5})
        self.assertAlmostEqual(self.convexHull.height[0], 0.642857)
        self.assertAlmostEqual(self.convexHull.height[2], 0.35)
        self.assertAlmostEqual(self.convexHull.height[3], 0.25)

        # self.convexHull.extend([system2])
        # self.assertEqual(len(self.convexHull.lower_bound), 3)

        self.convexHull.extend([system7])
        self.assertEqual(len(self.convexHull.lower_bound), 3)
        self.assertEqual(set(s['ID'] for s in self.convexHull.lower_bound), {1, 4, 5})
        self.assertAlmostEqual(self.convexHull.height[0], 0.642857)
        self.assertAlmostEqual(self.convexHull.height[2], 0.35)
        self.assertAlmostEqual(self.convexHull.height[3], 0.25)
        self.assertAlmostEqual(self.convexHull.height[6], 0.58)

        self.convexHull.extend([system8])
        self.assertEqual(len(self.convexHull.lower_bound), 3)
        self.assertEqual(set(s['ID'] for s in self.convexHull.lower_bound), {4, 5, 7})
        self.assertAlmostEqual(self.convexHull.height[0], 1.128571)
        self.assertAlmostEqual(self.convexHull.height[1], 0.485714)
        self.assertAlmostEqual(self.convexHull.height[2], 0.916667)
        self.assertAlmostEqual(self.convexHull.height[3], 0.25)
        self.assertAlmostEqual(self.convexHull.height[6], 1.033333)

    def test_bicomponent1(self):
        compositionSpace = CompositionSpace(symbols=['Mo', 'B'], blocks=[[1, 0], [0, 1]], range=[[0, 18], [0, 18]],
                                            minAt=8, maxAt=18)
        simpleMoleculeUtility = SimpleMoleculeUtility()
        extensions = dict(
            simpleMoleculeUtility=simpleMoleculeUtility.propertyExtension(simpleMoleculeUtility)
        )
        system1 = {'ID': 0, 'isBad': False, '.enthalpy': -5.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 4 + ['B'] * 10]}
        system1 = PoolEntry(0, EntryFlavour(extensions=extensions, **system1))
        system2 = {'ID': 1, 'isBad': False, '.enthalpy': -14.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 4 + ['B'] * 10]}
        system2 = PoolEntry(1, EntryFlavour(extensions=extensions, **system2))
        system3 = {'ID': 2, 'isBad': False, '.enthalpy': -8.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 8 + ['B'] * 8]}
        system3 = PoolEntry(2, EntryFlavour(extensions=extensions, **system3))
        system4 = {'ID': 3, 'isBad': False, '.enthalpy': -2.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 8]}
        system4 = PoolEntry(3, EntryFlavour(extensions=extensions, **system4))
        system5 = {'ID': 4, 'isBad': False, '.enthalpy': -4.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 8]}
        system5 = PoolEntry(4, EntryFlavour(extensions=extensions, **system5))
        system6 = {'ID': 5, 'isBad': False, '.enthalpy': -12.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['B'] * 10]}
        system6 = PoolEntry(5, EntryFlavour(extensions=extensions, **system6))
        system7 = {'ID': 6, 'isBad': False, '.enthalpy': -2.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 6 + ['B'] * 4]}
        system7 = PoolEntry(6, EntryFlavour(extensions=extensions, **system7))
        system8 = {'ID': 7, 'isBad': False, '.enthalpy': -16.0,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 4 + ['B'] * 6]}
        system8 = PoolEntry(7, EntryFlavour(extensions=extensions, **system8))

        systems = [system1, system2, system3, system4, system5, system6, system7, system8]
        self.convexHull = CompositionCH(systems, compositionSpace)

        self.assertEqual(set(s['ID'] for s in self.convexHull.lower_bound), {4, 5, 7})
        self.assertEqual(set(s['ID'] for s in self.convexHull.upper_bound), {0, 3, 5, 6})

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
        compositionSpace = CompositionSpace(symbols=['Mo', 'B'], blocks=[[1, 0], [0, 1]], range=[[0, 18], [0, 18]],
                                            minAt=8, maxAt=18)
        simpleMoleculeUtility = SimpleMoleculeUtility()

        extensions = dict(
            simpleMoleculeUtility=simpleMoleculeUtility.propertyExtension(simpleMoleculeUtility)
        )
        system1 = {'ID': 0, 'isBad': False, '.enthalpy': -178.845,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 6 + ['B'] * 14]}
        system1 = PoolEntry(0, EntryFlavour(extensions=extensions, **system1))
        system2 = {'ID': 1, 'isBad': False, '.enthalpy': -225.103,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 16 + ['B'] * 10]}
        system2 = PoolEntry(1, EntryFlavour(extensions=extensions, **system2))
        system3 = {'ID': 2, 'isBad': False, '.enthalpy': -162.761,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 5 + ['B'] * 14]}
        system3 = PoolEntry(2, EntryFlavour(extensions=extensions, **system3))
        system4 = {'ID': 3, 'isBad': False, '.enthalpy': -176.250,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 6 + ['B'] * 14]}
        system4 = PoolEntry(3, EntryFlavour(extensions=extensions, **system4))
        system5 = {'ID': 4, 'isBad': False, '.enthalpy': -150.735,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 13 + ['B'] * 5]}
        system5 = PoolEntry(4, EntryFlavour(extensions=extensions, **system5))
        system6 = {'ID': 5, 'isBad': False, '.enthalpy': -77.661,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 9 + ['B'] * 1]}
        system6 = PoolEntry(5, EntryFlavour(extensions=extensions, **system6))
        systems = [system1, system2, system3, system4, system5, system6]

        self.convexHull = CompositionCH(systems, compositionSpace)

        self.assertEqual(set(s['ID'] for s in self.convexHull.lower_bound), {0,1,2,4,5})
        self.assertEqual(set(s['ID'] for s in self.convexHull.upper_bound), {2,5})

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
        compositionSpace = CompositionSpace(symbols=['Mo', 'B'], blocks=[[1, 0], [0, 1]], range=[[0, 18], [0, 18]],
                                            minAt=8, maxAt=18)
        simpleMoleculeUtility = SimpleMoleculeUtility()

        extensions = dict(
            simpleMoleculeUtility=simpleMoleculeUtility.propertyExtension(simpleMoleculeUtility)
        )
        system1 = {'ID': 0, 'isBad': False, '.enthalpy': -173.325,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 5 + ['B'] * 15]}
        system1 = PoolEntry(0, EntryFlavour(extensions=extensions, **system1))
        system2 = {'ID': 1, 'isBad': False, '.enthalpy': -42.944,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 5]}
        system2 = PoolEntry(1, EntryFlavour(extensions=extensions, **system2))
        system3 = {'ID': 2, 'isBad': False, '.enthalpy': -104.041,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 3 + ['B'] * 9]}
        system3 = PoolEntry(2, EntryFlavour(extensions=extensions, **system3))
        system4 = {'ID': 3, 'isBad': False, '.enthalpy': -177.431,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 6 + ['B'] * 14]}
        system4 = PoolEntry(3, EntryFlavour(extensions=extensions, **system4))
        system5 = {'ID': 4, 'isBad': False, '.enthalpy': -69.214,
                   'atomistic.molecules': [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mo'] * 9]}
        system5 = PoolEntry(4, EntryFlavour(extensions=extensions, **system5))
        systems = [system1, system2, system3, system4, system5]

        self.convexHull = CompositionCH(systems, compositionSpace)

        self.assertEqual(set(s['ID'] for s in self.convexHull.lower_bound), {1,2,3})
        self.assertEqual(set(s['ID'] for s in self.convexHull.upper_bound), {0,4})

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
