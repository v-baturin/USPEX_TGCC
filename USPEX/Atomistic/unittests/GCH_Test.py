'''
@file        GCH_Test.py
@author:     Artem Samtsevich
@copyright:  2019 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        16 September 2019
@brief       Class for GeneralizedConvexHull testing
'''


import numpy as np
import unittest

from pathlib import Path

from ..GCH import GeneralizedConvexHull
from ...Optimizers.PoolEntry import PoolEntry, EntryFlavour
from ...components import RadialDistributionUtility, CompositionSpace, Atomistic

TESTPATH = Path(__file__).parent

Si_gch_path = TESTPATH/'Si_gch_test'
FeC_gch_path = TESTPATH/'FeC_gch_test'


def read_structures_and_energies(symbols, folder: Path):
    with open(folder/'Individuals', 'r') as fp:
        info = fp.readlines()[2:]
    all_systems = Atomistic.readAtomicStructures(folder/'gatheredPOSCARS')
    assert all_systems
    radialDistributionUtility = RadialDistributionUtility(symbols=symbols, suffix='origin')
    atomistic = Atomistic()
    extensions = dict(
        atomistic=Atomistic.propertyExtension(atomistic),
        radialDistributionUtility=radialDistributionUtility.propertyExtension(radialDistributionUtility)
    )

    generations = []
    IDs = []
    populations = []
    systems = []
    for _info, system in zip(info, all_systems):
        tmp = _info.split()
        gen = int(tmp[0])
        ID = int(tmp[1])
        _b, _e = tmp.index('['), tmp.index(']')
        # composition = [int(x) for x in tmp[_b+1:_e]]
        enthalpy = float(tmp[_e+1])
        generations.append(gen)
        IDs.append(ID)
        system['ID'] = ID
        system['isBad'] = False
        system['.enthalpy'] = enthalpy
        system = PoolEntry(ID, EntryFlavour(extensions=extensions, **system))
        system.getProperty('structure', extension='atomistic')
        systems.append(system)
    all_systems = systems

    generations = np.asarray(generations, dtype=int)
    IDs = np.asarray(IDs, dtype=int)
    for gen in np.unique(generations):
        ids = IDs[np.where(generations == gen)]
        populations.append([system for system in all_systems if system['ID'] in ids])
    return populations, all_systems


class GenConvexHull_Si_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.populations, cls.all_systems = read_structures_and_energies(symbols=['Si'], folder=Si_gch_path)
        cls.config = CompositionSpace(symbols=['Si'], blocks=[[8]], range=[[1, 1]])
        # All systems will be added to the convex hull at one moment.
        # Systems will be added to the convex hull step by step.


    def test_is_on_CH_short(self):
        system = self.all_systems[0]
        self.convexHull_short = GeneralizedConvexHull([system], config=self.config)
        self.assertEqual(self.convexHull_short.height[0], 0)
        self.assertEqual(self.convexHull_short.depth[0], 0)
        self.assertTrue(system in self.convexHull_short.lower_bound)
        self.assertTrue(system in self.convexHull_short.upper_bound)

    def test_is_on_CH1(self):
        init_systems = self.populations[0]
        self.convexHull_1 = GeneralizedConvexHull([system for system in init_systems], config=self.config)
        indices = np.argsort(np.fromiter((x['.enthalpy.origin'] for x in init_systems), dtype=float))
        lowest_energy_structure_index = indices[0]
        highest_energy_structure_index = indices[-1]
        self.assertEqual(self.convexHull_1.height[lowest_energy_structure_index], 0)
        self.assertEqual(self.convexHull_1.depth[highest_energy_structure_index], 0)
        # self.assertTrue(lowest_energy_structure in self.convexHull_1.lower_bound)
        # self.assertTrue(highest_energy_structure in self.convexHull_1.upper_bound)

    def test_is_on_CH(self):
        self.convexHull = GeneralizedConvexHull([x for x in self.all_systems], config=self.config)
        indices = np.argsort(np.fromiter((x['.enthalpy.origin'] for x in self.all_systems), dtype=float))
        lowest_energy_structure_index = indices[0]
        highest_energy_structure_index = indices[-1]
        self.assertEqual(self.convexHull.height[lowest_energy_structure_index], 0)
        self.assertEqual(self.convexHull.depth[highest_energy_structure_index], 0)
        # self.assertTrue(lowest_energy_structure in self.convexHull.lower_bound)
        # self.assertTrue(highest_energy_structure in self.convexHull.upper_bound)


class GenConvexHull_FeC_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.populations, cls.all_systems = read_structures_and_energies(symbols=['Fe', 'C'], folder=FeC_gch_path)
        cls.config = CompositionSpace(symbols=['Fe', 'C'], blocks=[[3,1]], range=[[1, 10]], minAt=4, maxAt=40)

    def test_is_on_CH_short(self):
        system = self.all_systems[0]
        self.convexHull_short = GeneralizedConvexHull([system], config=self.config)
        self.assertEqual(self.convexHull_short.height[0], 0)
        self.assertEqual(self.convexHull_short.depth[0], 0)
        self.assertTrue(system in self.convexHull_short.lower_bound)
        self.assertTrue(system in self.convexHull_short.upper_bound)

    def test_is_on_CH1(self):
        init_systems = self.populations[0]
        self.convexHull_1 = GeneralizedConvexHull([system for system in init_systems], config=self.config)
        indices = np.argsort(np.fromiter((x['.enthalpy.origin'] for x in init_systems), dtype=float))
        lowest_energy_structure_index = indices[0]
        highest_energy_structure_index = indices[-1]
        self.assertEqual(self.convexHull_1.height[lowest_energy_structure_index], 0)
        self.assertEqual(self.convexHull_1.depth[highest_energy_structure_index], 0)
        # self.assertTrue(lowest_energy_structure in self.convexHull_1.lower_bound)
        # self.assertTrue(highest_energy_structure in self.convexHull_1.upper_bound)

    def test_is_on_CH(self):
        self.convexHull = GeneralizedConvexHull([x for x in self.all_systems], config=self.config)
        indices = np.argsort(np.fromiter((x['.enthalpy.origin'] for x in self.all_systems), dtype=float))
        lowest_energy_structure_index = indices[0]
        highest_energy_structure_index = indices[-1]
        self.assertEqual(self.convexHull.height[lowest_energy_structure_index], 0)
        self.assertEqual(self.convexHull.depth[highest_energy_structure_index], 0)
        # self.assertTrue(lowest_energy_structure in self.convexHull.lower_bound)
        # self.assertTrue(highest_energy_structure in self.convexHull.upper_bound)


