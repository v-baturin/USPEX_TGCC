'''
@file        GCH_Test.py
@author:     Artem Samtsevich
@copyright:  2019 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        16 September 2019
@brief       Class for GeneralizedConvexHull testing
'''


import numpy as np
import pandas as pd
import os
import unittest

from ase.io.vasp import read_vasp
from os.path import join as pj

from ..CompositionSpace import CompositionSpace
from ..Crystal import Crystal
from ..GCH import GeneralizedConvexHull

TESTPATH = os.path.dirname(os.path.abspath(__file__))

Si_gch_path = pj(TESTPATH, 'Si_gch_test')
FeC_gch_path = pj(TESTPATH, 'FeC_gch_test')


def read_structures_and_energies(folder : str):
    with open(pj(folder, 'Individuals'), 'r') as fp:
        info = fp.readlines()[2:]
    DATA = pd.DataFrame(columns=['Generation', 'ID', 'composition', 'enthalpy'], dtype=int)
    all_systems = []
    try:
        with open(pj(folder, 'gatheredPOSCARS'), 'r') as fp:
            while True:
                tmp = read_vasp(fp)
                system = Crystal(symbols=tmp.get_chemical_symbols(),
                                 scaled_positions=tmp.get_scaled_positions(),
                                 cell=tmp.get_cell_lengths_and_angles())
                all_systems.append(system)
    except:
        print('Reading of the pathway has finished.')
    assert len(all_systems)

    populations = []
    for i, (_info, system) in enumerate(zip(info, all_systems)):
        tmp = _info.split()
        gen = int(tmp[0])
        ID = int(tmp[1])
        _b, _e = tmp.index('['), tmp.index(']')
        composition = [int(x) for x in tmp[_b+1:_e]]
        enthalpy = float(tmp[_e+1])
        DATA.loc[i] = gen, ID, composition, enthalpy
        system.ID = ID
        system.enthalpy = enthalpy

    for gen in np.unique(DATA.Generation.astype(int)):
        ids = DATA[DATA.Generation == gen]['ID'].astype(int)
        populations.append([system for system in all_systems if system.ID in ids])
    return DATA, populations, all_systems



class GenConvexHull_Si_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, cls.populations, cls.all_systems = read_structures_and_energies(folder=Si_gch_path)
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
        indices = np.argsort(np.fromiter((x.enthalpy for x in init_systems), dtype=float))
        lowest_energy_structure_index = indices[0]
        highest_energy_structure_index = indices[-1]
        self.assertEqual(self.convexHull_1.height[lowest_energy_structure_index], 0)
        self.assertEqual(self.convexHull_1.depth[highest_energy_structure_index], 0)
        # self.assertTrue(lowest_energy_structure in self.convexHull_1.lower_bound)
        # self.assertTrue(highest_energy_structure in self.convexHull_1.upper_bound)

    def test_is_on_CH(self):
        self.convexHull = GeneralizedConvexHull([x for x in self.all_systems], config=self.config)
        indices = np.argsort(np.fromiter((x.enthalpy for x in self.all_systems), dtype=float))
        lowest_energy_structure_index = indices[0]
        highest_energy_structure_index = indices[-1]
        self.assertEqual(self.convexHull.height[lowest_energy_structure_index], 0)
        self.assertEqual(self.convexHull.depth[highest_energy_structure_index], 0)
        # self.assertTrue(lowest_energy_structure in self.convexHull.lower_bound)
        # self.assertTrue(highest_energy_structure in self.convexHull.upper_bound)


class GenConvexHull_FeC_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, cls.populations, cls.all_systems = read_structures_and_energies(folder=FeC_gch_path)
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
        indices = np.argsort(np.fromiter((x.enthalpy for x in init_systems), dtype=float))
        lowest_energy_structure_index = indices[0]
        highest_energy_structure_index = indices[-1]
        self.assertEqual(self.convexHull_1.height[lowest_energy_structure_index], 0)
        self.assertEqual(self.convexHull_1.depth[highest_energy_structure_index], 0)
        # self.assertTrue(lowest_energy_structure in self.convexHull_1.lower_bound)
        # self.assertTrue(highest_energy_structure in self.convexHull_1.upper_bound)

    def test_is_on_CH(self):
        self.convexHull = GeneralizedConvexHull([x for x in self.all_systems], config=self.config)
        indices = np.argsort(np.fromiter((x.enthalpy for x in self.all_systems), dtype=float))
        lowest_energy_structure_index = indices[0]
        highest_energy_structure_index = indices[-1]
        self.assertEqual(self.convexHull.height[lowest_energy_structure_index], 0)
        self.assertEqual(self.convexHull.depth[highest_energy_structure_index], 0)
        # self.assertTrue(lowest_energy_structure in self.convexHull.lower_bound)
        # self.assertTrue(highest_energy_structure in self.convexHull.upper_bound)


