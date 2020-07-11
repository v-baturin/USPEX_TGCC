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
from scipy.spatial import ConvexHull
from typing import List

from ..CompositionSpace import CompositionSpace
from ..Crystal import Crystal
from ..GCH import GeneralizedConvexHull
from ...ConvexHull import Simplex

TESTPATH = os.path.dirname(os.path.abspath(__file__))

Si_gch_path = pj(TESTPATH, 'Si_gch_test')
FeC_gch_path = pj(TESTPATH, 'FeC_gch_test')
DF_NAME_FORMAT = 'gch_{}.dump'



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
        df_short = DF_NAME_FORMAT.format('Si_short')
        if os.path.exists(df_short):
            os.remove(df_short)
        self.convexHull_short = GeneralizedConvexHull(config=self.config)
        system = self.all_systems[0]
        self.convexHull_short.extend([system])
        # self.assertTrue(system in self.convexHull_short.lower_bound)
        # self.assertTrue(system in self.convexHull_short.upper_bound)

    def test_is_on_CH1(self):
        df_1 = pj(TESTPATH, DF_NAME_FORMAT.format(f'Si_{1}'))
        if os.path.exists(df_1):
            os.remove(df_1)

        init_systems = self.populations[0]
        self.convexHull_1 = GeneralizedConvexHull(config=self.config)
        self.convexHull_1.extend([system for system in init_systems])
        # lowest_energy_structure = sorted(init_systems, key=lambda x: x.enthalpy)[0]
        # highest_energy_structure = sorted(init_systems, key=lambda x: x.enthalpy)[-1]
        # self.assertTrue(lowest_energy_structure in self.convexHull_1.lower_bound)
        # self.assertTrue(highest_energy_structure in self.convexHull_1.upper_bound)

    def test_is_on_CH(self):
        df_0 = DF_NAME_FORMAT.format(f'Si_{0}')
        if os.path.exists(df_0):
            os.remove(df_0)

        self.convexHull = GeneralizedConvexHull(config=self.config)
        self.convexHull.extend([x for x in self.all_systems])
        # lowest_energy_structure = sorted(self.all_systems, key=lambda x: x.enthalpy)[0]
        # highest_energy_structure = sorted(self.all_systems, key=lambda x: x.enthalpy)[-1]
        # self.assertTrue(lowest_energy_structure in self.convexHull.lower_bound)
        # self.assertTrue(highest_energy_structure in self.convexHull.upper_bound)


class GenConvexHull_FeC_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data, cls.populations, cls.all_systems = read_structures_and_energies(folder=FeC_gch_path)
        cls.config = CompositionSpace(symbols=['Fe', 'C'], blocks=[[3,1]], range=[[1, 10]], minAt=4, maxAt=40)

    def test_is_on_CH_short(self):
        df_short = pj(TESTPATH, DF_NAME_FORMAT.format('FeC_short'))
        if os.path.exists(df_short):
            os.remove(df_short)
        self.convexHull_short = GeneralizedConvexHull(config=self.config)
        system = self.all_systems[0]
        self.convexHull_short.extend([system])
        # self.assertTrue(system in self.convexHull_short.lower_bound)
        # self.assertTrue(system in self.convexHull_short.upper_bound)

    def test_is_on_CH1(self):
        df_1 = pj(TESTPATH, DF_NAME_FORMAT.format(f'FeC_{1}'))
        if os.path.exists(df_1):
            os.remove(df_1)
        init_systems = self.populations[0]
        self.convexHull_1 = GeneralizedConvexHull(config=self.config)
        self.convexHull_1.extend([system for system in init_systems])
        # lowest_energy_structure = sorted(init_systems, key=lambda x: x.enthalpy)[0]
        # highest_energy_structure = sorted(init_systems, key=lambda x: x.enthalpy)[-1]
        # self.assertTrue(lowest_energy_structure in self.convexHull_1.lower_bound)
        # self.assertTrue(highest_energy_structure in self.convexHull_1.upper_bound)

    def test_is_on_CH(self):
        df_0 = pj(TESTPATH, DF_NAME_FORMAT.format(f'FeC_{0}'))
        if os.path.exists(df_0):
            os.remove(df_0)
        self.convexHull = GeneralizedConvexHull(config=self.config)
        self.convexHull.extend([x for x in self.all_systems])
        # lowest_energy_structure = sorted(self.all_systems, key=lambda x: x.enthalpy)[0]
        # highest_energy_structure = sorted(self.all_systems, key=lambda x: x.enthalpy)[-1]
        # self.assertTrue(lowest_energy_structure in self.convexHull.lower_bound)
        # self.assertTrue(highest_energy_structure in self.convexHull.upper_bound)


square = np.array([[2,4], [4,12], [12,10], [10,2]])

class ConvexHull_Square_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.qhull = ConvexHull(square)
        # plt.plot(square[:, 0], square[:, 1], 'o')
        # for simplex in qhull.simplices:
        #     plt.plot(principal_component_set[simplex, 0], principal_component_set[simplex, 1], 'k-')
        # plt.show()

        cls.simplecies = []
        cls.energies = []
        for facet in cls.qhull.simplices:
            cls.simplecies.append(Simplex(np.array([square[x][1:] for x in facet])))
            cls.energies.append(np.array([square[x][0] for x in facet]))

    def get_heights(self, point) -> List[float]:
        '''
        For GCH first number is energy, next - coordinates
        :param point:
        :return:
        '''
        energy, coord = point[0], point[1:]
        heights = []
        for e, s in zip(self.energies, self.simplecies):
            y = s.bary_coords(coord)
            if np.abs(np.sum(np.sign(y))) == len(y) or np.any(np.isclose(y, 0.0)):
                # composition = self.config.numBlocks(row['Structure'].composition)
                heights.append(np.around(np.dot(y, e) - energy, decimals=6))
        return heights

    def test_inside1(self):
        p1 = [8,4]
        print(f'test_indside1 of point {p1}')
        p1_heights = self.get_heights(p1)
        print(f'{p1_heights}\n===================')
        self.assertTrue(np.isclose(np.abs(np.min(p1_heights)), 6.0))

    def test_inside2(self):
        p1 = [8,8]
        print(f'test_indside2 of point {p1}')
        p1_heights = self.get_heights(p1)
        print(f'{p1_heights}\n===================')
        self.assertTrue(np.isclose(np.abs(np.min(p1_heights)), 5.0))

    def test_vertex(self):
        p1 = [2,4]
        print(f'test_vertex of point {p1}')
        p1_heights = self.get_heights(p1)
        print(f'{p1_heights}\n===================')
        self.assertTrue(np.isclose(np.abs(np.min(p1_heights)), 0.0))

    def test_on_lower_edge(self):
        p1 = [3,8]
        print(f'test_lower_edge of point {p1}')
        p1_heights = self.get_heights(p1)
        print(f'{p1_heights}\n===================')
        self.assertTrue(np.isclose(np.abs(np.min(p1_heights)), 0.0))

    def test_on_top_edge(self):
        p1 = [11,6]
        print(f'test_top_edge of point {p1}')
        p1_heights = self.get_heights(p1)
        print(f'{p1_heights}\n===================')
        self.assertTrue(np.isclose(np.abs(np.min(p1_heights)), 8.5))

    def test_outside(self):
        p1 = [12,6]
        print(f'test_outside of point {p1}')
        p1_heights = self.get_heights(p1)
        print(f'{p1_heights}\n===================')
        self.assertTrue(np.isclose(np.abs(np.min(p1_heights)), 9.5))


class Simplex_Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simplicies = [Simplex(np.array([square[-1][1:], square[0][1:]]))]
        cls.energies = [np.array([square[-1][0], square[0][0]])]
        for v1, v2 in zip(square[:-1], square[1:]):
            cls.simplicies.append(Simplex(np.array([v1[1:], v2[1:]])))
            cls.energies.append(np.array([v1[0], v2[0]]))

    def get_heights(self, point) -> List[float]:
        '''
        For GCH first number is energy, next - coordinates
        :param point:
        :return:
        '''
        energy, coord = point[0], point[1:]
        heights = []
        for e, s in zip(self.energies, self.simplicies):
            y = s.bary_coords(coord)
            if np.abs(np.sum(np.sign(y))) == len(y) or np.any(np.isclose(y, 0.0)):
                # composition = self.config.numBlocks(row['Structure'].composition)
                heights.append(np.around(np.dot(y, e) - energy, decimals=6))
        return heights

    def test_inside1(self):
        p1 = [8,4]
        print(f'test_indside1 of point {p1}')
        p1_heights = self.get_heights(p1)
        print(f'{p1_heights}\n===================')

    def test_inside2(self):
        p1 = [8,8]
        print(f'test_indside2 of point {p1}')
        p1_heights = self.get_heights(p1)
        print(f'{p1_heights}\n===================')

    def test_vertex(self):
        p1 = [2,4]
        print(f'test_vertex of point {p1}')
        p1_heights = self.get_heights(p1)
        print(f'{p1_heights}\n===================')

    def test_on_lower_edge(self):
        p1 = [3,8]
        print(f'test_lower_edge of point {p1}')
        p1_heights = self.get_heights(p1)
        print(f'{p1_heights}\n===================')

    def test_on_top_edge(self):
        p1 = [11,6]
        print(f'test_top_edge of point {p1}')
        p1_heights = self.get_heights(p1)
        print(f'{p1_heights}\n===================')

    def test_outside(self):
        p1 = [12,6]
        print(f'test_outside of point {p1}')
        p1_heights = self.get_heights(p1)
        print(f'{p1_heights}\n===================')
