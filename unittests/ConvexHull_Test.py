"""
USPEX.Common.Atomistic.unittests.ConvexHull_Test
================================================

Class for ConvexHull testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import unittest
import numpy as np


from scipy.spatial import ConvexHull
from typing import List


from ..ConvexHull import ConvexHull, Simplex

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


class Simplex_Test2(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.coords = np.array([[0.8, 0.0], [1.0, 0.0], [0.8, 0.2]])
        cls.energies = np.array([-8.0, -6.5, -7.5])
        cls.simplex = Simplex(cls.coords)

    def test_out(self):
        coords = [[0.0,0.0], [0.0,0.2], [1.2,0.0], [0.8, -1.0], [0.8, 1.0]]
        print('Out of simplex:\n')
        for x in coords:
            y = self.simplex.bary_coords(x)
            y = list(filter(lambda x: not np.isclose(x, 0.0), y))
            self.assertNotEqual(np.abs(np.sum(np.sign(y))), len(y))

    def test_vertices(self):
        print('Out of vertices:\n')
        for x in self.coords:
            y = self.simplex.bary_coords(x)
            y = list(filter(lambda x: not np.isclose(x, 0.0), y))
            self.assertEqual(np.abs(np.sum(np.sign(y))), len(y))

    def test_edges(self):
        coords = [[0.8, 0.1], [0.9, 0.0], [0.9, 0.1]]
        print('Edges:\n')
        for x in coords:
            y = self.simplex.bary_coords(x)
            y = list(filter(lambda x: not np.isclose(x, 0.0), y))
            self.assertEqual(np.abs(np.sum(np.sign(y))), len(y))
