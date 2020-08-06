"""
USPEX.Common.Atomistic.unittests.FitnessHull_Test
================================================

Class for Fitness testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import unittest
import numpy as np
import os
import json

from ..Fitness import Fitness


class System(object):
    def __init__(self, ID: int, composition: dict, enthalpy: float):
        self.ID = ID
        self.composition = composition
        self.enthalpy = enthalpy


class Fitness_Test(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = [System(0, {'Mg': 4, 'Al': 8, 'O': 16}, -646.695),
                     System(1, {'Mg': 4, 'Al': 8, 'O': 16}, -644.480),
                     System(2, {'Mg': 4, 'Al': 8, 'O': 16}, -650.098),
                     System(3, {'Mg': 4, 'Al': 8, 'O': 16}, -649.082),
                     System(4, {'Mg': 4, 'Al': 8, 'O': 16}, -651.279),
                     System(5, {'Mg': 4, 'Al': 8, 'O': 16}, -643.925),
                     System(6, {'Mg': 4, 'Al': 8, 'O': 16}, np.inf),
                     System(7, {'Mg': 4, 'Al': 8, 'O': 16}, -652.042),
                     System(8, {'Mg': 4, 'Al': 8, 'O': 16}, -648.368),
                     System(9, {'Mg': 4, 'Al': 8, 'O': 16}, -648.335)]
        self.fitness = Fitness(self.pool)

    def test_enthalpy(self):
        print(self.fitness.calcFitness('enthalpy'))

    def test_composition(self):
        print(self.fitness.calcFitness('composition'))

    def test_tabulate(self):
        print(self.fitness.calcFitness(('tabulate', 'composition')))

    def test_getRelativeCHSpace(self):
        print(self.fitness.calcFitness(('getRelativeCHSpace', ('tabulate', 'composition'), 'enthalpy')))

    def test_convexHullHeight(self):
        print(self.fitness.calcFitness(('convexHullHeight',
                                                    ('getRelativeCHSpace', ('tabulate', 'composition'), 'enthalpy'))))

    def test_pareto(self):
        print(self.fitness.calcFitness(('pareto', ('convexHullHeight',
                                                ('getRelativeCHSpace', ('tabulate', 'composition'), 'enthalpy')))))
