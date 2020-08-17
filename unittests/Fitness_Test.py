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
from ..Atomistic.CompositionSpace import CompositionSpace
from ..Atomistic.Fingerprints.fingerprint import Fingerprint


class System(object):
    def __init__(self, ID: int, composition: dict, enthalpy: float, fingerprint: Fingerprint):
        self.ID = ID
        self.composition = composition
        self.enthalpy = enthalpy
        self.fingerprint = fingerprint


class Fitness_Test(unittest.TestCase):
    def setUp(self) -> None:
        self.pool = [System(0, {'Mg': 4, 'Al': 8, 'O': 16}, -646.695, Fingerprint({'a':[0.2,-0.2], 'b': [0.2,-0.2]}, None)),
                     System(1, {'Mg': 4, 'Al': 8, 'O': 16}, -644.480, Fingerprint({'a':[0.2,-0.2]}, None)),
                     System(2, {'Mg': 4, 'Al': 8, 'O': 16}, -650.098, Fingerprint({'a':[0.3,-0.3], 'b': [0.4,-0.4]}, None)),
                     System(3, {'Mg': 4, 'Al': 8, 'O': 16}, -649.082, Fingerprint({'b': [0.1,-0.5]}, None)),
                     System(4, {'Mg': 4, 'Al': 8, 'O': 16}, -651.279, Fingerprint({'a':[0.3,-0.3], 'b': [0.4,-0.4]}, None)),
                     System(5, {'Mg': 4, 'Al': 8, 'O': 16}, -643.925, Fingerprint({'a':[-0.3,-0.2], 'b': [0.7,-0.2]}, None)),
                     System(6, {'Mg': 4, 'Al': 8, 'O': 16}, -652.042, Fingerprint({'b': [0.1,-0.2]}, None)),
                     System(7, {'Mg': 4, 'Al': 8, 'O': 16}, -648.368, Fingerprint({'a':[0.2,-0.2], 'b': [0.2,-0.2]}, None)),
                     System(8, {'Mg': 4, 'Al': 8, 'O': 16}, -648.335, Fingerprint({'a':[0.2,-0.2], 'b': [0.2,-0.2]}, None))]
        self.compositionSpace = CompositionSpace(symbols=['Mg','Al','O'], blocks=[[4,8,16]], range=[[1,1]])
        self.fitness = Fitness(self.pool, {'compositionSpace': self.compositionSpace})

    def test_enthalpy(self):
        print(self.fitness.calcFitness('enthalpy'))

    def test_composition(self):
        print(self.fitness.calcFitness('composition'))

    def test_tabulateComposition(self):
        print(self.fitness.calcFitness(('tabulate', 'composition')))

    def test_compositionBlocks(self):
        print(self.fitness.calcFitness(('compositionBlocks', ('tabulate', 'composition'))))

    def test_getRelativeCHSpace(self):
        print(self.fitness.calcFitness(('getRelativeCHSpace',
                                        ('compositionBlocks', ('tabulate', 'composition')), 'enthalpy')))

    def test_convexHullHeightComposition(self):
        print(self.fitness.calcFitness(('convexHullHeight',
                                                    ('getRelativeCHSpace',
                                                     ('compositionBlocks', ('tabulate', 'composition')), 'enthalpy'))))

    def test_pareto(self):
        print(self.fitness.calcFitness(('pareto', ('convexHullHeight',
                                                ('getRelativeCHSpace', ('compositionBlocks',
                                                                        ('tabulate', 'composition')), 'enthalpy')))))

    def test_fingerprint(self):
        print(self.fitness.calcFitness('fingerprint'))

    def test_tabulateFingerprint(self):
        print(self.fitness.calcFitness(('tabulate', 'fingerprint')))

    def test_vstack(self):
        print(self.fitness.calcFitness(('vstack', ('tabulate', 'fingerprint'))))

    def test_getPrincipalComponents(self):
        print(self.fitness.calcFitness(('getPrincipalComponents', 2, ('vstack', ('tabulate', 'fingerprint')))))

    def test_getAbsoluteCHSpace(self):
        print(self.fitness.calcFitness(('getAbsoluteCHSpace',
                                        ('getPrincipalComponents', 2, ('vstack', ('tabulate', 'fingerprint'))),
                                        'enthalpy'
                                        )))

    def test_convexHullHeightFingerprint(self):
        print(self.fitness.calcFitness(('convexHullHeight',
                                        ('getAbsoluteCHSpace',
                                         ('getPrincipalComponents', 2, ('vstack', ('tabulate', 'fingerprint'))),
                                         'enthalpy'
                                         ))))
