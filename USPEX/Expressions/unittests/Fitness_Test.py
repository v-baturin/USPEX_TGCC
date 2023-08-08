"""
USPEX.Common.Atomistic.unittests.FitnessHull_Test
================================================

Class for Fitness testing

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import unittest
import numpy as np
import os
from os.path import join as pj


from ..ExpressionEvaluator import ExpressionEvaluator
from USPEX.Expressions.Functions.BasicFunctions import BasicFunctions
from ...Optimizers.PoolEntry import PoolEntry, EntryFlavour
from ...components import CompositionSpace, SimpleMoleculeUtility, AtomisticRepresentation, Atomistic
from USPEX.Atomistic.Primitives.AtomicPrimitives import AtomicStructure
from ...Atomistic.RadialDistributionUtility import Fingerprint
from ...XRay.PowderSpectrumAnalyzer import PowderSpectrumAnalyzer


HOMEPATH = os.path.dirname(os.path.abspath(__file__))


class System(object):
    def __init__(self, composition: dict):
        self.composition = composition



class Fitness_Test(unittest.TestCase):
    def setUp(self) -> None:
        molecules = [AtomicStructure([symbol], np.zeros((1, 3), dtype=float), np.eye(3, dtype=float))
                     for symbol in ['Mg'] * 4 + ['Al'] * 8 + ['O'] * 16]
        self.systems = [{'ID': 0, 'atomistic.molecules': molecules, '.enthalpy': -646.695,
                         '.fingerprint': Fingerprint({'a':[0.2,-0.2], 'b': [0.2,-0.2]}, None, None)},
                        {'ID': 1, 'atomistic.molecules': molecules, '.enthalpy': -644.480,
                         '.fingerprint': Fingerprint({'a':[0.2,-0.2]}, None, None)},
                        {'ID': 2, 'atomistic.molecules': molecules, '.enthalpy': -650.098,
                         '.fingerprint': Fingerprint({'a':[0.3,-0.3], 'b': [0.4,-0.4]}, None, None)},
                        {'ID': 3, 'atomistic.molecules': molecules, '.enthalpy': -649.082,
                         '.fingerprint': Fingerprint({'b': [0.1,-0.5]}, None, None)},
                        {'ID': 4, 'atomistic.molecules': molecules, '.enthalpy': -651.279,
                         '.fingerprint': Fingerprint({'a':[0.3,-0.3], 'b': [0.4,-0.4]}, None, None)},
                        {'ID': 5, 'atomistic.molecules': molecules, '.enthalpy': -643.925,
                         '.fingerprint': Fingerprint({'a':[-0.3,-0.2], 'b': [0.7,-0.2]}, None, None)},
                        {'ID': 6, 'atomistic.molecules': molecules, '.enthalpy': -652.042,
                         '.fingerprint': Fingerprint({'b': [0.1,-0.2]}, None, None)},
                        {'ID': 7, 'atomistic.molecules': molecules, '.enthalpy': -648.368,
                         '.fingerprint': Fingerprint({'a':[0.2,-0.2], 'b': [0.2,-0.2]}, None, None)},
                        {'ID': 8, 'atomistic.molecules': molecules, '.enthalpy': -648.335,
                         '.fingerprint': Fingerprint({'a':[0.2,-0.2], 'b': [0.2,-0.2]}, None, None)}]
        self.compositionSpace = CompositionSpace(symbols=['Mg', 'Al', 'O'], blocks=[[4, 8, 16]], range=[[1, 1]])
        self.simpleMoleculeUtility = SimpleMoleculeUtility()
        expressionExtensions = dict(
            basic=BasicFunctions(),
            compositionSpace=self.compositionSpace.expressionExtension(self.compositionSpace),
        )
        propertyExtensions = dict(
            simpleMoleculeUtility=self.simpleMoleculeUtility.propertyExtension(self.simpleMoleculeUtility)
        )
        self.systems = [PoolEntry(system['ID'], EntryFlavour(extensions=propertyExtensions, **system)) for system in self.systems]

        self.fitness = ExpressionEvaluator(tuple(self.systems), expressionExtensions)

    def test_enthalpy(self):
        ref = [-646.695, -644.48,  -650.098, -649.082, -651.279, -643.925, -652.042, -648.368, -648.335]
        self.assertTrue(np.allclose(self.fitness.evaluate('.enthalpy.origin'), ref))

    def test_composition(self):
        ref = [{'Mg': 4, 'Al': 8, 'O': 16},
               {'Mg': 4, 'Al': 8, 'O': 16},
               {'Mg': 4, 'Al': 8, 'O': 16},
               {'Mg': 4, 'Al': 8, 'O': 16},
               {'Mg': 4, 'Al': 8, 'O': 16},
               {'Mg': 4, 'Al': 8, 'O': 16},
               {'Mg': 4, 'Al': 8, 'O': 16},
               {'Mg': 4, 'Al': 8, 'O': 16},
               {'Mg': 4, 'Al': 8, 'O': 16}]
        self.assertTrue(np.all([value == ref_value for value, ref_value in zip(self.fitness.evaluate('simpleMoleculeUtility.composition.origin'), ref)]))

    def test_compositionSpace_numIons(self):
        ref = [[4, 8, 16, ],
               [4, 8, 16, ],
               [4, 8, 16, ],
               [4, 8, 16, ],
               [4, 8, 16, ],
               [4, 8, 16, ],
               [4, 8, 16, ],
               [4, 8, 16, ],
               [4, 8, 16, ]]
        self.assertTrue(np.allclose(self.fitness.evaluate(('compositionSpace.numMolsFromCompositions',
                                                              'simpleMoleculeUtility.composition.origin')), ref))

    def test_compositionSpace_numBlocks(self):
        ref = [[1], [1], [1], [1], [1], [1], [1], [1], [1]]
        self.assertTrue(np.allclose(self.fitness.evaluate(('compositionSpace.numBlocksFromCompositions',
                                                              'simpleMoleculeUtility.composition.origin')), ref))

    def test_getRelativeCHSpace(self):
        ref = [[-646.695], [-644.48 ], [-650.098], [-649.082], [-651.279], [-643.925], [-652.042], [-648.368], [-648.335]]
        self.assertTrue(np.allclose(self.fitness.evaluate(('getRelativeCHSpace',
                                                           ('compositionSpace.numBlocksFromCompositions',
                                                              'simpleMoleculeUtility.composition.origin'), '.enthalpy.origin')), ref))

    def test_convexHullHeightComposition(self):
        ref = [5.347, 7.562, 1.944, 2.96,  0.763, 8.117, 0., 3.674, 3.707]
        self.assertTrue(np.allclose(self.fitness.evaluate(('convexHullHeight',
                                                           ('getRelativeCHSpace',
                                                               ('compositionSpace.numBlocksFromCompositions',
                                                              'simpleMoleculeUtility.composition.origin'), '.enthalpy.origin'))), ref))

    def test_simpleHeightComposition(self):
        ref = [5.347, 7.562, 1.944, 2.96, 0.763, 8.117, 0., 3.674, 3.707]
        self.assertTrue(np.allclose(self.fitness.evaluate(('simpleHeight',
                                                           ('getRelativeCHSpace',
                                                               ('compositionSpace.numBlocksFromCompositions',
                                                              'simpleMoleculeUtility.composition.origin'), '.enthalpy.origin'))), ref))

    def test_pareto(self):
        ref = [6, 7, 2, 3, 1, 8, 0, 4, 5]
        self.assertTrue(np.allclose(self.fitness.evaluate(('pareto', ('convexHullHeight',
                                                                      ('getRelativeCHSpace',
                                                                ('compositionSpace.numBlocksFromCompositions',
                                                              'simpleMoleculeUtility.composition.origin'), '.enthalpy.origin')))), ref))

    def test_fingerprint(self):
        ref = [{'a': [0.2, -0.2], 'b': [0.2, -0.2]},
               {'a': [0.2, -0.2]},
               {'a': [0.3, -0.3], 'b': [0.4, -0.4]},
               {'b': [0.1, -0.5]},
               {'a': [0.3, -0.3], 'b': [0.4, -0.4]},
               {'a': [-0.3, -0.2], 'b': [0.7, -0.2]},
               {'b': [0.1, -0.2]},
               {'a': [0.2, -0.2], 'b': [0.2, -0.2]},
               {'a': [0.2, -0.2], 'b': [0.2, -0.2]}]
        self.assertTrue(np.all([value == ref_value for value, ref_value in zip(self.fitness.evaluate('.fingerprint.origin'), ref)]))

    def test_tabulateFingerprint(self):
        ref = [[[ 0.2, -0.2], [ 0.2, -0.2]],
               [[ 0.2, -0.2], [-1.,  -1. ]],
               [[ 0.3, -0.3], [ 0.4, -0.4]],
               [[-1.,  -1. ], [ 0.1, -0.5]],
               [[ 0.3, -0.3], [ 0.4, -0.4]],
               [[-0.3, -0.2], [ 0.7, -0.2]],
               [[-1.,  -1. ], [ 0.1, -0.2]],
               [[ 0.2, -0.2], [ 0.2, -0.2]],
               [[ 0.2, -0.2], [ 0.2, -0.2]]]
        self.assertTrue(np.allclose(self.fitness.evaluate(('tabulate', '.fingerprint.origin')), ref))

    def test_hstack(self):
        ref = [[0.2, -0.2, 0.2, -0.2],
               [0.2, -0.2, -1.,  -1.],
               [0.3, -0.3, 0.4, -0.4],
               [-1.,  -1., 0.1, -0.5],
               [0.3, -0.3, 0.4, -0.4],
               [-0.3, -0.2, 0.7, -0.2],
               [-1.,  -1., 0.1, -0.2],
               [0.2, -0.2, 0.2, -0.2],
               [0.2, -0.2, 0.2, -0.2]]
        self.assertTrue(np.allclose(self.fitness.evaluate(('hstack', ('tabulate', '.fingerprint.origin'))), ref))

    def test_getPrincipalComponents(self):
        ref = [[-0.33720674, -0.17273979],
               [-0.53465894,  1.24425809],
               [-0.36473363, -0.25874203],
               [ 1.05472675,  0.24687327],
               [-0.36473363, -0.25874203],
               [ 0.14210271, -0.56868036],
               [ 1.07891698,  0.11325245],
               [-0.33720674, -0.17273979],
               [-0.33720674, -0.17273979]]
        self.assertTrue(np.allclose(self.fitness.evaluate(('getPrincipalComponents', 2,
                                                           ('hstack', ('tabulate', '.fingerprint.origin')))), ref))

    def test_getAbsoluteCHSpace(self):
        ref = [[-3.37206745e-01, -1.72739794e-01, -6.46695000e+02],
               [-5.34658944e-01,  1.24425809e+00, -6.44480000e+02],
               [-3.64733632e-01, -2.58742032e-01, -6.50098000e+02],
               [ 1.05472675e+00,  2.46873266e-01, -6.49082000e+02],
               [-3.64733632e-01, -2.58742032e-01, -6.51279000e+02],
               [ 1.42102712e-01, -5.68680363e-01, -6.43925000e+02],
               [ 1.07891698e+00,  1.13252449e-01, -6.52042000e+02],
               [-3.37206745e-01, -1.72739794e-01, -6.48368000e+02],
               [-3.37206745e-01, -1.72739794e-01, -6.48335000e+02]]
        self.assertTrue(np.allclose(self.fitness.evaluate(('getAbsoluteCHSpace',
                                                           ('getPrincipalComponents', 2,
                                                               ('hstack', ('tabulate', '.fingerprint.origin'))),
                                                              '.enthalpy.origin')), ref))

    def test_convexHullHeightFingerprint(self):
        ref = [4.256279, 0., 1.181, 0., 0., 0., 0., 2.583279, 2.616279]
        self.assertTrue(np.allclose(self.fitness.evaluate(('convexHullHeight',
                                                           ('getAbsoluteCHSpace',
                                                               ('getPrincipalComponents', 2,
                                                                ('hstack', ('tabulate', '.fingerprint.origin'))),
                                                               '.enthalpy.origin'))), ref))


class FitnessXray_Test(unittest.TestCase):
    def setUp(self) -> None:
        # 'externalPressure': 135,
        filename = pj(HOMEPATH,'XRay_POSCARS')
        self.systems = AtomisticRepresentation.readAtomicStructures(filename)
        enthalpies = [0.001, 0.103, 0.000, 0.033, 0.130, 0.037, 12.011, 0.054, 0.044, 0.228]

        self.compositionSpace = CompositionSpace(symbols=['Ba', 'H'], blocks=[[1, 12]], range=[[4, 4]])
        self.powderSpectrumAnalyzer = PowderSpectrumAnalyzer(**PowderSpectrumAnalyzer.parse(pj(HOMEPATH, 'spectrum.txt')))
        self.simpleMoleculeUtility = SimpleMoleculeUtility()
        self.atomistic = Atomistic()

        expressionExtensions = dict(
            basic=BasicFunctions(),
            compositionSpace=self.compositionSpace.expressionExtension(self.compositionSpace),
        )
        propertyExtensions = dict(

            atomistic = Atomistic.propertyExtension(self.atomistic),
            simpleMoleculeUtility=self.simpleMoleculeUtility.propertyExtension(self.simpleMoleculeUtility),
            powderSpectrumAnalyzer=self.powderSpectrumAnalyzer.propertyExtension(self.powderSpectrumAnalyzer),
        )
        for ID, system in enumerate(self.systems):
            system['.enthalpy'] = enthalpies[ID]
            self.systems[ID] = PoolEntry(ID, EntryFlavour(extensions=propertyExtensions, **system))
            self.systems[ID].getProperty('structure', extension='atomistic')

        self.fitness = ExpressionEvaluator(tuple(self.systems), expressionExtensions)

    def test_xraydistance(self):
        ref = [0.190, 0.028,  0.192, 0.165, 0.028, 0.104, 0.028, 0.122, 0.132, 0.042]
        self.assertTrue(np.allclose(np.round(self.fitness.evaluate('powderSpectrumAnalyzer.xraydistance.origin'),
                                             decimals=3), ref))

    def test_k(self):
        ref = [1.003, 1.005,  1.003, 1.005, 1.006, 1.011, 0.994, 1.004, 1.005, 1.004]
        self.assertTrue(np.allclose(np.round(self.fitness.evaluate('powderSpectrumAnalyzer.k.origin'), decimals=3), ref))

    def test_pareto(self):
        ref = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
        self.assertTrue(np.allclose(self.fitness.evaluate(('pareto', '.enthalpy.origin',
                                                              'powderSpectrumAnalyzer.xraydistance.origin')), ref))
