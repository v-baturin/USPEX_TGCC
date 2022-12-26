import unittest
import os

from ..compileParams import compileParams, read_molecule, PowderSpectrumAnalyzer


TESTPATH = os.path.dirname(os.path.abspath(__file__))
PATH_BACKUP = os.getcwd()


class CompileParams_Test(unittest.TestCase):
    def setUp(self) -> None:
        os.chdir(TESTPATH)

    def tearDown(self) -> None:
        os.chdir(PATH_BACKUP)

    def test_stages(self):
        definitions = {
            'optimizer': {
                'type': 'GlobalOptimizer',
                'target': {
                    'type': 'Atomistic',
                    'conditions': {'externalPressure': 100},
                    'compositionSpace': {'symbols': ['Mg', 'Al', 'O'],
                                         'blocks': [[4, 8, 16]]},
                },
                'optType': 'enthalpy',
                'selection': {
                    'type': 'USPEXClassic',
                    'popSize': 40,
                    'bestFrac': 0.6,
                    'optType': ('aging', 'enthalpy'),
                    'fractions': {
                        'heredity': (0.1, 1.0, 0.5),
                        'softmodemutation': (0.1, 1.0, 0.2),
                        'randSym': (0.05, 1.0, 0.1),
                        'randTop': (0.05, 1.0, 0.1),
                        'permutation': (0.05, 1.0, 0.1)
                    }
                }
            },
            'stages': [
                {'name': 'glp', 'type': 'gulp', 'commandExecutable': 'gulp'},
                {'name': 'glp', 'type': 'gulp', 'commandExecutable': 'gulp'},
                {'name': 'glp', 'type': 'gulp', 'commandExecutable': 'gulp'},
                {'name': 'glp', 'type': 'gulp', 'commandExecutable': 'gulp'}],
            'numParallelCalcs': 20,
            'numGenerations': 60,
            'stopCrit': 30
        }
        params_ref = {
            'optimizer': {
                'type': 'GlobalOptimizer',
                'target': {
                    'type': 'Atomistic',
                    'conditions': {'externalPressure': 100},
                    'compositionSpace': {'symbols': ['Mg', 'Al', 'O'],
                                         'blocks': [[4, 8, 16]]},
                    'radialDistributionUtility': {'symbols': ['Al', 'Mg', 'O']},
                    'bondUtility': {'volumeType': 0}
                },
                'fingerprintUtility': 'radialDistributionUtility',
                'optType': 'enthalpy',
                'selection': {
                    'type': 'USPEXClassic',
                    'popSize': 40,
                    'bestFrac': 0.6,
                    'optType': ('aging', 'enthalpy'),
                    'fractions': {
                        'heredity': (0.1, 1.0, 0.5),
                        'softmodemutation': (0.1, 1.0, 0.2),
                        'randSym': (0.05, 1.0, 0.1),
                        'randTop': (0.05, 1.0, 0.1),
                        'permutation': (0.05, 1.0, 0.1)
                    }
                }
            },
            'stages': [
                {'name': 'glp', 'stageType': 'atomistic', 'type': 'gulp', 'commandExecutable': 'gulp', 'tag': '1'},
                {'name': 'glp', 'stageType': 'atomistic', 'type': 'gulp', 'commandExecutable': 'gulp', 'tag': '2'},
                {'name': 'glp', 'stageType': 'atomistic', 'type': 'gulp', 'commandExecutable': 'gulp', 'tag': '3'},
                {'name': 'glp', 'stageType': 'atomistic', 'type': 'gulp', 'commandExecutable': 'gulp', 'tag': '4'}],
            'numParallelCalcs': 20,
            'numGenerations': 60,
            'stopCrit': 30
        }
        params = compileParams(definitions)
        self.assertEqual(params, params_ref)

    def test_molecules(self):
        definitions = {
            'optimizer': {
                'type': 'GlobalOptimizer',
                'target': {
                    'type': 'Atomistic',
                    'conditions': {'externalPressure': 20.0},
                    'compositionSpace': {'symbols': [{'name': 'mol_h2o', 'filename': 'MOL_H2O'}],
                                         'blocks': [[4]]},
                },
                'optType': 'enthalpy',
                'selection': {'type': 'USPEXClassic',
                              'optType': ('aging', 'enthalpy'),
                              'popSize': 20,
                              'fractions': {'heredity': (0.1, 1.0, 0.5),
                                             'softmodemutation': (0.1, 1.0, 0.3),
                                             'randTop': (0.1, 1.0, 0.2)}}
            },
            'stages': [],
            'numParallelCalcs': 20,
            'numGenerations': 30,
            'stopCrit': 6,
        }
        params_ref = {
            'optimizer': {
                'type': 'GlobalOptimizer',
                'target': {
                    'type': 'Atomistic',
                    'conditions': {'externalPressure': 20.0},
                    'simpleMoleculeUtility': {'molecules': {'mol_h2o': {'symbols': ['H', 'O', 'H'],
                                                                        'labels': ['', '', ''],
                                                                        'positions': [[0.0, -0.1988, -0.7632],
                                                                                      [0.0, 0.3975, 0.0],
                                                                                      [0.0, -0.1988, 0.7632]],
                                                                        'configZMatrix': [[0, 0, 0],[1, 0, 0],[2, 1, 0]],
                                                                        'flexDihedrals': []}}},
                    'compositionSpace': {'symbols': ['mol_h2o'],
                                         'blocks': [[4]]},
                    'radialDistributionUtility': {'symbols': ['H', 'O']},
                    'bondUtility': {'volumeType': 0.5}
                },
                'fingerprintUtility': 'radialDistributionUtility',
                'optType': 'enthalpy',
                'selection': {'type': 'USPEXClassic',
                              'optType': ('aging', 'enthalpy'),
                              'popSize': 20,
                              'fractions': {'heredity': (0.1, 1.0, 0.5),
                                             'softmodemutation': (0.1, 1.0, 0.3),
                                             'randTop': (0.1, 1.0, 0.2)}}
            },
            'stages': [],
            'numParallelCalcs': 20,
            'numGenerations': 30,
            'stopCrit': 6,
        }
        params = compileParams(definitions)
        self.assertEqual(params, params_ref)

    def test_XRay(self):
        definitions = {
                'optimizer': {
                    'type': 'GlobalOptimizer',
                    'target': {
                        'type': 'Atomistic',
                        'conditions': {'externalPressure': 100},
                        'powderSpectrumAnalyzer': 'spectrum.txt',
                        'compositionSpace': {'symbols': ['Na', 'Cl'],
                                             'blocks': [[8,24]],
                                             'range': [[1,1]]}
                    },
                    'optType': 'enthalpy',
                    'selection': {}
                },
                'stages': [],
                'numParallelCalcs': 2,
                'numGenerations': 3,
                'stopCrit': 3
        }
        params_ref = {
            'optimizer': {
                'type': 'GlobalOptimizer',
                'target': {
                    'type': 'Atomistic',
                    'conditions': {'externalPressure': 100},
                    'powderSpectrumAnalyzer': PowderSpectrumAnalyzer.parse('spectrum.txt'),
                    'compositionSpace': {'symbols': ['Na', 'Cl'],
                                         'blocks': [[8,24]],
                                         'range': [[1,1]]},
                    'radialDistributionUtility': {'symbols': ['Cl', 'Na']},
                    'bondUtility': {'volumeType': 0}
                },
                'fingerprintUtility': 'radialDistributionUtility',
                'optType': 'enthalpy',
                'selection': {'optType': 'enthalpy'}
            },
            'stages': [],
            'numParallelCalcs': 2,
            'numGenerations': 3,
            'stopCrit': 3,
        }
        params = compileParams(definitions)
        self.assertEqual(params, params_ref)
