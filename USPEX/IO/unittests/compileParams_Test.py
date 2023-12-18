import unittest
import os

from pathlib import Path

from ...components import compileParams, PowderSpectrumAnalyzer


TESTPATH = Path(__file__).parent
PATH_BACKUP = Path.cwd()


class CompileParams_Test(unittest.TestCase):
    def setUp(self) -> None:
        os.chdir(TESTPATH)

    def tearDown(self) -> None:
        os.chdir(PATH_BACKUP)

    def test_stages(self):
        definitions = {
            'generator': {
                'type': 'Evolution',
                'target': {
                    'type': 'Atomistic',
                    'conditions': {'externalPressure': 100},
                    'compositionSpace': {'symbols': ['Mg', 'Al', 'O'],
                                         'blocks': [[4, 8, 16]]},
                },
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
            },
            'optimizer': {
                'type': 'GlobalOptimizer',
                'optType': 'enthalpy',
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
            'generator': {
                'type': 'Evolution',
                'target': {
                    'type': 'Atomistic',
                    'conditions': {'externalPressure': 100},
                    'compositionSpace': {'symbols': ['Mg', 'Al', 'O'],
                                         'blocks': [[4, 8, 16]]},
                    'radialDistributionUtility': {'symbols': ['Al', 'Mg', 'O'], 'suffix': '4'},
                    'defaultSuffix': '4',
                    'bondUtility': {'volumeType': 0, 'cutoff': 'strong'},
                    'junctionUtility': {'molSitesMapping': {}}
                },
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
            },
            'optimizer': {
                'type': 'GlobalOptimizer',
                'optType': 'enthalpy',
                'goodSystemsSuffixes': {'enthalpy'},
            },
            'stages': [
                {'name': 'glp', 'stageType': 'atomistic', 'type': 'gulp', 'commandExecutable': 'gulp', 'tag': '1', 'source': 'origin'},
                {'name': 'glp', 'stageType': 'atomistic', 'type': 'gulp', 'commandExecutable': 'gulp', 'tag': '2', 'source': '1'},
                {'name': 'glp', 'stageType': 'atomistic', 'type': 'gulp', 'commandExecutable': 'gulp', 'tag': '3', 'source': '2'},
                {'name': 'glp', 'stageType': 'atomistic', 'type': 'gulp', 'commandExecutable': 'gulp', 'tag': '4', 'source': '3'}],
            'numParallelCalcs': 20,
            'numGenerations': 60,
            'stopCrit': 30,
            'output': {
                'stages': ['1', '2', '3', '4'],
                'suffix': '4'
            }

        }
        params = compileParams(definitions)
        self.assertEqual(params, params_ref)

    # def test_molecules(self):
    #     definitions = {
    #         'optimizer': {
    #             'type': 'GlobalOptimizer',
    #             'target': {
    #                 'type': 'Atomistic',
    #                 'conditions': {'externalPressure': 20.0},
    #                 'compositionSpace': {'symbols': [{'name': 'mol_h2o', 'filename': 'MOL_H2O'}],
    #                                      'blocks': [[4]]},
    #             },
    #             'optType': 'enthalpy',
    #             'selection': {'type': 'USPEXClassic',
    #                           'optType': ('aging', 'enthalpy'),
    #                           'popSize': 20,
    #                           'fractions': {'heredity': (0.1, 1.0, 0.5),
    #                                          'softmodemutation': (0.1, 1.0, 0.3),
    #                                          'randTop': (0.1, 1.0, 0.2)}}
    #         },
    #         'stages': [],
    #         'numParallelCalcs': 20,
    #         'numGenerations': 30,
    #         'stopCrit': 6,
    #     }
    #     params_ref = {
    #         'optimizer': {
    #             'type': 'GlobalOptimizer',
    #             'target': {
    #                 'type': 'Atomistic',
    #                 'conditions': {'externalPressure': 20.0},
    #                 'simpleMoleculeUtility': {'molecules': {'mol_h2o': {'symbols': ['H', 'O', 'H'],
    #                                                                     'labels': ['', '', ''],
    #                                                                     'positions': [[0.0, -0.1988, -0.7632],
    #                                                                                   [0.0, 0.3975, 0.0],
    #                                                                                   [0.0, -0.1988, 0.7632]],
    #                                                                     'configZMatrix': [[0, 0, 0],[1, 0, 0],[2, 1, 0]],
    #                                                                     'flexDihedrals': []}}},
    #                 'compositionSpace': {'symbols': ['mol_h2o'],
    #                                      'blocks': [[4]]},
    #                 'radialDistributionUtility': {'symbols': ['H', 'O']},
    #                 'bondUtility': {'volumeType': 0.5, 'cutoff': 'strong'},
    #                 'junctionUtility': {'molSitesMapping': {}}
    #             },
    #             'optType': 'enthalpy',
    #             'selection': {'type': 'USPEXClassic',
    #                           'optType': ('aging', 'enthalpy'),
    #                           'popSize': 20,
    #                           'fractions': {'heredity': (0.1, 1.0, 0.5),
    #                                          'softmodemutation': (0.1, 1.0, 0.3),
    #                                          'randTop': (0.1, 1.0, 0.2)}}
    #         },
    #         'stages': [],
    #         'numParallelCalcs': 20,
    #         'numGenerations': 30,
    #         'stopCrit': 6,
    #     }
    #     params = compileParams(definitions)
    #     self.assertEqual(params, params_ref)

    def test_XRay(self):
        definitions = {
            'generator': {
                'target': {
                    'type': 'Atomistic',
                    'conditions': {'externalPressure': 100},
                    'powderSpectrumAnalyzer': 'spectrum.txt',
                    'compositionSpace': {'symbols': ['Na', 'Cl'],
                                         'blocks': [[8, 24]],
                                         'range': [[1, 1]]},
                },
            },
            'optimizer': {
                    'type': 'GlobalOptimizer',
                    'optType': 'enthalpy',
                },
                'stages': [],
                'numParallelCalcs': 2,
                'numGenerations': 3,
                'stopCrit': 3
        }
        params_ref = {
            'generator': {
                'target': {
                    'type': 'Atomistic',
                    'conditions': {'externalPressure': 100},
                    'powderSpectrumAnalyzer': PowderSpectrumAnalyzer.parse('spectrum.txt'),
                    'compositionSpace': {'symbols': ['Na', 'Cl'],
                                         'blocks': [[8, 24]],
                                         'range': [[1, 1]]},
                    'radialDistributionUtility': {'symbols': ['Cl', 'Na'], 'suffix': 'origin'},
                    'defaultSuffix': 'origin',
                    'bondUtility': {'volumeType': 0, 'cutoff': 'strong'},
                    'junctionUtility': {'molSitesMapping': {}}
                },
                'optType': 'enthalpy'
            },
            'optimizer': {
                'type': 'GlobalOptimizer',
                'optType': 'enthalpy',
                'goodSystemsSuffixes': {'enthalpy'},
            },
            'stages': [],
            'numParallelCalcs': 2,
            'numGenerations': 3,
            'stopCrit': 3,
            'output': {
                'stages': [],
                'suffix': 'origin'
            }
        }
        params = compileParams(definitions)
        self.assertEqual(params, params_ref)
