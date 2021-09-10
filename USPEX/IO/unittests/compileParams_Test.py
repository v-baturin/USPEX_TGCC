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
            'main':{
                'optimizer': {
                    'type': 'GlobalOptimizer',
                    'target': {
                        'type': 'Crystal',
                        'conditions': {'externalPressure': 100},
                        'compositionSpace': {'symbols': ['Mg', 'Al', 'O'],
                                             'blocks': [[4, 8, 16]]}
                    },
                    'optType': 'enthalpy',
                    'stopFitness': -655.062,
                    'selection': {
                        'type': 'USPEXClassic',
                        'popSize': 10,
                        'optType': ('aging', 'enthalpy'),
                        'fractions': {
                            'heredity': [0.0, 1.0, 0.5],
                            'twinning': [0.0, 1.0, 0.1],
                            'softmodemutation': [0.0, 1.0, 0.1],
                            'randSym': [0.0, 1.0, 0.0],
                            'randTop': [0.0, 1.0, 0.2],
                            'permutation': [0.0, 1.0, 0.1]
                        }
                    }
                },
                'stages': ['gulp', 'gulp', 'gulp', 'gulp', 'gulp5'],
                'numParallelCalcs': 2,
                'numGenerations': 3,
                'stopCrit': 3
            },
        'gulp': {'type' : 'gulp', 'commandExecutable' : 'gulp', 'goptions' : './Specific/goptions'},
        'gulp5': {'type' : 'gulp', 'commandExecutable' : 'gulp', 'goptions' : './Specific/goptions',
                  'ginput' : './Specific/ginput_4'}

        }
        params_ref = {
            'optimizer': {
                'type': 'GlobalOptimizer',
                'target': {
                    'type': 'Crystal',
                    'conditions': {'externalPressure': 100, 'volumeType': 0},
                    'compositionSpace': {'symbols': ['Mg', 'Al', 'O'],
                                         'blocks': [[4, 8, 16]]},
                    'cellUtility': {'pbc': (1, 1, 1)}
                },
                'fingerprintUtility': 'radialDistributionUtility',
                'optType': 'enthalpy',
                'stopFitness': -655.062,
                'selection': {
                    'type': 'USPEXClassic',
                    'popSize': 10,
                    'optType': ('aging', 'enthalpy'),
                    'fractions': {
                        'heredity': [0.0, 1.0, 0.5],
                        'twinning': [0.0, 1.0, 0.1],
                        'softmodemutation': [0.0, 1.0, 0.1],
                        'randSym': [0.0, 1.0, 0.0],
                        'randTop': [0.0, 1.0, 0.2],
                        'permutation': [0.0, 1.0, 0.1]
                    }
                }
            },
            'stages': [
                {'type' : 'gulp', 'commandExecutable' : 'gulp', 'goptions' : './Specific/goptions', 'tag': '1'},
                {'type' : 'gulp', 'commandExecutable' : 'gulp', 'goptions' : './Specific/goptions', 'tag': '2'},
                {'type' : 'gulp', 'commandExecutable' : 'gulp', 'goptions' : './Specific/goptions', 'tag': '3'},
                {'type' : 'gulp', 'commandExecutable' : 'gulp', 'goptions' : './Specific/goptions', 'tag': '4'},
                {'type': 'gulp', 'commandExecutable': 'gulp', 'goptions': './Specific/goptions',
                 'ginput': './Specific/ginput_4', 'tag': '5'}],
            'numParallelCalcs': 2,
            'numGenerations': 3,
            'stopCrit': 3
        }
        params = compileParams(**definitions)
        self.assertEqual(params, params_ref)

    def test_molecules(self):
        definitions = {
            'main':{
                'optimizer': {
                    'type': 'GlobalOptimizer',
                    'target': {
                        'type': 'Crystal',
                        'conditions': {'externalPressure': 100},
                        'compositionSpace': {'symbols': ['mol_h2o'],
                                             'blocks': [[4]],
                                             'range': [[1,1]]}
                    },
                    'optType': 'enthalpy',
                    'selection': {},
                },
                'stages': [],
                'numParallelCalcs': 2,
                'numGenerations': 3,
                'stopCrit': 3
            },
        'mol_h2o': {'filename': 'MOL_H2O'}

        }
        params_ref = {
            'optimizer': {
                'type': 'GlobalOptimizer',
                'target': {
                    'type': 'Crystal',
                    'conditions': {'externalPressure': 100, 'volumeType': 0.5},
                    'simpleMoleculeUtility': {'molecules': {'mol_h2o': {'symbols': ['H', 'O', 'H'],
                                                                        'labels': ['', '', ''],
                                                                        'positions': [[0.0, -0.1988, -0.7632],
                                                                                      [0.0, 0.3975, 0.0],
                                                                                      [0.0, -0.1988, 0.7632]],
                                                                        'configZMatrix': [[0, 0, 0],[1, 0, 0],[2, 1, 0]],
                                                                        'flexDihedrals': []}}},
                    'compositionSpace': {'symbols': ['mol_h2o'],
                                         'blocks': [[4]],
                                         'range': [[1,1]]},
                    'cellUtility': {'pbc': (1, 1, 1)}
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
        params = compileParams(**definitions)
        self.assertEqual(params, params_ref)

    def test_XRay(self):
        definitions = {
            'main':{
                'optimizer': {
                    'type': 'GlobalOptimizer',
                    'target': {
                        'type': 'Crystal',
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
        }
        params_ref = {
            'optimizer': {
                'type': 'GlobalOptimizer',
                'target': {
                    'type': 'Crystal',
                    'conditions': {'externalPressure': 100, 'volumeType': 0},
                    'powderSpectrumAnalyzer': PowderSpectrumAnalyzer.parse('spectrum.txt'),
                    'compositionSpace': {'symbols': ['Na', 'Cl'],
                                         'blocks': [[8,24]],
                                         'range': [[1,1]]},
                    'cellUtility': {'pbc': (1, 1, 1)}
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
        params = compileParams(**definitions)
        self.assertEqual(params, params_ref)
