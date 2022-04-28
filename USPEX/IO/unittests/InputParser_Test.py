import unittest
import os

from ..InputParser import read, write


HOMEPATH = os.path.dirname(os.path.abspath(__file__))


class InputParser_Test(unittest.TestCase):
    def setUp(self) -> None:
        self.params_ref = {
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


    def test_read(self):
        self.assertEqual(read(os.path.join(HOMEPATH, 'input1.uspex')), self.params_ref)

    def test_write_read(self):
        filename = os.path.join(HOMEPATH, 'input_test.uspex')
        write(filename, self.params_ref)
        definitions = read(filename)
        self.assertEqual(definitions, self.params_ref)
        os.remove(filename)

    def test_molecules(self):
        params_ref = {
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
                                              'randTop': (0.1, 1.0, 0.2),
                                          }
                              }
            },
            'stages': [],
            'numParallelCalcs': 20,
            'numGenerations': 30,
            'stopCrit': 6,
        }
        self.assertEqual(read(os.path.join(HOMEPATH, 'input2.uspex')), params_ref)
