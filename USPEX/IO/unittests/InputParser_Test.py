import unittest
import os

from ..InputParser import read


HOMEPATH = os.path.dirname(os.path.abspath(__file__))


class InputParser_Test(unittest.TestCase):
    def setUp(self) -> None:
        self.params_ref = {
            'optimizer': {
                'type': 'GlobalOptimizer',
                'target': {
                    'type': 'Crystal',
                    'conditions': {'externalPressure': 100},
                    'compositionSpace': {'symbols': ['Mg', 'Al', 'O'],
                                         'blocks': [[4, 8, 16]]},
                },
                'optType': 'enthalpy',
                'stopFitness': -655.062,
                'selection': {
                    'type': 'USPEXClassic',
                    'popSize': 10,
                    'optType': ('aging', 'enthalpy'),
                    'fractions': {
                        'heredity': (0.0, 1.0, 0.5),
                        'twinning': (0.0, 1.0, 0.1),
                        'softmodemutation': (0.0, 1.0, 0.1),
                        'randSym': (0.0, 1.0, 0.0),
                        'randTop': (0.0, 1.0, 0.2),
                        'permutation': (0.0, 1.0, 0.1)
                    }
                }
            },
            'stages': [
                {'type': 'gulp', 'commandExecutable': 'gulp', 'goptions': './Specific/goptions'},
                {'type': 'gulp', 'commandExecutable': 'gulp', 'goptions': './Specific/goptions'},
                {'type': 'gulp', 'commandExecutable': 'gulp', 'goptions': './Specific/goptions'},
                {'type': 'gulp', 'commandExecutable': 'gulp', 'goptions': './Specific/goptions'},
                {'type': 'gulp', 'commandExecutable': 'gulp', 'goptions': './Specific/goptions',
                 'ginput': './Specific/ginput_4'}],
            'numParallelCalcs': 2,
            'numGenerations': 3,
            'stopCrit': 3
        }


    def test_read(self):
        self.assertEqual(read(os.path.join(HOMEPATH, 'input.uspex')), self.params_ref)

    # def test_write_read(self):
    #     filename = os.path.join(HOMEPATH, 'input_test.uspex')
    #     write(filename, self.reference)
    #     definitions = read(filename)
    #     self.assertEqual(definitions, self.reference)
    #     os.remove(filename)
