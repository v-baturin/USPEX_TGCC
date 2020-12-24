import unittest
import os

from ..InputParser import read, write


HOMEPATH = os.path.dirname(os.path.abspath(__file__))


class InputParser_Test(unittest.TestCase):
    def setUp(self) -> None:
        self.reference = {'gulp': {'commandExecutable': 'gulp', 'goptions': './Specific/goptions', 'type': 'gulp'},
                          'gulp5': {'commandExecutable': 'gulp',
                                 'ginput': './Specific/ginput_4',
                                 'goptions': './Specific/goptions',
                                 'type': 'gulp'},
                          'main': {'numGenerations': 3,
                                'numParallelCalcs': 2,
                                'optimizer': {'fitness': 'enthalpy',
                                              'selection': {'fitness': ('getAntiseedsCorrections', 'enthalpy'),
                                                            'fractions': {'heredity': [0.0, 1.0, 0.5],
                                                                          'permutation': [0.0, 1.0, 0.1],
                                                                          'randSym': [0.0, 1.0, 0.0],
                                                                          'randTop': [0.0, 1.0, 0.2],
                                                                          'softmodemutation': [0.0, 1.0, 0.1],
                                                                          'twinning': [0.0, 1.0, 0.1]},
                                                            'popSize': 10,
                                                            'type': 'USPEXClassic'},
                                              'stopFitness': -655.062,
                                              'target': {'compositionSpace': {'blocks': [[4, 8, 16]],
                                                                              'range': [[1, 1]],
                                                                              'symbols': ['Mg', 'Al', 'O']},
                                                         'config': {'externalPressure': 100},
                                                         'type': 'Crystal'},
                                              'type': 'GlobalOptimizer'},
                                'stages': ['gulp', 'gulp', 'gulp', 'gulp', 'gulp5'],
                                'stopCrit': 3}}


    def test_read(self):
        definitions = read(os.path.join(HOMEPATH, 'input.uspex'))
        self.assertEqual(definitions, self.reference)

    def test_write_read(self):
        filename = os.path.join(HOMEPATH, 'input_test.uspex')
        write(filename, self.reference)
        definitions = read(filename)
        self.assertEqual(definitions, self.reference)
        os.remove(filename)
