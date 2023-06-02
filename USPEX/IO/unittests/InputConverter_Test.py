'''
@file        InputParserTest.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        15 October 2017
@brief       Class for testing InputParser class.
'''

import unittest

from pathlib import Path
from ..InputConverter import InputConverter

TESTPATH = Path(__file__).parent


class InputParser_Test(unittest.TestCase):
    def test_c2(self):
        inputParser = InputConverter('INPUT.txt', wd =TESTPATH/'c2')
        params = inputParser.parse()
        params_ref = {'system': {'type': 'Crystal','heredity': {'initFrac': 0.5},
                                             'random': {'initFrac': 0.2},
                                             'permutation': {'initFrac': 0.1},
                                             'softmodemutation': {'initFrac': 0.2},
                                             'symbols': ['Mg', 'Al', 'O'], 'blocks': [[4, 8, 16]],
                                             'externalPressure': 100.0, 'fixed': [[1, 1]]},
                      'stages': [{'type': 'gulp', 'params': {'commandExecutable': 'gulp < input > output',
                                                             'workingDirectory': './', 'libs': [],
                                                             'ginput': 'Specific/ginput_1', 'goptions': 'Specific/goptions_1'}},
                                 {'type': 'gulp', 'params': {'commandExecutable': 'gulp < input > output',
                                                             'workingDirectory': './', 'libs': [],
                                                             'ginput': 'Specific/ginput_2', 'goptions': 'Specific/goptions_2'}},
                                 {'type': 'gulp', 'params': {'commandExecutable': 'gulp < input > output',
                                                             'workingDirectory': './', 'libs': [],
                                                             'ginput': 'Specific/ginput_3', 'goptions': 'Specific/goptions_3'}},
                                 {'type': 'gulp', 'params': {'commandExecutable': 'gulp < input > output',
                                                             'workingDirectory': './', 'libs': [],
                                                             'ginput': 'Specific/ginput_4', 'goptions': 'Specific/goptions_4'}}],
                      'engine': {'type': 'USPEX', 'popSize': 40, 'initialPopSize': 40},
                      'numGenerations': 60, 'stopCrit': 30,
                      'numParallelCalcs': 1}
        self.assertEqual(params,params_ref)

    def test_c1(self):
        inputParser = InputConverter('INPUT.txt', wd =TESTPATH/'c1')
        params = inputParser.parse()
        HEADER = '#!/bin/sh\n#BSUB -sp 100\n#BSUB -a  intelmpi\n#BSUB -R  "span[ptile=8]"\n#BSUB -J  USPEX\n#BSUB -n  8\n' \
                 '#BSUB -W  06:00\n#BSUB -q  intel\n#BSUB -o  output\n#\n#\n#            -__-  have fun in Rurik cluster @ MIPT !\n#'
        params_ref = {'system': {'type': 'Crystal', 'heredity': {'initFrac': 0.5},
                                             'random': {'initFrac': 0.2},
                                             'softmodemutation': {'initFrac': 0.2},
                                             'symbols': ['Si'], 'blocks': [[8]], 'ionDistances': [[1.1]],
                                             'externalPressure': 1e-05, 'fixed': [[1, 1]]},
                      'stages': [{'type': 'vasp', 'params': {'commandExecutable': 'vasp',
                                                             'workingDirectory': './',
                                           'taskManager': {'type': 'BSUB', 'header': HEADER},
                                           'potcars': ['Specific/POTCAR_Si'],
                                           'kresol': 0.13, 'incar': 'Specific/INCAR_1'}},
                                 {'type': 'vasp', 'params': {'commandExecutable': 'vasp',
                                                             'workingDirectory': './',
                                           'taskManager': {'type': 'BSUB', 'header': HEADER},
                                           'potcars': ['Specific/POTCAR_Si'],
                                           'kresol': 0.11, 'incar': 'Specific/INCAR_2'}},
                                 {'type': 'vasp', 'params': {'commandExecutable': 'vasp',
                                                             'workingDirectory': './',
                                           'taskManager': {'type': 'BSUB', 'header': HEADER},
                                           'potcars': ['Specific/POTCAR_Si'],
                                           'kresol': 0.09, 'incar': 'Specific/INCAR_3'}},
                                 {'type': 'vasp', 'params': {'commandExecutable': 'vasp',
                                                             'workingDirectory': './',
                                           'taskManager': {'type': 'BSUB', 'header': HEADER},
                                           'potcars': ['Specific/POTCAR_Si'],
                                           'kresol': 0.07, 'incar': 'Specific/INCAR_4'}},
                                 {'type': 'vasp', 'params': {'commandExecutable': 'vasp',
                                                             'workingDirectory': './',
                                           'taskManager': {'type': 'BSUB', 'header': HEADER},
                                           'potcars': ['Specific/POTCAR_Si'],
                                           'kresol': 0.05, 'incar': 'Specific/INCAR_5'}},
                                 {'type': 'vasp', 'params': {'commandExecutable': 'vasp',
                                                             'workingDirectory': './',
                                           'taskManager': {'type': 'BSUB', 'header': HEADER},
                                           'potcars': ['Specific/POTCAR_Si'],
                                           'kresol': 0.04, 'incar': 'Specific/INCAR_6'}}],
                      'engine': {'type': 'USPEX', 'popSize': 20, 'initialPopSize': 20},
                      'numGenerations': 25, 'stopCrit': 8,
                      'numParallelCalcs': 10}
        self.assertEqual(params,params_ref)

    def test_c4(self):
        inputParser = InputConverter('INPUT.txt', wd =TESTPATH/'c4/')
        params = inputParser.parse()
        params_ref = {'system': {'type': 'Crystal', 'heredity': {'initFrac': 0.5},
                                             'random': {'initFrac': 0.1},
                                             'permutation': {'initFrac': 0.2},
                                             'softmodemutation': {'initFrac': 0.2},
                                             'symbols': ['C'], 'blocks': [[16]], 'ionDistances': [[1.0]],
                                             'fixed': [[1, 1]]},
                      'stages': [{'type': 'lammps', 'params': {'commandExecutable': 'lammps < lammps.in > lammps.out',
                                                             'workingDirectory': './',
                                             'libs': ['Specific/SiC.tersoff'], 'lammps_in': 'Specific/lammps.in_1'}},
                                 {'type': 'lammps', 'params': {'commandExecutable': 'lammps < lammps.in > lammps.out',
                                                             'workingDirectory': './',
                                             'libs': ['Specific/SiC.tersoff'], 'lammps_in': 'Specific/lammps.in_2'}},
                                 {'type': 'lammps', 'params': {'commandExecutable': 'lammps < lammps.in > lammps.out',
                                                             'workingDirectory': './',
                                             'libs': ['Specific/SiC.tersoff'], 'lammps_in': 'Specific/lammps.in_3'}},
                                 {'type': 'lammps', 'params': {'commandExecutable': 'lammps < lammps.in > lammps.out',
                                                             'workingDirectory': './',
                                             'libs': ['Specific/SiC.tersoff'], 'lammps_in': 'Specific/lammps.in_4'}},
                                 {'type': 'lammps', 'params': {'commandExecutable': 'lammps < lammps.in > lammps.out',
                                                             'workingDirectory': './',
                                             'libs': ['Specific/SiC.tersoff'], 'lammps_in': 'Specific/lammps.in_5'}}],
                      'engine': {'type': 'USPEX','popSize': 30, 'initialPopSize': 30},
                      'numGenerations': 50, 'stopCrit': 20,
                      'numParallelCalcs': 1}
        self.assertEqual(params,params_ref)
