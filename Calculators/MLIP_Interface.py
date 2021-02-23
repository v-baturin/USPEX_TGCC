import logging
logger = logging.getLogger(__name__)

'''
@file        MLIP_Interface.py
@author:     Michele Galasso
@copyright:  2020 Oganov's Lab. All rights reserved.
@contact:    m.galasso@yandex.com
@date        27 December 2020
@brief       Class for calculator of MLIP
'''

import os
import shutil
import numpy as np

from os.path import join as pj

from .Common.MLIPCfgParser import readcfg, savecfg
from .Common.SHELL_Interface import SHELL_Interface


EV_PER_CUBIC_ANGSTREM_PER_GPA = 1/160.21766208


class MLIP_Interface(SHELL_Interface):
    '''
    Calculator for MLIP.
    Local running
    '''

    # working input files
    in_cfg_file = 'for_relax.cfg'

    # working output files
    out_cfg_file = 'relaxed.cfg_0'


    _DEFAULT_SLEEP_TIME = 10

    def __init__(self, tag: str, input: str = None, potential: str = None, **kwargs):
        super().__init__(**kwargs)

        if input is not None:
            self.input = input
        else:
            self.input = pj(os.getcwd(), f'Specific/input_{tag}.ini')

        if potential is not None:
            self.potential = potential
        else:
            self.potential = pj(os.getcwd(), 'Specific/potential.mtp')

    def prepareLocalCalculation(self, system, calcFolder: str):
        # create empty input file
        with open(pj(calcFolder, self.inputFile), 'wt') as f:
            pass

        # cfg file
        savecfg(pj(calcFolder, self.in_cfg_file), system['structure'].atoms)

        # input file
        shutil.copy2(self.input, calcFolder)

        # potential file
        shutil.copy2(self.potential, calcFolder)

    def isConverged(self, calcFolder: str):
        if os.path.isfile(pj(calcFolder, self.out_cfg_file)):
            with open(pj(calcFolder, self.out_cfg_file), 'r') as f:
                content = f.read()
            if content:
                return True
        return False

    def readOutput(self, system, calcFolder: str):
        atoms = readcfg(pj(calcFolder, self.out_cfg_file))

        system['structure'].set_cell(atoms.get_cell(), optimize=True)
        system['structure'].set_positions(atoms.get_positions())
        system['enthalpy'] = atoms.energy + system['structure'].get_volume() * \
                             system['structure'].externalPressure * EV_PER_CUBIC_ANGSTREM_PER_GPA

        try:
            stress_tensor = np.zeros((3, 3))
            stress_tensor[0, 0] = atoms.stresses[0]
            stress_tensor[1, 1] = atoms.stresses[1]
            stress_tensor[2, 2] = atoms.stresses[2]
            stress_tensor[1, 2] = atoms.stresses[3]
            stress_tensor[2, 1] = atoms.stresses[3]
            stress_tensor[0, 2] = atoms.stresses[4]
            stress_tensor[2, 0] = atoms.stresses[4]
            stress_tensor[0, 1] = atoms.stresses[5]
            stress_tensor[1, 0] = atoms.stresses[5]

            system['stressTensor'] = stress_tensor
        except:
            logger.debug('Pressure tensor can\'t be find in output')
