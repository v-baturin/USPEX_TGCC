"""
USPEX.Stages.MLIP_Interface
===========================

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>

"""
import logging
import os
import shutil
import numpy as np
from os.path import join as pj

from ...Presets import udateSystemWithPrefix as usp


logger = logging.getLogger(__name__)
EV_PER_CUBIC_ANGSTREM_PER_GPA = 1 / 160.21766208


class MLIP_Interface:
    '''
    Calculator for MLIP.
    Local running
    '''


    inputFile, outputFile, errorFile = 'input', 'output', 'error'

    # working input files
    in_cfg_file = 'for_relax.cfg'

    # working output files
    out_cfg_file = 'relaxed.cfg_0'
    out_sampled_file = 'sampled.cfg_0'

    DEFAULT_SLEEP_TIME = 10
    atomisticRepresentation = None
    atomicDisassemblerType = None

    @classmethod
    def registerTypes(cls, atomisticRepresentation, atomicDisassemblerType):
        cls.atomisticRepresentation = atomisticRepresentation
        cls.atomicDisassemblerType = atomicDisassemblerType

    def __init__(self, tag: str, input: str = None, potential: str = None, vacuumSize = 10,
                 environmentStyle=None, inStyle=None, targetProperties: list = None, **kwargs):

        self.tag = tag
        self.tmp = f'tmp_{tag}'
        if input is not None:
            self.input = input
        else:
            self.input = pj(os.getcwd(), f'Specific/input_{tag}.ini')

        if potential is not None:
            self.potential = potential
        else:
            self.potential = pj(os.getcwd(), 'Specific/potential.mtp')
        self.vacuumSize = vacuumSize
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']
        self.environmentStyle = environmentStyle
        self.inStyle = inStyle

    def prepareLocalCalculation(self, system, calcFolder: str):
        structure, disassembler = self.atomicDisassemblerType.assemble(**system,
                                                                       style=self.environmentStyle,
                                                                       inStyle=self.inStyle,
                                                                       vacuumSize=self.vacuumSize)
        system[self.tmp]['disassembler'] = disassembler
        cell = structure.getCell()
        system['pbc'] = cell.getPBC()

        # create empty input file
        with open(pj(calcFolder, self.inputFile), 'wt') as f:
            pass

        # cfg file
        self.atomisticRepresentation.saveMLIPcfg(pj(calcFolder, self.in_cfg_file), structure, system)

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
            # if the structure ended up unrelaxed because of extrapolation
            elif os.path.isfile(pj(calcFolder, self.out_sampled_file)):
                with open(pj(calcFolder, self.errorFile)) as stderr:
                    content = stderr.read()
                if not content:
                    return True
        return False

    def readOutput(self, system, calcFolder: str):
        with open(pj(calcFolder, self.out_cfg_file)) as f:
            data = self.atomisticRepresentation.readMLIPcfg(f, )
        if 'structure' in self.targetProperties:
            usp(system, system[self.tmp].pop('disassembler').disassemble(data['structure']), 'system', self.environmentStyle)
        if 'enthalpy' in self.targetProperties:
            enthalpy = data['energy'] + \
                       data['structure'].getCell().getVolume() * system['externalPressure'] * EV_PER_CUBIC_ANGSTREM_PER_GPA
            usp(system, enthalpy, 'enthalpy', self.environmentStyle)
        if 'stressTensor' in self.targetProperties:
            stress_tensor = np.zeros((3, 3))
            stress_tensor[0, 0] = data['stresses'][0]
            stress_tensor[1, 1] = data['stresses'][1]
            stress_tensor[2, 2] = data['stresses'][2]
            stress_tensor[1, 2] = data['stresses'][3]
            stress_tensor[2, 1] = data['stresses'][3]
            stress_tensor[0, 2] = data['stresses'][4]
            stress_tensor[2, 0] = data['stresses'][4]
            stress_tensor[0, 1] = data['stresses'][5]
            stress_tensor[1, 0] = data['stresses'][5]
            system['stressTensor'] = stress_tensor
            usp(system, stress_tensor, 'stressTensor', self.environmentStyle)

        # else:
        #     ID = system['ID']
        #     logger.info(f'structure {ID} led to extrapolation and will be discarded.')
        #     # system['structure'].set_cell(np.identity(3) * system['structure'].minVectorLength * 0.9)
        #     system['enthalpy'] = 1000
