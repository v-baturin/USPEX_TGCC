"""
USPEX.Calculators.MLIP_Interface
================================

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>

"""
import logging
import os
import shutil
import numpy as np
from os.path import join as pj
from ase.atoms import Atoms

from USPEX.Calculators.Interfaces.MLIPCfgParser import readcfg, savecfg
from USPEX.Calculators.SHELL_Interface import SHELL_Interface

logger = logging.getLogger(__name__)
EV_PER_CUBIC_ANGSTREM_PER_GPA = 1 / 160.21766208


class MLIP_Interface(SHELL_Interface):
    '''
    Calculator for MLIP.
    Local running
    '''

    # working input files
    in_cfg_file = 'for_relax.cfg'

    # working output files
    out_cfg_file = 'relaxed.cfg_0'
    out_sampled_file = 'sampled.cfg_0'

    _DEFAULT_SLEEP_TIME = 10
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    def __init__(self, tag: str, input: str = None, potential: str = None, vacuumSize = 10,
                 targetProperties: list = None, **kwargs):
        super().__init__(**kwargs)

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

    def prepareLocalCalculation(self, system, calcFolder: str):
        structure, disassembler = self.structureType.assemble(**system, vacuumSize=self.vacuumSize)
        system['disassembler'] = disassembler
        cell = structure.getCell()
        system['assembledCell'] = cell
        coordinates = structure.getCartesianCoordinates()
        atomTypes = structure.getAtomTypes()

        # create empty input file
        with open(pj(calcFolder, self.inputFile), 'wt') as f:
            pass

        # cfg file
        atoms = Atoms([el.short_name for el in atomTypes], positions = coordinates,
                      cell = cell.getCellVectors())
        savecfg(pj(calcFolder, self.in_cfg_file), atoms)

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
        atoms = readcfg(pj(calcFolder, self.out_cfg_file))
        if atoms:
            if 'structure' in self.targetProperties:
                self.readStructure(system, atoms)
            if 'enthalpy' in self.targetProperties:
                system['enthalpy'] = atoms.energy + atoms.get_volume() * \
                                     system['externalPressure'] * EV_PER_CUBIC_ANGSTREM_PER_GPA
            if 'stressTensor' in self.targetProperties:
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
        # else:
        #     ID = system['ID']
        #     logger.info(f'structure {ID} led to extrapolation and will be discarded.')
        #     # system['structure'].set_cell(np.identity(3) * system['structure'].minVectorLength * 0.9)
        #     system['enthalpy'] = 1000

    def readStructure(self, system, aseStructure):
        assembledCell = system.pop('assembledCell')
        disassembler = system.pop('disassembler')

        cell = self.cellType(aseStructure.get_cell().array, assembledCell.getPBC())
        positions = aseStructure.get_positions()
        structure = self.structureType([self.atomType(el) for el in aseStructure.get_chemical_symbols()], positions,
                                       cell=cell)
        system.update(disassembler.disassemble(structure))


    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType
