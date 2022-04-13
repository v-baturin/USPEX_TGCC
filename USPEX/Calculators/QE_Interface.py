"""
USPEX.Calculators.QE_Interface
==============================

"""

import logging
import os
import shutil
import numpy as np
from os.path import join as pj
from ase.io.espresso import read_espresso_out

from .Common.KPoints import KPoints, BadKPoints
from .Common.SHELL_Interface import SHELL_Interface

logger = logging.getLogger(__name__)

class QE_Interface(SHELL_Interface):
    '''
    Calculator for QE.
    Local running
    '''


    SPECIFIC_FOLDER = os.getcwd() + '/Specific'

    _DEFAULT_SLEEP_TIME = 30
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    def __init__(self, tag : str, kresol : float, options: str = None, libs: list = None, vacuumSize: float = 10,
                 targetProperties: list = None, **kwargs):
        '''

        :param tag: tag of the stage
        :param kresol: float of K-points resolution
        :param options:(str) path to qEspresso_options-file.
        :param libs: (list) list of paths to interatomic potentials.
        :param kwargs:
        '''
        super().__init__(**kwargs)

        if options is not None:
            self.options = options
        else:
            self.options = pj(os.getcwd(), f'Specific/qEspresso_options_{tag}')
        self.libs = libs if libs else []
        self.kPoints = KPoints(kresol)
        self.vacuumSize = vacuumSize
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']


    def prepareLocalCalculation(self, system: dict, calcFolder: str):
        structure, disassembler = self.structureType.assemble(**system, vacuumSize=self.vacuumSize)
        system['disassembler'] = disassembler
        cell = structure.getCell()
        system['assembledCell'] = cell
        coordinates = structure.getCartesianCoordinates()


        atomTypes = structure.getAtomTypes()
        numIons_size = len(np.unique(atomTypes))

        with open(self.options, 'rt') as source:
            data = source.readlines()
        for i, line in enumerate(data):
            if 'AAAA' in line:
                data[i] = line.replace('AAAA', '{}'.format(len(atomTypes)))
            elif 'BBBB' in line:
                data[i] = line.replace('BBBB', '{}'.format(numIons_size))


        data.append('CELL_PARAMETERS bohr\n')

        BOHR = 0.52917721067  # Angstrom
        lat = cell.getCellVectors() / BOHR

        data.append('{:8.4f} {:8.4f} {:8.4f}\n'.format(*lat[0, :]))
        data.append('{:8.4f} {:8.4f} {:8.4f}\n'.format(*lat[1, :]))
        data.append('{:8.4f} {:8.4f} {:8.4f}\n'.format(*lat[2, :]))

        data.append('ATOMIC_POSITIONS {crystal}\n')

        fixedIndices = disassembler.envIndices[system['environment'].getFixedIndices()] if 'environment' in system else []
        for i, (symbol, coord) in enumerate(zip(structure.getAtomTypes(), cell.cartesianToFractional(coordinates))):
            if cell.dim == 2:
                if i in fixedIndices:
                    data.append('{:4s} {:12.6f} {:12.6f} {:12.6f}  1  1  1\n'.format(symbol.short_name, *coord))
                else:
                    data.append('{:4s} {:12.6f} {:12.6f} {:12.6f}  0  0  0\n'.format(symbol.short_name, *coord))
            else:
                data.append('{:4s} {:12.6f} {:12.6f} {:12.6f}\n'.format(symbol.short_name, *coord))


        ############################# KPOINTS #################################
        try:
            kPoints = self.kPoints.build(cell)
        except BadKPoints:
            # This LATTICE is extremely wrong, let's skip it from now
            logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
            kPoints = [1, 1, 1]

        data.append('K_POINTS {automatic}\n')
        data.append('{:4d} {:4d} {:4d}  0 0 0\n'.format(*kPoints))


        with open(pj(calcFolder, self.inputFile), 'wt') as dest:
            dest.write(''.join(data))

        for lib in self.libs:
            if isinstance(lib,str) and os.path.exists(lib):
                shutil.copy(lib, calcFolder)


    def isConverged(self, calcFolder: str):
        if not os.path.exists(pj(calcFolder, self.outputFile)):
            res = False
        else:
            with open(pj(calcFolder, self.outputFile), 'rt') as out:
                res = 'JOB DONE' in out.read()
        if not res:
            logger.error('Quantum Espresso is not completely Done')
        return res

    def readOutput(self, system : dict, calcFolder: str):

        with open(pj(calcFolder, self.outputFile), 'rt') as f:
            aseStructure = next(read_espresso_out(f, index=slice(None, -2, -1)))
            f.seek(0)
            content = f.readlines()

        if aseStructure:
            if 'structure' in self.targetProperties:
                self.readStructure(system, aseStructure)
            if 'enthalpy' in  self.targetProperties:
                system['enthalpy'] = aseStructure.get_calculator().results['energy']
            if 'forces' in self.targetProperties:
                system['forces'] = np.copy(aseStructure.get_calculator().results['forces'])
        if 'stressTensor' in self.targetProperties:
            system['stressTensor'] = self.readStressTensor(content)

    def readStructure(self, system, aseStructure):
        assembledCell = system.pop('assembledCell')
        disassembler = system.pop('disassembler')

        cell = self.cellType(aseStructure.get_cell().array, assembledCell.getPBC())
        positions = aseStructure.get_positions()
        structure = self.structureType([self.atomType(el) for el in aseStructure.get_chemical_symbols()], positions,
                                       cell=cell)
        system.update(disassembler.disassemble(structure))

    def readStressTensor(self, content):
        stressTensor = np.zeros((3, 3), dtype=float)
        for i, line in enumerate(content):
            if 'total   stress' in line:
                for row in content[i + 1: i + 4]:
                    stressTensor[i, :] = np.array(row.split()[3: 6], dtype=float)
        return stressTensor

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType
