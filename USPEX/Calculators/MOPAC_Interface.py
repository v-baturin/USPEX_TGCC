'''
@file        MOPAC_Interface.py
@author:     Vladimir Baturin
@copyright:  2021 Oganov's Lab. All rights reserved.
@contact:    vsbat@yandex.ru
@date        21 Jul 2021
@brief       Class for calculator of LAMMPS
'''

__author__ = 'v.baturin'

import logging
import os
import re
import shutil

import numpy as np
from os.path import join as pj

# from ..Atomistic.Transformation import Transformation
from .Common.SHELL_Interface import SHELL_Interface

logger = logging.getLogger(__name__)


class MOPAC_Interface(SHELL_Interface):
    """
     Calculator for Gulp.
     Local running
     """
    inputFile, mopacOut, arcFile = 'calc.mop', 'calc.out', 'calc.arc'
    _DEFAULT_SLEEP_TIME = 1
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    def __init__(self, tag: str, mop_input: str = None,
                 **kwargs):
        '''

        :param params: dictionary with parameters:
                * mop_input: (str) path to ginput-file.
                * vacuumSize=10
        '''

        super().__init__(**kwargs)
        if mop_input is None:
            mop_input = pj(os.getcwd(), f'Specific/mop_{tag}')
        assert os.path.exists(mop_input)

        with open(mop_input, 'r') as f:
            self.mop_input = f.read().strip()

        # if moleculeSpecifics != None:
        #     self.moleculeSpecifics = moleculeSpecifics
        # else:
        #     self.moleculeSpecifics = {}

        logger.debug('MOPAC calculator created.')

    def prepareLocalCalculation(self, system, calcFolder: str):
        '''

        :param system:
        :param isFullRelaxation:
        '''

        structure, disassembler = self.structureType.assemble(**system)
        system['disassembler'] = disassembler

        coordinates = structure.getCartesianCoordinates()
        cell = structure.getCell()

        # files_to_delete = ['output', 'optimized.structure']
        # for f in files_to_delete:
        #     if os.path.isfile(f):
        #         os.remove(f)

        content_to_write = ''

        fixedIndices = disassembler.envIndices[system['environment'].getFixedIndices()] if 'environment' in system else []
        for i, (symbol, coord) in enumerate(zip(structure.getAtomTypes(), coordinates)):
            tuple_to_format = tuple([symbol.short_name] + coord.tolist())
            if i in fixedIndices:
                content_to_write += '%4s %12.6f 0 %12.6f 0 %12.6f 0\n' % tuple_to_format
            else:
                content_to_write += '%4s %12.6f 1 %12.6f 1 %12.6f 1\n' % tuple_to_format

        for i, dim in enumerate(cell.getPBC()):
            if dim:
                content_to_write += 'Tv %12.6f 1 %12.6f 1 %12.6f 1\n' % tuple(cell.getCellVectors()[i])

        if system['externalPressure'] >= 0.05:
            self.mop_input += f" P={system['externalPressure']:.2f}Gpa\n"

        total_content = self.mop_input + '\n' + content_to_write + '\n'

        with open(pj(calcFolder, self.inputFile), 'wt') as f:
            f.write(total_content)

        logger.debug('MOPAC calculator prepared calculation.')

    def isConverged(self, calcFolder: str):
        '''
        :param SYSTEM:
        :return: whether optimization converged
        '''

        if not (os.path.exists(pj(calcFolder, self.mopacOut))
                and os.path.exists(pj(calcFolder, self.arcFile))):
            return False

        with open(pj(calcFolder, self.arcFile), 'rt') as arc_fid:
            arc_content = arc_fid.read()
            if 'FINAL GEOMETRY OBTAINED' not in arc_content:
                return False
            else:
                return True

    def readOutput(self, system, calcFolder: str):

        with open(pj(calcFolder, self.arcFile), 'rt') as arc_fid:

            cell = system['cell']
            disassembler = system['disassembler']
            del system['disassembler']


            #  Parsing energy
            line = ''
            while 'TOTAL ENERGY' not in line:
                line = arc_fid.readline()
            e = re.match(r'\s*TOTAL ENERGY\s*=\s*(\S+)\s*EV', line)
            system['enthalpy'] = float(e.group(1))


            # Parsing geometry
            while 'FINAL GEOMETRY OBTAINED' not in line:
                line = arc_fid.readline()
            atomTypes = []
            positions = []
            new_lattice = []
            coords_regex = r'\s*([A-Z][a-z]?)' + r'\s+(-?\d*\.\d+)\s+\S+' * 3
            for line in arc_fid:
                coordsgroup = re.findall(coords_regex, line)
                if coordsgroup:
                    sym = coordsgroup[0][0]
                    vector = [float(x) for x in coordsgroup[0][1:]]
                    if sym == 'Tv':
                        new_lattice.append(vector)
                    else:
                        positions.append(vector)
                        atomTypes.append(self.atomType(sym))

            positions = np.asarray(positions)
            lattice_vectors = cell.getCellVectors()
            for i, dim in enumerate(cell.getPBC()):
                if dim:
                    lattice_vectors[i] = np.asarray(new_lattice.pop(0))
            cell = self.cellType(lattice_vectors, pbc=cell.getPBC())
            cell = cell.getEnvelopeCell(positions, 0)
            positions = cell.center(positions)
            structure = self.structureType(atomTypes, positions, cell=cell)
            system.update(disassembler.disassemble(structure))

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType