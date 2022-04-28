"""
USPEX.Stages.MOPAC_Interface
============================

.. codeauthor:: Vladimir Baturin <vsbat@yandex.ru>

"""

import logging
import os
import re
import numpy as np
from os.path import join as pj

logger = logging.getLogger(__name__)


class MOPAC_Interface:
    """
     Calculator for Gulp.
     Local running
     """
    outputFile, errorFile = 'output', 'error'
    inputFile, mopacOut, arcFile = 'calc.mop', 'calc.out', 'calc.arc'
    DEFAULT_SLEEP_TIME = 1
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    def __init__(self, tag: str, mop_input: str = None,
                 targetProperties: list = None, **kwargs):
        """

        :param params: dictionary with parameters:
                * mop_input: (str) path to ginput-file.
                * vacuumSize=10
        """

        if mop_input is None:
            mop_input = pj(os.getcwd(), f'Specific/mop_{tag}')
        assert os.path.exists(mop_input)

        with open(mop_input, 'r') as f:
            self.mop_input = f.read().strip()

        # if moleculeSpecifics != None:
        #     self.moleculeSpecifics = moleculeSpecifics
        # else:
        #     self.moleculeSpecifics = {}
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']

        logger.debug('MOPAC calculator created.')

    def prepareLocalCalculation(self, system, calcFolder: str):
        """

        :param system:
        :param isFullRelaxation:
        """

        structure, disassembler = self.structureType.assemble(**system, vacuumSize=0)
        system['disassembler'] = disassembler
        cell = structure.getCell()
        system['assembledCell'] = cell
        coordinates = structure.getCartesianCoordinates()

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
        """
        :param SYSTEM:
        :return: whether optimization converged
        """

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
            content = arc_fid.readlines()

        if 'structure' in self.targetProperties:
            self.readStructure(system, content)
        if 'enthalpy' in self.targetProperties:
            for line in content:
                if 'TOTAL ENERGY' in line:
                    e = re.match(r'\s*TOTAL ENERGY\s*=\s*(\S+)\s*EV', line)
                    system['enthalpy'] = float(e.group(1))
                break

    def readStructure(self, system, content):

        assembledCell = system.pop('assembledCell')
        disassembler = system.pop('disassembler')

        atomTypes = []
        positions = []
        lattice_vectors = assembledCell.getCellVectors()
        new_lattice = []
        for i, line in enumerate(content):
            if 'FINAL GEOMETRY OBTAINED' in line:
                coords_regex = r'\s*([A-Z][a-z]?)' + r'\s+(-?\d*\.\d+)\s+\S+' * 3
                for line in content[i:]:
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
                for i, dim in enumerate(assembledCell.getPBC()):
                    if dim:
                        lattice_vectors[i] = np.asarray(new_lattice.pop(0))
        cell = self.cellType(lattice_vectors, pbc=assembledCell.getPBC())
        structure = self.structureType(atomTypes, positions, cell=cell)
        system.update(disassembler.disassemble(structure))

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType