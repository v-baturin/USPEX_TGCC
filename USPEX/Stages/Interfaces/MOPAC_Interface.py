"""
USPEX.Stages.MOPAC_Interface
============================

.. codeauthor:: Vladimir Baturin <vsbat@yandex.ru>

"""

import logging
import re
import numpy as np

from pathlib import Path


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

    @classmethod
    def registerTypes(cls, structureType, atomType, cellType):
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType

    def __init__(self, tag: str, mop_input: str = None, targetProperties: list = None, **kwargs):
        """

        :param params: dictionary with parameters:
                * mop_input: (str) path to ginput-file.
                * vacuumSize=10
        """

        self.tag = tag

        mop_input = Path.cwd()/f'Specific/mop_{tag}' if mop_input is None else Path(mop_input)
        assert mop_input.exists()

        with open(mop_input, 'r') as f:
            self.mop_input = f.read().strip()

        # if moleculeSpecifics != None:
        #     self.moleculeSpecifics = moleculeSpecifics
        # else:
        #     self.moleculeSpecifics = {}
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']

        logger.debug('MOPAC calculator created.')

    def prepareLocalCalculation(self, system, calcFolder: Path):
        """

        :param system:
        :param calcFolder:
        """
        structure = system.getProperty('structure', extension='atomistic', suffix='intermediate')

        cell = structure.getCell()
        with open(calcFolder/'pbc', 'wt') as f:
            f.write(' '.join(f'{c}' for c in cell.getPBC()))

        # files_to_delete = ['output', 'optimized.structure']
        # for f in files_to_delete:
        #     if os.path.isfile(f):
        #         os.remove(f)

        content_to_write = ''

        for i, (symbol, coord) in enumerate(zip(structure.getAtomTypes(), structure.getCartesianCoordinates())):
            tuple_to_format = (symbol.short_name, ) +\
                              tuple(np.format_float_positional(c if not np.isclose(c, 0) else 0, unique=False,
                                                               precision=6) for c in coord)
            disassembler = system.getProperty('disassembler', extension='atomistic', suffix='intermediate')
            if i in disassembler.allFixedIndices:
                content_to_write += '%4s %12s 0 %12s 0 %12s 0\n' % tuple_to_format
            else:
                content_to_write += '%4s %12s 1 %12s 1 %12s 1\n' % tuple_to_format

        for i, dim in enumerate(cell.getPBC()):
            if dim:
                content_to_write += 'Tv %12.6f 1 %12.6f 1 %12.6f 1\n' % tuple(cell.getCellVectors()[i])

        externalPressure = system.getProperty('externalPressure', suffix='origin')
        if externalPressure >= 0.05:
            self.mop_input += f" P={externalPressure:.2f}Gpa\n"

        total_content = self.mop_input + '\n' + content_to_write + '\n'

        with open(calcFolder/self.inputFile, 'wt') as f:
            f.write(total_content)

        logger.debug('MOPAC calculator prepared calculation.')
        return ''

    def isConverged(self, calcFolder: Path):
        """
        :param calcFolder:
        :return: whether optimization converged
        """

        if not (calcFolder.joinpath(self.mopacOut).exists()
                and calcFolder.joinpath(self.arcFile).exists()):
            return False

        with open(calcFolder/self.arcFile, 'rt') as arc_fid:
            arc_content = arc_fid.read()
            return 'FINAL GEOMETRY OBTAINED' in arc_content

    def readOutput(self, system, calcFolder: Path):
        with open(calcFolder/self.arcFile, 'rt') as arc_fid:
            content = arc_fid.readlines()

        if 'structure' in self.targetProperties:
            with open(calcFolder/'pbc', 'rt') as f:
                pbc = tuple(int(c) for c in f.read().split())
            system.setProperty('structure', self.readStructure(content, pbc), extension='atomistic', suffix=self.tag)


        if 'enthalpy' in self.targetProperties:
            for line in content:
                if 'TOTAL ENERGY' in line:
                    e = re.match(r'\s*TOTAL ENERGY\s*=\s*(\S+)\s*EV', line)
                    system.setProperty('enthalpy', float(e.group(1)), suffix=self.tag)
                    break
            else:
                raise RuntimeError('Can not read enthalpy.')

    def readStructure(self, content, pbc):
        atomTypes = []
        positions = []
        new_lattice = []
        cell = None
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
                cell = self.cellType.initFromCellVectors(pbc, new_lattice)
        return self.structureType(atomTypes, positions, cell=cell)
