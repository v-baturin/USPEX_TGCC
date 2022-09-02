"""
USPEX.Stages.Interfaces.QE_Interface
====================================

"""

import logging
import shutil
import numpy as np

from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.io.espresso import read_fortran_namelist, read_espresso_out, write_espresso_in
from pathlib import Path

from .KPoints import KPoints, BadKPoints

logger = logging.getLogger(__name__)

class QE_Interface:
    '''
    Calculator for QE.
    Local running
    '''


    SPECIFIC_FOLDER = Path.cwd()/'Specific'
    inputFile, outputFile, errorFile = 'input', 'output', 'error'

    DEFAULT_SLEEP_TIME = 30
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    def __init__(self, tag: str,
                 kresol: float,
                 options: str = None,
                 pseudopotentials: dict = None,
                 vacuumSize: float = 10.0,      # Angtrom
                 targetProperties: list = None,
                 **kwargs):
        '''

        :param tag: tag of the stage
        :param kresol: float of K-points resolution
        :param options:(str) path to qEspresso_options-file.
        :param pseudopotentials: (dict) A filename for each atomic species, e.g.
            {'O': 'O.pbe-rrkjus.UPF', 'H': 'H.pbe-rrkjus.UPF'}.
        :param libs: (list) list of paths to interatomic potentials.
        :param kwargs:
        '''

        self.options = Path.cwd()/f'Specific/qEspresso_options_{tag}' if not options else options
        with open(options) as fp:
            data, card_lines = read_fortran_namelist(fp)
        if 'system' not in data:
            raise KeyError('Required section &SYSTEM not found.')
        self.data = data

        self.pseudopotentials = {x: Path(p) for x, p in pseudopotentials.items()}
        assert kresol > 0
        self.kPoints = KPoints(kresol)
        self.vacuumSize = vacuumSize
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']

    def prepareLocalCalculation(self, system: dict, calcFolder: Path):
        calcFolder = Path(calcFolder)
        structure, disassembler = self.structureType.assemble(**system, vacuumSize=self.vacuumSize)
        system['disassembler'] = disassembler
        cell = structure.getCell()
        system['assembledCell'] = cell
        fixedIndices = disassembler.envIndices[system['environment'].getFixedIndices()] if 'environment' in system else []

        if 'environmentEnthalpy' in self.targetProperties:
            environment = system['environment'].getStructure()
            atoms = Atoms(symbols=[el.short_name for el in environment.getAtomTypes()],
                          positions=environment.getCartesianCoordinates(),
                          cell=cell.getCellVectors())
        else:
            atoms = Atoms(symbols=[el.short_name for el in structure.getAtomTypes()],
                          positions=structure.getCartesianCoordinates(),
                          cell=cell.getCellVectors())
            if 'environment' in system:
                indices = disassembler.envIndices[system['environment'].getFixedIndices()]
                atoms.set_constraint(FixAtoms(indices=fixedIndices))

        # Copying pseudopotentials to calc folder
        for s, pseudo in self.pseudopotentials.items():
            if Path(pseudo).exists():
                shutil.copy(pseudo, calcFolder)

        ############################# KPOINTS #################################
        try:
            kPoints = self.kPoints.build(cell)
        except BadKPoints:
            # This LATTICE is extremely wrong, let's skip it from now
            logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
            kPoints = [1, 1, 1]

        with open(calcFolder/self.inputFile, 'wt') as dest:
            write_espresso_in(fd=dest, atoms=atoms, input_data=self.data,
                              pseudopotentials={s:p.name for s,p in self.pseudopotentials.items()},
                              kpts=kPoints,
                              crystal_coordinates=True)

    def isConverged(self, calcFolder: Path):
        calcFolder = Path(calcFolder)
        if not calcFolder.joinpath(self.outputFile).exists():
            res = False
        else:
            with open(calcFolder/self.outputFile, 'rt') as out:
                res = 'JOB DONE' in out.read()
        if not res:
            logger.error('Quantum Espresso is not completely Done')
        return res

    def readOutput(self, system: dict, calcFolder: Path):
        calcFolder = Path(calcFolder)
        with open(calcFolder/self.outputFile, 'rt') as f:
            aseStructure = next(read_espresso_out(f, index=slice(None, -2, -1)))
            f.seek(0)
            content = f.readlines()

        if aseStructure:
            if 'structure' in self.targetProperties:
                self.readStructure(system, aseStructure)
            if 'enthalpy' in self.targetProperties:
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
