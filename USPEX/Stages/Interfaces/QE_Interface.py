"""
USPEX.Stages.Interfaces.QE_Interface
====================================

"""

import logging
import shutil
import numpy as np

from pathlib import Path
from ase.io.espresso import read_fortran_namelist, read_espresso_out, write_espresso_in
from ase.constraints import FixAtoms


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

    AtomicStructureRepresentation = None

    @classmethod
    def registerTypes(cls, AtomicStructureRepresentation):
        cls.AtomicStructureRepresentation = AtomicStructureRepresentation

    def __init__(self, tag: str,
                 kresol: float,
                 options: str = None,
                 pseudopotentials: dict = None,
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

        self.tag = tag
        self.options = Path.cwd()/f'Specific/qEspresso_options_{tag}' if not options else Path(options)

        self.pseudopotentials = {s:Path(p) for s,p in pseudopotentials.items()}
        for x, p in self.pseudopotentials.items():
            assert p.exists()

        assert kresol > 0
        self.kPoints = KPoints(kresol)

        with open(options) as fp:
            data, card_lines = read_fortran_namelist(fp)
        if 'system' not in data:
            raise KeyError('Required section &SYSTEM not found.')
        self.data = data
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']

    def prepareLocalCalculation(self, system, calcFolder: Path):
        structure = system.getProperty('structure', extension='atomistic')
        cell = structure.getCell()

        # Copying pseudopotentials to calc folder
        for s, pseudo in self.pseudopotentials.items():
            if pseudo.exists():
                shutil.copy(pseudo, calcFolder)

        ############################# KPOINTS #################################
        try:
            kPoints = self.kPoints.build(structure.getCell())
        except BadKPoints:
            # This LATTICE is extremely wrong, let's skip it from now
            logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
            kPoints = [1, 1, 1]


        with open(calcFolder/'pbc', 'wt') as f:
            f.write(' '.join(f'{c}' for c in cell.getPBC()))

        disassembler = system.getProperty('disassembler', extension='atomistic')
        atoms = self.AtomicStructureRepresentation.toAtoms(structure)
        if len(disassembler.fixedIndices) > 0:
            atoms.set_constraint(FixAtoms(indices=disassembler.fixedIndices))
        with open(calcFolder / self.inputFile, 'wt') as f:
            write_espresso_in(f,
                              atoms=atoms, input_data=self.data,
                              pseudopotentials={s: p.name for s, p in self.pseudopotentials.items()},
                              kpts=kPoints,
                              crystal_coordinates=True)

        return ''

    def isConverged(self, calcFolder: Path):
        if not Path(calcFolder).joinpath(self.outputFile).exists():
            res = False
        else:
            with open(calcFolder/self.outputFile, 'rt') as out:
                res = 'JOB DONE' in out.read()
        if not res:
            logger.error('Quantum Espresso is not completely Done')
        return res

    def readOutput(self, system, calcFolder: str):
        calcFolder = Path(calcFolder)
        with open(calcFolder / 'pbc', 'rt') as f:
            pbc = tuple(int(c) for c in f.read().split())
        with open(calcFolder / self.outputFile) as f:
            atoms = next(read_espresso_out(f, index=slice(None, -2, -1)))
        atoms.set_pbc(pbc)
        results = atoms.get_calculator().results

        factory = system.getFactory()
        result = factory()

        if 'structure' in self.targetProperties:
            result.setProperty('structure', self.AtomicStructureRepresentation.fromAtoms(atoms), extension='atomistic')
        if 'enthalpy' in self.targetProperties:
            if 'enthalpy' in results:
                result.setProperty('enthalpy', results['energy'])
            else:
                result.setProperty('energy', results['energy'])
        if 'energy' in self.targetProperties:
            result.setProperty('energy', results['energy'])
        if 'forces' in self.targetProperties:
            result.setProperty('forces', results['forces'])

        with open(calcFolder/self.outputFile, 'rt') as f:
            content = f.readlines()
        if 'stressTensor' in self.targetProperties:
            result.setProperty('stressTensor', self.readStressTensor(content))
        return result

    def readStressTensor(self, content):
        stressTensor = np.zeros((3, 3), dtype=float)
        for i, line in enumerate(content):
            if 'total   stress' in line:
                for row in content[i + 1: i + 4]:
                    stressTensor[i, :] = np.array(row.split()[3: 6], dtype=float)
        return stressTensor
