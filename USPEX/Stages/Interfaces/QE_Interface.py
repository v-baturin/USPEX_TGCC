"""
USPEX.Stages.Interfaces.QE_Interface
====================================

"""

import logging
import shutil
import numpy as np

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

    aseAdapterType = None

    @classmethod
    def registerTypes(cls, aseAdapterType):
        cls.aseAdapterType = aseAdapterType

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

    def prepareLocalCalculation(self, system: dict, calcFolder: Path):
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
            if pseudo.exists():
                shutil.copy(pseudo, calcFolder)

        ############################# KPOINTS #################################
        try:
            kPoints = self.kPoints.build(structure.getCell())
        except BadKPoints:
            # This LATTICE is extremely wrong, let's skip it from now
            logger.info('K-points cannot be built, so it\'s set as   [1, 1, 1]')
            kPoints = [1, 1, 1]

        system['ase'] = self.adapter.write(structure, system['disassembler'].fixedIndices,
                                           kPoints, self.pseudopotentials, calcFolder)

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

    def readOutput(self, system: dict, calcFolder: Path):
        with open(Path(calcFolder)/self.outputFile, 'rt') as f:
            aseStructure = next(read_espresso_out(f, index=slice(None, -2, -1)))
            f.seek(0)
    def readOutput(self, system: dict, calcFolder: str):
        calcFolder = Path(calcFolder)
        aseData = self.adapter.read(calcFolder, **system.pop('ase'))
        results = {}
        if 'structure' in self.targetProperties:
            results['structure'] = aseData['structure']
        if 'enthalpy' in self.targetProperties:
            results['enthalpy'] = aseData['results'].getEnthalpy(system['externalPressure'])
        if 'energy' in self.targetProperties:
            results['energy'] = aseData['results']['energy']
        if 'forces' in self.targetProperties:
            results['forces'] = aseData['results']['forces']

        with open(calcFolder/self.outputFile, 'rt') as f:
            content = f.readlines()
        if 'stressTensor' in self.targetProperties:
            results['stressTensor'] = self.readStressTensor(content)
        return results

    def readStressTensor(self, content):
        stressTensor = np.zeros((3, 3), dtype=float)
        for i, line in enumerate(content):
            if 'total   stress' in line:
                for row in content[i + 1: i + 4]:
                    stressTensor[i, :] = np.array(row.split()[3: 6], dtype=float)
        return stressTensor
