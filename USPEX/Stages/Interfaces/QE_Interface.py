"""
USPEX.Stages.Interfaces.QE_Interface
====================================

"""

import logging
import shutil
import numpy as np

from pathlib import Path

from .KPoints import KPoints, BadKPoints
from ...Presets import udateSystemWithPrefix as usp

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

        self.adapter = self.aseAdapterType(self.options)
        self.targetProperties = targetProperties if targetProperties is not None else ['structure', 'enthalpy']

    def prepareLocalCalculation(self, system: dict, calcFolder: str):
        calcFolder = Path(calcFolder)

        structure = system['structure']

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

    def isConverged(self, calcFolder: str):
        calcFolder = Path(calcFolder)
        if not calcFolder.joinpath(self.outputFile).exists():
            res = False
        else:
            with open(calcFolder/self.outputFile, 'rt') as out:
                res = 'JOB DONE' in out.read()
        if not res:
            logger.error('Quantum Espresso is not completely Done')
        return res

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
