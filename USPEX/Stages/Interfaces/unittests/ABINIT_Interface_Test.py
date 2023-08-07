"""
@file        ABINIT_Interface_Test.py
@author:     Michele Galasso
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    m.galasso@yandex.com
@date        8 June 2020
@brief       Class for testing ABINIT_Interface class.
"""

import shutil
import unittest
import filecmp
import numpy as np

from pathlib import Path

from ....Optimizers.PoolEntry import PoolEntry, EntryFlavour
from ....components import AtomisticRepresentation, ABINIT_Interface, Atomistic


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'abinitSpecific'
GATHEREDPATH = HOMEPATH/'abinitGatheredData'
WORKPATH = HOMEPATH/'Eu2H18_abinit'

try:
    from abipy import abilab
except Exception:
    pass
else:
    class ABINIT_Interface_Test(unittest.TestCase):
        """
        Checking correct parsing properties
        """

        def test_life(self):
            abinit = ABINIT_Interface(tag='0',
                                      in_file=SPECIFICPATH/'abinit.in_1',
                                      kresol=0.13,
                                      pp_files=[SPECIFICPATH/'H.psp8', SPECIFICPATH/'Eu.psp8'])
            atomistic = Atomistic()
            extensions = dict(
                atomistic=atomistic.propertyExtension(atomistic)
            )

            for ID in range(10):
                structure = AtomisticRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
                disassembler = AtomisticRepresentation.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
                system = PoolEntry(ID, EntryFlavour(extensions=extensions))
                system.setProperty('externalPressure', 130)
                system.setProperty('disassembler', disassembler, extension='atomistic', suffix='intermediate')
                system.setProperty('disassembler', disassembler, extension='atomistic', suffix='0')
                system.setProperty('structure', structure, extension='atomistic', suffix='intermediate')
                WORKPATH.mkdir(parents=True, exist_ok=True)
                abinit.prepareLocalCalculation(system, WORKPATH)
                folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
                dcmp = filecmp.dircmp(folder, WORKPATH)
                match = not dcmp.diff_files
                for common_dir in dcmp.common_dirs:
                    match = match and not dcmp.subdirs[common_dir].diff_files
                shutil.rmtree(WORKPATH)
                self.assertTrue(match)
                folder = GATHEREDPATH/'output'
                shutil.copytree(folder/f"CalcFold{system['ID']}", WORKPATH)
                abinit.readOutput(system, WORKPATH)
                shutil.rmtree(WORKPATH)
                structureRef = AtomisticRepresentation.readPOSCAR(folder/f"system{system['ID']}.vasp", (1, 1, 1))
                cell = system['atomistic.structure.0'].getCell()
                cellRef = structureRef.getCell()
                self.assertTrue(np.allclose(cell.getCellVectors(),
                                            cellRef.getCellVectors()))
                # self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(results['structure'].getCartesianCoordinates()),
                #                             cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))
