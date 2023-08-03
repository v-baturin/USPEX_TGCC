"""
@file        VASP_CalculatorTest.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        19 September 2016
@brief       Class for testing VASP_Calculator class.
"""

__author__ = 'asamtsevich'

import shutil
import unittest
import filecmp
import numpy as np

from pathlib import Path

from ....Optimizers.PoolEntry import PoolEntry
from ....components import AtomisticRepresentation, FHIaims_Interface, Atomistic


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'aimsSpecific'
GATHEREDPATH = HOMEPATH/'aimsGatheredData'
WORKPATH = HOMEPATH/'F2_aims'


class VASP_CalculatorTest2(unittest.TestCase):
    """
    Checking correct parsing properties
    """
    def test_life(self):
        aims = FHIaims_Interface(tag='1',
                                 control=SPECIFICPATH/'aims_control_1',
                                 kresol=0.14)
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )

        for ID in range(10):
            structure = AtomisticRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
            disassembler = AtomisticRepresentation.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
            system = PoolEntry(extensions=extensions, ID=ID)
            system.setProperty('externalPressure', 0.0001)
            system.setProperty('disassembler', disassembler, extension='atomistic', suffix='intermediate')
            system.setProperty('disassembler', disassembler, extension='atomistic', suffix='1')
            system.setProperty('structure', structure, extension='atomistic', suffix='intermediate')
            WORKPATH.mkdir(parents=True, exist_ok=True)
            aims.prepareLocalCalculation(system, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{system['ID']}", WORKPATH)
            aims.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomisticRepresentation.readPOSCAR(folder/f"system{system['ID']}.vasp", (1, 1, 1))
            structure = system.getProperty('structure', extension='atomistic', suffix='1')
            cell = structure.getCell()
            cellRef = structureRef.getCell()
            self.assertTrue(np.allclose(cell.getCellVectors(),
                                        cellRef.getCellVectors()))
            self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(structure.getCartesianCoordinates()),
                                        cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))

