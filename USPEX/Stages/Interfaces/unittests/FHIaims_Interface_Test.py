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


from ....components import AtomisticRepresentation, FHIaims_Interface


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
                              control=pj(SPECIFICPATH, 'aims_control_1'), kresol=0.14)


        for ID in range(10):
            structure = AtomisticRepresentation.readPOSCAR(pj(GATHEREDPATH, f'input/system{ID}.vasp'), (1, 1, 1))
            system = dict(
                ID=ID,
                structure=structure,
                disassembler=AtomisticRepresentation.atomicDisassemblerType(
                    np.arange(len(structure)).reshape((-1, 1))),
                externalPressure=0.0001
            )
            os.mkdir(WORKPATH)
            aims.prepareLocalCalculation(system, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = pj(GATHEREDPATH, 'output')
            shutil.copytree(pj(folder, f"CalcFold{system['ID']}"), WORKPATH)
            results = aims.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomisticRepresentation.readPOSCAR(pj(folder, f"system{system['ID']}.vasp"), (1, 1, 1))
            cell = results['structure'].getCell()
            cellRef = structureRef.getCell()
            self.assertTrue(np.allclose(cell.getCellVectors(),
                                        cellRef.getCellVectors()))
            self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(results['structure'].getCartesianCoordinates()),
                                        cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))

