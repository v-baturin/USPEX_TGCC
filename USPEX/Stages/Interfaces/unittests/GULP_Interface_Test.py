"""
@file        GULP_CalculatorTest.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        19 September 2016
@brief       Class for testing GULP_Calculator class.
"""

__author__ = 'asamtsevich'

import numpy as np
import shutil
import unittest
import filecmp

from pathlib import Path

from ....components import AtomisticRepresentation, GULP_Interface, AtomisticPoolEntry


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'gulpSpecific'
GATHEREDPATH = HOMEPATH/'gulpGatheredData'
WORKPATH = HOMEPATH/'Mg4Al8O16_gulp'


class GULP_CalculatorTest(unittest.TestCase):


    def test_life(self):

        gulp = GULP_Interface(tag='0', goptions=SPECIFICPATH/'goptions', ginput=SPECIFICPATH/'ginput_1')

        for ID in range(10):
            structure = AtomisticRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
            system = AtomisticPoolEntry(
                ID=ID,
                structure=structure,
                disassembler=AtomisticRepresentation.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1))),
                externalPressure=100
            )
            WORKPATH.mkdir(parents=True, exist_ok=True)
            gulp.prepareLocalCalculation(system, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            shutil.rmtree(WORKPATH)
            self.assertTrue(match)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{system['ID']}", WORKPATH)
            gulp.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomisticRepresentation.readPOSCAR(folder/f"system{system['ID']}.vasp", (1, 1, 1))
            structure = system.getAtomicStructure()
            cell = structure.getCell()
            cellRef = structureRef.getCell()
            self.assertTrue(np.allclose(cell.getCellVectors(),
                                        cellRef.getCellVectors()))
            # self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(results['structure'].getCartesianCoordinates()),
            #                             cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))


class GULP_InterfaceTest(unittest.TestCase):
    def test_read_output(self):
        ID = 0
        # HERE what is written in ginput and goption no make sense.
        # Only output will be parsed and properties checked
        interface = GULP_Interface(tag='1', ginput=HOMEPATH/'Specific'/'ginput_1',
                                   goptions= HOMEPATH/'Specific'/'goptions_1',
                                   targetProperties=['structure', 'enthalpy', 'stressTensor', 'strains'])
        # with open(GATHEREDPATH/f'input/system{ID}', 'rt') as f:
        #     system = {'ID': ID, 'structure': Crystal.fromJSON(f.read())}

        structure = AtomisticRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
        system = AtomisticPoolEntry(
            ID=ID,
            structure=structure,
            disassembler=AtomisticRepresentation.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1))),
            externalPressure=100,
            pbc=(1, 1, 1)
        )
        interface.readOutput(system=system, calcFolder=HOMEPATH/'gulp_test')
        self.assertTrue(np.isclose(system['enthalpy'], -645.80329121))
        stress_ref = np.array([[-99.960848, 0.394719, -0.211383], [0.394719,  -100.008335,  0.019149], [-0.211383,  0.019149,  -100.271584]])
        self.assertTrue(np.allclose(system['stressTensor'], stress_ref))
        strains_ref = np.array([0.012703, -0.031870, -0.039885, -0.000385, -0.008334, 0.182955])
        self.assertTrue(np.allclose(system['strains'], strains_ref))

    def test_read_energy(self):
         with open(HOMEPATH/'gulp_test'/'output_bad_1st_SCF', 'rt') as f:
             contents = f.readlines()
         ev = GULP_Interface.readEnergy(None, contents)
         self.assertAlmostEqual(ev, -1604.1694, delta=0.001)
