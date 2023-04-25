import filecmp
import shutil
import unittest
import numpy as np
from pathlib import Path

from ....components import AtomisticRepresentation, QE_Interface


HOMEPATH = Path(__file__).parent
SPECIFICPATH = HOMEPATH/'qeSpecific'
GATHEREDPATH = HOMEPATH/'qeGatheredData'
WORKPATH = HOMEPATH/'Ca4F8_qe'


class QE_CalculatorTest2(unittest.TestCase):
    """
    Checking correct parsing properties
    """
    def test_life(self):
        qe = QE_Interface(tag='1', kresol=0.16, options=SPECIFICPATH/'qEspresso_options_1',
                          pseudopotentials={'C': SPECIFICPATH/'C.pbe-van_bm.upf'})

        for ID in range(10):
            structure = AtomisticRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
            system = dict(
                ID=ID,
                structure=structure,
                disassembler=AtomisticRepresentation.atomicDisassemblerType(
                    np.arange(len(structure)).reshape((-1, 1))),
                externalPressure=0.0001
            )
            WORKPATH.mkdir(parents=True, exist_ok=True)
            qe.prepareLocalCalculation(system, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{system['ID']}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            self.assertTrue(match)
            shutil.rmtree(WORKPATH)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{system['ID']}", WORKPATH)
            results = qe.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomisticRepresentation.readPOSCAR(folder/f"system{system['ID']}.vasp", (1, 1, 1))
            cell = results['structure'].getCell()
            cellRef = structureRef.getCell()
            self.assertTrue(np.allclose(cell.getCellVectors(), cellRef.getCellVectors(), atol=1.0e-5))
            # self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(results['structure'].getCartesianCoordinates()),
            #                             cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))
