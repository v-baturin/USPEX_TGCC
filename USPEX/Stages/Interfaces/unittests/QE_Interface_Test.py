import filecmp
import shutil
import unittest
import numpy as np
from pathlib import Path

from ....Optimizers.PoolEntry import PoolEntry, EntryFlavour
from ....components import AtomicStructureRepresentation, QE_Interface, Atomistic


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
        atomistic = Atomistic()
        extensions = dict(
            atomistic=atomistic.propertyExtension(atomistic)
        )


        for ID in range(10):
            structure = AtomicStructureRepresentation.readPOSCAR(GATHEREDPATH/f'input/system{ID}.vasp', (1, 1, 1))
            disassembler = Atomistic.atomicDisassemblerType(np.arange(len(structure)).reshape((-1, 1)))
            system = PoolEntry(ID, EntryFlavour(extensions=extensions))
            system.setProperty('externalPressure', 0.0001)
            system.setProperty('disassembler', disassembler, extension='atomistic', suffix='intermediate')
            system.setProperty('disassembler', disassembler, extension='atomistic', suffix='1')
            system.setProperty('structure', structure, extension='atomistic', suffix='intermediate')
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
            qe.readOutput(system, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomicStructureRepresentation.readPOSCAR(folder/f"system{system['ID']}.vasp", (1, 1, 1))
            structure = system.getProperty('structure', extension='atomistic', suffix='1')
            cell = structure.getCell()
            cellRef = structureRef.getCell()
            self.assertTrue(np.allclose(cell.getCellVectors(), cellRef.getCellVectors(), atol=1.0e-5))
            # self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(results['structure'].getCartesianCoordinates()),
            #                             cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))
