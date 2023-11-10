import filecmp
import shutil
import unittest
import numpy as np
from pathlib import Path

from ..QE_Interface import QE_Interface
from ....Optimizers.PoolEntry import EntryFlavour
from ....Atomistic.Primitives.Element import Element
from ....Atomistic.Primitives.Cell import Cell
from ....Atomistic.Primitives.AtomicStructure import AtomicStructure
from ....IO.AtomicStructureRepresentation import AtomicStructureRepresentation
from ....Atomistic.Atomistic import Atomistic
Atomistic.registerTypes(AtomicStructure, Element, Cell, AtomicStructureRepresentation)
QE_Interface.registerTypes(AtomicStructureRepresentation)



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
            intermediate = dict()
            intermediate['atomistic.structure'] = structure
            intermediate['.externalPressure'] = 0.0001
            intermediate['atomistic.disassembler'] = disassembler
            intermediate = EntryFlavour(extensions=extensions, **intermediate)
            WORKPATH.mkdir(parents=True, exist_ok=True)
            qe.prepareLocalCalculation(intermediate, WORKPATH)
            folder = GATHEREDPATH/'input'/f"CalcFold{ID}"
            dcmp = filecmp.dircmp(folder, WORKPATH)
            match = not dcmp.diff_files
            for common_dir in dcmp.common_dirs:
                match = match and not dcmp.subdirs[common_dir].diff_files
            self.assertTrue(match)
            shutil.rmtree(WORKPATH)
            folder = GATHEREDPATH/'output'
            shutil.copytree(folder/f"CalcFold{ID}", WORKPATH)
            result = qe.readOutput(intermediate, WORKPATH)
            shutil.rmtree(WORKPATH)
            structureRef = AtomicStructureRepresentation.readPOSCAR(folder/f"system{ID}.vasp", (1, 1, 1))
            structure = result.getProperty('structure', extension='atomistic')
            cell = structure.getCell()
            cellRef = structureRef.getCell()
            self.assertTrue(np.allclose(cell.getCellVectors(), cellRef.getCellVectors(), atol=1.0e-5))
            # self.assertTrue(np.allclose(cell.getWrapedCartesianCoordinates(results['structure'].getCartesianCoordinates()),
            #                             cellRef.getWrapedCartesianCoordinates(structureRef.getCartesianCoordinates())))
