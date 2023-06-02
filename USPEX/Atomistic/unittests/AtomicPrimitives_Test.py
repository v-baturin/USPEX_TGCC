import unittest
import numpy as np

from pathlib import Path
from scipy.spatial.distance import cosine

from ..AtomicPrimitives import AtomicStructure, AtomicDisassembler
from ..CellUtility import Cell
from ...components import AtomisticRepresentation

PATH_WITH_TESTS = Path(__file__).parent


class GetPrincipalCell_Test(unittest.TestCase):

    def setUp(self) -> None:
        self.testFile = PATH_WITH_TESTS/'POSCARS/POSCAR_B36'
        self.test_pbc = (0, 1, 0)
        self.testStruct = AtomisticRepresentation.readAtomicStructure(self.testFile)
        self.testStruct['cell'] = Cell.initFromCellVectors(self.test_pbc, [self.testStruct['cell'].getCellVectors()[1]])
        self.structure, _ = AtomicDisassembler.assemble(**self.testStruct)

    def test_PBCorder_1d(self):
        which_pbc = np.nonzero(self.test_pbc)[0][0]
        pbcvec = self.structure.getCell().getCellVectorsPBC()[0]
        newvectors = self.structure.getRectifiedCell().getCellVectors()
        print('\npbc_index is ', which_pbc)
        for i, vec in enumerate(newvectors):
            print(f"cos(pbcvec, newvec[{i}]) = ", cosine(newvectors[i], pbcvec))
        self.assertAlmostEquals(cosine(newvectors[which_pbc], pbcvec), 0.)

class bad_principal_test(unittest.TestCase):

    def setUp(self) -> None:
        self.testFile = PATH_WITH_TESTS/'POSCARS/bad_cart2frac_POSCAR'
        self.test_pbc = (0, 1, 0)
        self.testStruct = AtomisticRepresentation.readAtomicStructure(self.testFile)
        self.testStruct['cell'] = Cell(self.testStruct['cell'].getCellVectors(), pbc=self.test_pbc)
        self.structure, _ = AtomicDisassembler.assemble(**self.testStruct, vacuumSize=1.0)

    def test_bad_principal(self):
        newCell = self.structure.getRectifiedCell()
        assert np.linalg.det(newCell.getCellVectors()) > 0


if __name__ == '__main__':
    unittest.main()
