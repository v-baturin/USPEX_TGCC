import unittest
import numpy as np

from pathlib import Path
from scipy.spatial.distance import cosine

from ...components import AtomicStructureRepresentation

PATH_WITH_TESTS = Path(__file__).parent


class GetPrincipalCell_Test(unittest.TestCase):

    def setUp(self) -> None:
        self.testFile = PATH_WITH_TESTS/'POSCARS/POSCAR_B36'
        self.test_pbc = (0, 1, 0)
        self.structure = AtomicStructureRepresentation.readPOSCAR(self.testFile, pbc=self.test_pbc)

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
        self.structure = AtomicStructureRepresentation.readPOSCAR(self.testFile, pbc=self.test_pbc)

    def test_bad_principal(self):
        newCell = self.structure.getRectifiedCell()
        assert np.linalg.det(newCell.getCellVectors()) > 0


if __name__ == '__main__':
    unittest.main()
