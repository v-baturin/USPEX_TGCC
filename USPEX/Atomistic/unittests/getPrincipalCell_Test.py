import unittest

from ase.io import read as ase_read
from ase.io import write as ase_write
from scipy.spatial.distance import cosine
import numpy as np

from ..AtomicPrimitives import AtomicStructure
from ..CellUtility import Cell
from ...components import CrystalRepresentation

testFile = 'USPEX/Atomistic/unittests/POSCARS/POSCAR_B36'

test_pbc = (0, 1, 0)
testStruct = CrystalRepresentation.readAtomicStructure(testFile)
testStruct['cell'] = Cell(testStruct['cell'].getCellVectors(), pbc=test_pbc)
structure, disassembler = AtomicStructure.assemble(**testStruct)

class getPrincipalCell_Test(unittest.TestCase):

    # def test_Projection_1d(self):
    #     orthog = structure.getPrincipalCell(testno=1)
    #     structure.coordinates = orthog
    #     testStruct['molecules'] = [structure]
    #     testStruct['ID'] = 10
    #     CrystalRepresentation.writeAtomicStructure(testFile + '_out', testStruct)
    #     repere = orthog[0]
    #     n_at = len(orthog)
    #     for i in range(1, int(n_at/2)):
    #         u = orthog[i] - repere
    #         v = orthog[n_at-i] - repere
    #         if cosine(u, v) != 0:
    #             cos_dist = cosine(np.cross(u,v), structure.cell.getCellVectors()[-1])
    #             print(cos_dist)
    #             self.assertTrue(cos_dist < 0.01 or cos_dist > 1.99)

    def test_PBCorder_1d(self):
        which_pbc = np.nonzero(test_pbc)[0][0]
        pbcvec = structure.cell.getCellVectorsPBC()[0]
        newvectors = structure.getPrincipalCell().getCellVectors()
        print('pbc_index is ', which_pbc)
        for i, vec in enumerate(newvectors):
            print(f"pbcvec * newvec[{i}] = ", cosine(newvectors[i], pbcvec))
        self.assertAlmostEquals(cosine(newvectors[which_pbc], pbcvec), 0.)

        # initvectors = structure.cell.getCellVectors()
        # cosines = np.zeros((3,3))
        # for i in range(3):
        #     for j in range(3):
        #         cosines[i,j] = cosine(vectors[i], initvectors[j])
        # print(cosines)
        # print('hello')


if __name__ == '__main__':
    unittest.main()
