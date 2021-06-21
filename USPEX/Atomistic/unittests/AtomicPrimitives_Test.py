import unittest

import os
from os.path import join as pj
from scipy.spatial.distance import cosine
import numpy as np

from ..AtomicPrimitives import AtomicStructure
from ...components import CrystalRepresentation

PATH_WITH_TESTS = os.path.dirname(os.path.abspath(__file__))


class GetPrincipalCell_Test(unittest.TestCase):

    def setUp(self) -> None:
        self.testFile = pj(PATH_WITH_TESTS, 'POSCARS/POSCAR_B36')
        self.test_pbc = (0, 1, 0)
        self.testStruct = CrystalRepresentation.readAtomicStructure(self.testFile)
        self.testStruct['cell'] = type()(self.testStruct['cell'].getCellVectors(), pbc=self.test_pbc)
        self.structure, _ = AtomicStructure.assemble(**self.testStruct)

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
        which_pbc = np.nonzero(self.test_pbc)[0][0]
        pbcvec = self.structure.cell.getCellVectorsPBC()[0]
        newvectors = self.structure.getPrincipalCell().getCellVectors()
        print('\npbc_index is ', which_pbc)
        for i, vec in enumerate(newvectors):
            print(f"cos(pbcvec, newvec[{i}]) = ", cosine(newvectors[i], pbcvec))
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
