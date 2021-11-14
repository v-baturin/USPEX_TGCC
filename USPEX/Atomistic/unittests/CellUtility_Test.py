import unittest
import numpy as np

from ..CellUtility import CellUtility


class CellUtility3D_Test(unittest.TestCase):
    def setUp(self) -> None:
        self.cellUtility = CellUtility(dim = 3, cellParameters = {'a': 10.0, 'b': 12.0, 'c': 5.0,
                                                                  'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0})

    def test_getRandomCell(self):
        cell = self.cellUtility.getRandomCell({'Mo': 36}, 0)
        self.assertTrue(self.cellUtility.isGoodCell(cell))

    def test_adjustCell(self):
        adjustedCell = self.cellUtility.adjustCell(np.eye(3), {'Mo': 36}, 0)
        self.assertTrue(self.cellUtility.isGoodCell(adjustedCell))

    def test_getHybridCell(self):
        cell1 = self.cellUtility.getRandomCell({'Mo': 36}, 0)
        cell2 = self.cellUtility.getRandomCell({'Mo': 36}, 0)
        hybridCell = self.cellUtility.getHybridCell(cell1, cell2, 0.5)
        self.assertTrue(self.cellUtility.isGoodCell(hybridCell))

if __name__ == '__main__':
    unittest.main()
