import unittest
import numpy as np

from ..CellUtility import CellUtility, Cell


class CellUtility3D_Test(unittest.TestCase):
    def setUp(self) -> None:
        self.cellUtility3D = CellUtility(dim = 3, cellParameters = {'a': 10.0, 'b': 12.0, 'c': 5.0,
                                                                    'alpha': 90.0, 'beta': 90.0, 'gamma': 90.0})
        self.cellUtility2D = CellUtility(dim = 2, cellParameters = {'a': 10.0, 'b': 12.0, 'alpha': 90.0},
                                         axis=(0,0,1), thickness=5.0)
        self.cellUtility1D = CellUtility(dim = 1, cellParameters = {'a': 10.0}, axis=(1,0,0), thickness=10.0)
        self.cellUtility0D = CellUtility(dim = 0, thickness=10.0)

    def test_getRandomCell(self):
        cell = self.cellUtility3D.getRandomCell(500.0, 36)
        self.assertTrue(self.cellUtility3D.isGoodCell(cell))
        cell = self.cellUtility2D.getRandomCell(500.0, 36)
        self.assertTrue(self.cellUtility2D.isGoodCell(cell))
        cell = self.cellUtility1D.getRandomCell(500.0, 36)
        self.assertTrue(self.cellUtility1D.isGoodCell(cell))
        cell = self.cellUtility0D.getRandomCell(500.0, 36)
        self.assertTrue(self.cellUtility0D.isGoodCell(cell))

    def test_adjustCell(self):
        adjustedCell = self.cellUtility3D.adjustCell(np.eye(3), 500.0, 36)
        self.assertTrue(self.cellUtility3D.isGoodCell(adjustedCell))
        adjustedCell = self.cellUtility2D.adjustCell(np.eye(3), 500.0, 36)
        self.assertTrue(self.cellUtility2D.isGoodCell(adjustedCell))
        adjustedCell = self.cellUtility1D.adjustCell(np.eye(3), 500.0, 36)
        self.assertTrue(self.cellUtility1D.isGoodCell(adjustedCell))
        adjustedCell = self.cellUtility0D.adjustCell(np.eye(3), 500.0, 36)
        self.assertTrue(self.cellUtility0D.isGoodCell(adjustedCell))

    def test_getHybridCell(self):
        cell1 = self.cellUtility3D.getRandomCell(500.0, 36)
        cell2 = self.cellUtility3D.getRandomCell(500.0, 36)
        hybridCell = self.cellUtility3D.getHybridCell(cell1, cell2, 0.5)
        self.assertTrue(self.cellUtility3D.isGoodCell(hybridCell))
        cell1 = self.cellUtility2D.getRandomCell(500.0, 36)
        cell2 = self.cellUtility2D.getRandomCell(500.0, 36)
        hybridCell = self.cellUtility2D.getHybridCell(cell1, cell2, 0.5)
        self.assertTrue(self.cellUtility2D.isGoodCell(hybridCell))
        cell1 = self.cellUtility1D.getRandomCell(500.0, 36)
        cell2 = self.cellUtility1D.getRandomCell(500.0, 36)
        hybridCell = self.cellUtility1D.getHybridCell(cell1, cell2, 0.5)
        self.assertTrue(self.cellUtility1D.isGoodCell(hybridCell))
        cell1 = self.cellUtility0D.getRandomCell(500.0, 36)
        cell2 = self.cellUtility0D.getRandomCell(500.0, 36)
        hybridCell = self.cellUtility0D.getHybridCell(cell1, cell2, 0.5)
        self.assertTrue(self.cellUtility0D.isGoodCell(hybridCell))

class Cell_Test(unittest.TestCase):

    def test_init3D(self):
        cellVectors = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        cellParameters = (2, 3, 4, 80, 120, 80)
        cell1 = Cell.initFromCellVectors((1, 1, 1), cellVectors)
        cell2 = Cell.initFromCellParameters((1, 1, 1), *cellParameters)
        self.assertTrue(np.allclose(cell1.getCellVectors(), cellVectors))
        self.assertTrue(np.allclose(cell2.getCellParameters(), cellParameters))

    def test_init2D(self):
        cellVectors = [[3.0, 0.0, 0.5], [1.0, 2.0, 0.2]]
        cellParameters = {'a': 2, 'b': 3, 'alpha': 80}
        cellVectorsRef1 = [[3.0, 0.0, 0.5], [1.0, 2.0, 0.2], [-0.16437678, -0.01643768,  0.98626065]]
        cellVectorsRef2 = [[1.0, 2.0, 0.2], [-0.16437678, -0.01643768,  0.98626065], [3.0, 0.0, 0.5]]
        cellVectorsRef3 = [[-0.16437678, -0.01643768,  0.98626065], [3.0, 0.0, 0.5], [1.0, 2.0, 0.2]]
        cellVectorsRef4 = [[2.0, 0.0, 0.0], [0.52094453, 2.95442326, 0.0], [0.0, 0.0, 1.0]]
        cellVectorsRef5 = [[0.52094453, 2.95442326, 0.0], [0.0, 0.0, 1.0], [2.0, 0.0, 0.0]]
        cellVectorsRef6 = [[0.0, 0.0, 1.0], [2.0, 0.0, 0.0],[0.52094453, 2.95442326, 0.0]]
        cellVectorsRef7 = [[2.0, 0.0, 0.0], [0.52094453, 0.0, -2.95442326], [0.0, 1.0, 0.0]]
        cellVectorsRef8 = [[0.52094453, 0.0, -2.95442326], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]]
        cellVectorsRef9 = [[0.0, 1.0, 0.0], [2.0, 0.0, 0.0],[0.52094453, 0.0, -2.95442326]]
        cell1 = Cell.initFromCellVectors((1, 1, 0), cellVectors)
        cell2 = Cell.initFromCellVectors((1, 0, 1), cellVectors)
        cell3 = Cell.initFromCellVectors((0, 1, 1), cellVectors)
        cell4 = Cell.initFromCellParameters((1, 1, 0), **cellParameters, axis=(0,0,1))
        cell5 = Cell.initFromCellParameters((1, 0, 1), **cellParameters, axis=(0,0,1))
        cell6 = Cell.initFromCellParameters((0, 1, 1), **cellParameters, axis=(0,0,1))
        cell7 = Cell.initFromCellParameters((1, 1, 0), **cellParameters, axis=(0,1,0))
        cell8 = Cell.initFromCellParameters((1, 0, 1), **cellParameters, axis=(0,1,0))
        cell9 = Cell.initFromCellParameters((0, 1, 1), **cellParameters, axis=(0,1,0))
        self.assertTrue(np.allclose(cell1.getCellVectors(), cellVectorsRef1))
        self.assertTrue(np.allclose(cell2.getCellVectors(), cellVectorsRef2))
        self.assertTrue(np.allclose(cell3.getCellVectors(), cellVectorsRef3))
        self.assertTrue(np.allclose(cell4.getCellVectors(), cellVectorsRef4))
        self.assertTrue(np.allclose(cell5.getCellVectors(), cellVectorsRef5))
        self.assertTrue(np.allclose(cell6.getCellVectors(), cellVectorsRef6))
        self.assertTrue(np.allclose(cell7.getCellVectors(), cellVectorsRef7))
        self.assertTrue(np.allclose(cell8.getCellVectors(), cellVectorsRef8))
        self.assertTrue(np.allclose(cell9.getCellVectors(), cellVectorsRef9))

    def test_init1D(self):
        cellVectors = [[2.0, 0.0, 0.0]]
        cellVectorsRef1 = [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        cellVectorsRef2 = [[0.0, -1.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        cellVectorsRef3 = [[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]]
        cellVectorsRef4 = [[0.0, 0.0, 2.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
        cellVectorsRef5 = [[1.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, -1.0, 0.0]]
        cellVectorsRef6 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 2.0]]
        cell1 = Cell.initFromCellVectors((1, 0, 0), cellVectors)
        cell2 = Cell.initFromCellVectors((0, 1, 0), cellVectors)
        cell3 = Cell.initFromCellVectors((0, 0, 1), cellVectors)
        cell4 = Cell.initFromCellParameters((1, 0, 0), a=2, axis=(0,0,1))
        cell5 = Cell.initFromCellParameters((0, 1, 0), a=2, axis=(0,0,1))
        cell6 = Cell.initFromCellParameters((0, 0, 1), a=2, axis=(0,0,1))
        self.assertTrue(np.allclose(cell1.getCellVectors(), cellVectorsRef1))
        self.assertTrue(np.allclose(cell2.getCellVectors(), cellVectorsRef2))
        self.assertTrue(np.allclose(cell3.getCellVectors(), cellVectorsRef3))
        self.assertTrue(np.allclose(cell4.getCellVectors(), cellVectorsRef4))
        self.assertTrue(np.allclose(cell5.getCellVectors(), cellVectorsRef5))
        self.assertTrue(np.allclose(cell6.getCellVectors(), cellVectorsRef6))

    def test_init0D(self):
        cellVectorsRef = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        cell1 = Cell.initFromCellVectors((0,0,0))
        cell2 = Cell.initFromCellParameters((0, 0, 0))
        self.assertTrue(np.allclose(cell1.getCellVectors(), cellVectorsRef))
        self.assertTrue(np.allclose(cell2.getCellVectors(), cellVectorsRef))


if __name__ == '__main__':
    unittest.main()
