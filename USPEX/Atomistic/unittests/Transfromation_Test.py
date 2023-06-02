import unittest
import numpy as np

from ..Transformation import Transformation
from ..CellUtility import Cell

class Transformation_Test(unittest.TestCase):
    def test_transformCell1(self):
        cell = Cell(np.eye(3), pbc=(1,1,1))
        transformation = Transformation.fromRotVector([1.,0.,0.], [1.,0.,0.])
        modifiedCell = transformation.transformCell(cell)

        refCellVectors = np.array([[ 1.,  0.        ,  0.        ],
                                   [ 0.,  0.54030231,  0.84147098],
                                   [ 0., -0.84147098,  0.54030231]])
        self.assertTrue(np.allclose(modifiedCell.getCellVectors(), refCellVectors))

    def test_transformCell2(self):
        cell = Cell(np.eye(3), pbc=(1,1,1))
        rotVect = 2*np.pi/(3*np.sqrt(3)) * np.array([1.,1.,1.])
        transformation = Transformation.fromRotVector(rotVect, [1.,0.,0.])
        modifiedCell = transformation.transformCell(cell)

        refCellVectors = np.array([[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
        self.assertTrue(np.allclose(modifiedCell.getCellVectors(), refCellVectors))

    def test_multiplication1(self):
        cell = Cell(np.eye(3), pbc=(1,1,1))

        rotVect = 2*np.pi/(3*np.sqrt(3)) * np.array([1.,1.,1.])
        transformation1 = Transformation.fromRotVector(rotVect, [1.,0.,0.])
        transformation2 = Transformation.fromRotVector([1.,0.,0.], [1.,0.,0.])
        transformationProduct = transformation2*transformation1
        modifiedCell = transformationProduct.transformCell(cell)

        refCellVectors = np.array([[0.,  0.54030231, 0.84147098],
                                   [0., -0.84147098, 0.54030231],
                                   [1.,  0.        , 0.        ]])
        self.assertTrue(np.allclose(modifiedCell.getCellVectors(), refCellVectors))

    def test_multiplication2(self):
        coordinates = np.array([1.,0.,0.])

        rotVect = 2*np.pi/(3*np.sqrt(3)) * np.array([1.,1.,1.])
        transformation1 = Transformation.fromRotVector(rotVect, [1.,0.,0.])
        transformation2 = Transformation.fromRotVector([1.,0.,0.], [1.,0.,0.])
        transformationProduct = transformation2*transformation1
        modifiedCoordinates = transformationProduct.transformCoordinates(coordinates)

        refCoordinates = np.array([2., 0.54030231, 0.84147098])
        self.assertTrue(np.allclose(modifiedCoordinates, refCoordinates))

    def test_negate(self):
        transformation1 = Transformation.fromRotVector(Transformation.randomRotVector(), np.random.random(3))
        transformation2 = -transformation1
        transformationProduct = transformation2 * transformation1

        self.assertTrue(np.allclose(transformationProduct.rotMatrix, np.eye(3)))
        self.assertTrue(np.allclose(transformationProduct.transVec, np.array([0.,0.,0.])))


if __name__ == '__main__':
    unittest.main()
