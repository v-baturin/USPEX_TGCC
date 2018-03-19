import unittest
import numpy as np

from ..mol.zmatrix2coord import zmatrix2coord
from ..mol.coord2Zmatrix import coord2Zmatrix

class ZmatrixTest(unittest.TestCase):
    def test_both(self):
        coords = np.array([[-0.7906,  0.6960, -1.4100],
                           [-0.3037,  0.0847, -0.8378],
                           [-0.7356, -0.7818, -0.8432],
                           [-0.3166,  0.6117,  0.5423],
                           [ 1.1160, -0.0676, -1.3642],
                           [ 0.1265,  0.0559,  1.0769],
                           [-1.1559,  0.6904,  0.8228],
                           [ 0.0769,  1.4126,  0.5570],
                           [ 1.2554, -0.6507,  2.4647],
                           [ 2.0274,  0.5129, -0.7257]])
        format = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [2, 1, 0],
                           [2, 1, 3],
                           [2, 1, 3],
                           [4, 2, 1],
                           [4, 2, 6],
                           [4, 2, 6],
                           [5, 2, 1],
                           [5, 2, 9]])
        zmatrix = coord2Zmatrix(coords,format)
        coords_new = zmatrix2coord(zmatrix,format)
        self.assertTrue(np.allclose(coords,coords_new,rtol=1.e-3))
