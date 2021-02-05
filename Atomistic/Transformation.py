import numpy as np
from scipy.spatial.transform import Rotation


class Transformation:

    def __init__(self, rot_matrix, trans_vec, rot_vec):
        self.rotMatrix = rot_matrix
        self.transVec = trans_vec
        self.rotVec = rot_vec

    def __neg__(self):
        return Transformation(-self.rotMatrix, -self.transVec, -self.rotVec)

    def _transition(self, struc_coord):
        return struc_coord + self.transVec

    def _rotation(self, struc_coord):
        return np.dot(self.rotMatrix, struc_coord)

    def transform(self, struc_coord):
        struc_coord_transformed = self._rotation(struc_coord)
        struc_coord_transformed = self._transition(struc_coord_transformed)
        return struc_coord_transformed

    @staticmethod
    def fromMatrix(rot_matrix, trans_vec):
        rotation_sc = Rotation.from_matrix(rot_matrix)
        rot_vec = rotation_sc.as_rotvec()
        return Transformation(rot_matrix, trans_vec, rot_vec)

    @staticmethod
    def fromRotVector(rot_vec, trans_vec):
        rotation_sc = Rotation.from_rotvec(rot_vec)
        rot_matrix = rotation_sc.as_matrix()
        return Transformation(rot_matrix, trans_vec, rot_vec)

    @staticmethod
    def randomRotVector():
        pass
