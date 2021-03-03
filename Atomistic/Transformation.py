import numpy as np
from scipy.spatial.transform import Rotation


class Transformation:

    def __init__(self, rotMatrix, transVec, rotVec):
        self.rotMatrix = rotMatrix
        self.transVec = transVec
        self.rotVec = rotVec

    def __neg__(self):
        return Transformation(-self.rotMatrix, -self.transVec, -self.rotVec)

    def composition(self, transformation):
        pass

    def getTransformedCoordinates(self, coordinates):
        return np.dot(self.rotMatrix, coordinates) + self.transVec

    def transform(self, structure):
        coord = self.getTransformedCoordinates(structure.getCartesianCoordinates())
        return type(structure)(structure.getAtomTypes(), coord, cell = structure.getCell())

    @staticmethod
    def fromMatrix(rotMatrix, transVec):
        rotation_sc = Rotation.from_matrix(rotMatrix)
        rot_vec = rotation_sc.as_rotvec()
        return Transformation(rotMatrix, transVec, rot_vec)

    @staticmethod
    def fromRotVector(rotVec, transVec):
        rotation_sc = Rotation.from_rotvec(rotVec)
        rot_matrix = rotation_sc.as_matrix()
        return Transformation(rot_matrix, transVec, rotVec)

    @staticmethod
    def randomRotVector():
        pass
