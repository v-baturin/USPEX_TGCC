import numpy as np
from scipy.spatial.transform import Rotation


class Transformation:

    def __init__(self, rotMatrix, transVec):
        self.rotMatrix = rotMatrix
        self.transVec = transVec

    def __neg__(self):
        return Transformation.fromRotVector(-Rotation.from_matrix(self.rotMatrix).as_rotvec(), -self.transVec)

    def __mul__(self, other):
        return Transformation(np.dot(self.rotMatrix, other.rotMatrix), self.transVec + np.dot(self.rotMatrix, other.transVec))

    def getTransformedCoordinates(self, coordinates):
        return np.moveaxis(np.dot(self.rotMatrix, np.moveaxis(coordinates, -1, 0)), 0, -1) + self.transVec

    def transformCell(self, cell):
        return type(cell)(self.getTransformedCoordinates(cell.getCellVectors()), cell.getPBC())

    def transform(self, structure):
        coord = self.getTransformedCoordinates(structure.getCartesianCoordinates())
        cell = self.transformCell(structure.getCell()) if structure.getCell() is not None else None
        return type(structure)(structure.getAtomTypes(), coord, cell = cell, zmatrixConfig = structure.getZmatrixConfig())

    @staticmethod
    def fromMatrix(rotMatrix, transVec):
        return Transformation(rotMatrix, transVec)

    @staticmethod
    def fromRotVector(rotVec, transVec):
        rotation_sc = Rotation.from_rotvec(rotVec)
        rot_matrix = rotation_sc.as_matrix()
        return Transformation(rot_matrix, transVec)

    @staticmethod
    def randomRotVector():
        return Rotation.random().as_rotvec()

