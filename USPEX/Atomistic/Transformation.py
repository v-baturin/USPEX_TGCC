"""
USPEX.Atomistic.Transformation
==============================
"""

import numpy as np
from scipy.spatial.transform import Rotation

from ..Semantics.Atomistic.Primitives.Transformation import Transformation as TransfromationSemantics

class Transformation(TransfromationSemantics):
    """
    Class defining 3D space transformation: rotation plus shift.
    """

    def __init__(self, rotMatrix, transVec):
        """
        :param rotMatrix: rotation matrix.
        :param transVec: translation vector.
        """
        self.rotMatrix = rotMatrix
        self.transVec = transVec

    @staticmethod
    def fromMatrix(rotMatrix, transVec):
        """
        Explicit constructor from matrix and shift.

        :param rotMatrix: rotation matrix.
        :param transVec: translation vector.

        :return: **Transformation** object.
        """
        return Transformation(rotMatrix, transVec)

    @staticmethod
    def fromRotVector(rotVec, transVec):
        """
        Alternative constructor from rotation vector and shift.

        :param rotVec: rotation vector.
        :param transVec: translation vector.

        :return: **Transformation** object.
        """
        rotation_sc = Rotation.from_rotvec(rotVec)
        rot_matrix = rotation_sc.as_matrix()
        return Transformation(rot_matrix, transVec)

    def __neg__(self):
        rotation = Rotation.from_matrix(self.rotMatrix)
        return Transformation.fromRotVector(-rotation.as_rotvec(), -rotation.inv().apply(self.transVec))

    def __mul__(self, other):
        return Transformation(np.dot(self.rotMatrix, other.rotMatrix), self.transVec + np.dot(self.rotMatrix, other.transVec))

    def transformCoordinates(self, coordinates):
        """
        Applies transformation to array of coordinates.

        :param coordinates: N*3 array of coordinates.

        :return: N*3 array of transformed coordinates.
        """
        return np.dot(coordinates, self.rotMatrix.T) + self.transVec

    def transformCell(self, cell):
        """
        Applies transformation to **Cell** object.

        :param cell: **Cell** object to be transformed.

        :return: transformed **Cell** object.
        """
        return type(cell)(np.dot(cell.getCellVectors(), self.rotMatrix.T), cell.getPBC())

    def transform(self, structure):
        """
        Applies transformation to atomic structure.

        :param structure: atomic structure object to be transformed.

        :return: transformed atomic structure.
        """
        coord = self.transformCoordinates(structure.getCartesianCoordinates())
        cell = self.transformCell(structure.getCell()) if structure.getCell() is not None else None
        return type(structure)(structure.getAtomTypes(), coord, cell=cell, edges=structure.edges)

    @staticmethod
    def randomRotVector():
        """
        Generates random rotation vector.

        :return: rotation vector.
        """
        return Rotation.random().as_rotvec()

