import numpy as np
from scipy.spatial.transform import Rotation


class Transformation:
    # not finished yet
    # Is represented in direct coordinates.
    # Can be imported from fractional when created, and exported to it while applied

    def __init__(self, rotationMatrix, transitionVector, rotationVector, rotationAngle):
        self.rotationMatrix = rotationMatrix
        self.transitVector = transitionVector
        self.rotationVector = rotationVector
        self.rotationAxe = rotationAngle

    def fractional_copy(self, cell):
        # ??
        return Transformation.initFromMatrix(cell * self.rotationMatrix * cell ** -1, self.rotationVector * cell)

    def transition(self, cluster_coord):
        return cluster_coord + self.transitVector

    def rotation(self, cluster_coord, cell, coordinates='direct'):
        pass

    @staticmethod
    def move_origin(cluster_coord, cell, coordinates='direct', destination='center'):
        pass

    def transform(self, cluster_coord, cell):
        pass

    @staticmethod
    def initFromMatrix(rotationMatrix, transitionVector, coordinates='direct', cellVectors=[]):
        # coordinates = 'direct' or 'fractional'
        if coordinates == 'direct':
            rotation_sc = Rotation.from_matrix(rotationMatrix)
            rot_vect_angle = rotation_sc.as_rotvec()
            rotationAngle = np.dot(rot_vect_angle, rot_vect_angle) ** 0.5
            rotationVector = rot_vect_angle / rotationAngle
            return Transformation(rotationMatrix, transitionVector, rotationVector, rotationAngle)

    @staticmethod
    def initFromAngle(rotationVector, rotationAxe, transitionVector, coordinates='direct'):
        pass
