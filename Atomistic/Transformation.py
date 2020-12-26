import numpy as np
from scipy.spatial.transform import Rotation


class Transformation:
    # not finished yet
    # Is represented in direct coordinates.

    def __init__(self, rot_matrix, trans_vec, rot_vec, rot_angle):
        self.rot_matrix = rot_matrix
        self.trans_vec = trans_vec
        self.rot_vec = rot_vec
        self.rot_angle = rot_angle

    '''def fractional_copy(self, cell):
        # ??
        return Transformation.initFromMatrix(cell * self.rot_matrix * cell ** -1, self.rot_vec * cell)'''

    def transition(self, struc_coord, cell):
        return struc_coord + cell * self.trans_vec

    def rotation(self, struc_coord, cell):
        return cell * self.rot_matrix * (cell ** -1) * struc_coord

    @staticmethod
    def move_origin(struc_coord, direction='to_center'):
        if direction == 'to_center':
            return struc_coord - np.array([0.5, 0.5, 0.5])
        elif direction == 'to_corner':
            return struc_coord + np.array([0.5, 0.5, 0.5])

    def transform(self, struc_coord, cell):
        # methods, that "know" about cell: transform, transition, rotation
        # struc_coord is fractional (no direct yet)
        struc_coord_transformed = Transformation.move_origin(struc_coord, direction='to_center')
        struc_coord_transformed = self.rotation(struc_coord_transformed, cell)
        struc_coord_transformed = self.transition(struc_coord_transformed, cell)
        struc_coord_transformed = Transformation.move_origin(struc_coord_transformed, direction='to_corner')
        return struc_coord_transformed

    @staticmethod
    def from_matrix(rot_matrix, trans_vec):
        # Transformation attributes are in 'direct' format (only!)
        rotation_sc = Rotation.from_matrix(rot_matrix)
        rot_vect_angle = rotation_sc.as_rotvec()
        rot_angle = np.dot(rot_vect_angle, rot_vect_angle) ** 0.5
        rot_vec = rot_vect_angle / rot_angle
        return Transformation(rot_matrix, trans_vec, rot_vec, rot_angle)

    @staticmethod
    def from_angle(rot_vec, rot_angle, trans_vec):
        rotation_sc = Rotation.from_rotvec(rot_angle * rot_vec)
        rot_matrix = rotation_sc.as_matrix()
        return Transformation(rot_matrix, trans_vec, rot_vec, rot_angle)
