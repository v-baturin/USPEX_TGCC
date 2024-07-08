from .Generics import Vector3, VectorN3, Operation3


class Transformation:
    """
    Class defining 3D space transformation: rotation plus shift.
    """

    @staticmethod
    def fromMatrix(rotMatrix: Operation3, transVec: Vector3) -> 'Transformation':
        """
        Explicit constructor from matrix and shift.

        :param rotMatrix: rotation matrix.
        :param transVec: translation vector.

        :return: **Transformation** object.
        """
        pass

    @staticmethod
    def fromRotVector(rotVec: Vector3, transVec: VectorN3) -> 'Transformation':
        """
        Alternative constructor from rotation vector and shift.

        :param rotVec: rotation vector.
        :param transVec: translation vector.

        :return: **Transformation** object.
        """
        pass

    def __neg__(self) -> 'Transformation':
        pass

    def __mul__(self, other: 'Transformation') -> 'Transformation':
        pass

    def transformCoordinates(self, coordinates: VectorN3) -> VectorN3:
        """
        Applies transformation to array of coordinates.

        :param coordinates: N*3 array of coordinates.

        :return: N*3 array of transformed coordinates.
        """
        pass

    def transformCell(self, cell: 'Cell') -> 'Cell':
        """
        Applies transformation to **Cell** object.

        :param cell: **Cell** object to be transformed.

        :return: transformed **Cell** object.
        """
        pass

    def transform(self, structure: 'AtomicStructure') -> 'AtomicStructure':
        """
        Applies transformation to atomic structure.

        :param structure: atomic structure object to be transformed.

        :return: transformed atomic structure.
        """
        pass

    @staticmethod
    def randomRotVector() -> VectorN3:
        """
        Generates random rotation vector.

        :return: rotation vector.
        """
        pass

