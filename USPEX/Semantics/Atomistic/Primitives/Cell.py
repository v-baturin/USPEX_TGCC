from abc import ABC, abstractmethod
from .Generics import VectorN3, Vector3, Operation3, Operation4
from .Transformation import Transformation

class Cell(ABC):
    """
    Class representing unit cell of atomic structure.
    """

    @staticmethod
    @abstractmethod
    def initFromCellVectors(pbc: tuple[bool, bool, bool], cellVectors: VectorN3 = None) -> 'Cell':
        """
        Alternative constructor using cell vectors only for periodic directions.

        :param pbc: periodic boundary conditions in each direction.
        :param cellVectors: cell vectors only for periodic directions.
        :return: **Cell** object with appropriate parameters.
        """
        pass

    @staticmethod
    @abstractmethod
    def initFromCellParameters(pbc: tuple[bool, bool, bool], *args,
                               a: float = None, b: float = None, c: float = None,
                               alpha: float = None, beta: float = None, gamma: float = None,
                               axis: Vector3 = None) -> 'Cell':
        """
        Alternative constructor using cell parameters.

        :param pbc: periodic boundary conditions in each direction.
        :param a: parameter a of the cell.
        :param b: parameter b of the cell.
        :param c: parameter c of the cell.
        :param alpha: parameter alpha of the cell.
        :param beta: parameter beta of the cell.
        :param gamma: parameter gamma of the cell.
        :param axis: periodic axis in 1D, non-periodic axis in 2D, *None* otherwise.

        :return: **Cell** object with appropriate parameters.
        """
        pass

    @abstractmethod
    def getOrthogonallyTransformedCell(self, targetCell: 'Cell') -> 'Cell':
        """
        Transforms cellVectors closely to the cellVectors of the targetCell
        within orthogonal transformation.

        :param targetCell: targen cell to be aligned with.

        :return: **Cell** object with adjusted parameters.
        """
        pass

    @abstractmethod
    def getAlignedCell(self, axis: Vector3) -> 'Cell':
        """
        Creates cell with same parameters with given one but aligned along given axis.
        :param axis: peropdic axis (1D) or non-periodic axis (2D).
        :raises RuntimeError: if used on 0D or 3D structure.
        :return: Cell with lattice vectors with non-periodic (for 2D) or periodic (1D) aligned along axis.
        """
        pass

    @abstractmethod
    def getIntrinsicCell(self, coordinates: VectorN3) -> 'Cell':
        """
        :return: **Cell** object depending on dimensionality.

            0d: Cell made of unit principal eigenvectors

            1d: Keep periodic vector from original cell, The other two are perpendicular to it,
            directed along principal directions of
            a structure, flatten along periodic vector

            2d: Keep periodic vectors from original cell. The third is a unity vector perpendicular to those two.

            3d: Returns original Cell
        """
        pass

    @abstractmethod
    def getEnvelopeCell(self, coordinates: VectorN3 = None, vacuumSize: float = 0.0, intrinsic: bool = False) -> 'Cell':
        """
        :param coordinates: cartesian atomic coordinates
        :param vacuumSize: vacuum distance added along cell vector

        :return:  new cell object, corresponding to
        """
        pass

    @abstractmethod
    def getOptimizedCell(self) -> 'Cell':
        """
        Idea is as follows - if any lattice vector has projection onto any other lattice vector
        that is greater than half of length of this vector, we can reoptimize the shape
        i. e. if |a*b|/|b| > |b|/2 then a_new = a - ceil(|a*b|/|b|^2)*sign(a*b)*b

        :return: **Cell** object with optimized lattice vectors.
        """
        pass

    @abstractmethod
    def getPBC(self) -> tuple[bool, bool, bool]:
        """
        :return: periodic boundary conditions in each direction.
        """
        pass

    @abstractmethod
    def getAntiPBC(self) -> tuple[bool, bool, bool]:
        """
        :return: logic not of periodic boundary conditions in each direction.
        """
        pass

    @abstractmethod
    def getCellVectors(self) -> VectorN3:
        """
        :return: 3*3 array of cell vectors.
        """
        pass

    def getCellVectorsPBC(self) -> VectorN3:
        """
        :return: x*3 array of periodic cell vectors, where 0 <= x <= 3.
        """
        pass

    @abstractmethod
    def getCellVectorsAntiPBC(self) -> VectorN3:
        """
        :return: x*3 array of non-periodic cell vectors, where 0 <= x <= 3.
        """
        pass

    @abstractmethod
    def getCellParameters(self) -> tuple:  # TODO bad semantics
        """
        :return:  tuple of cell parameters: a, b, c, alpha, beta, gamma
        """
        pass

    @abstractmethod
    def __eq__(self, other) -> bool:
        pass

    @abstractmethod
    def getVolume(self) -> float:
        """
        :return: unit cell volume if cell is 3D periodic.
        """
        pass

    @abstractmethod
    def getArea(self) -> float:
        """
        :return: unit cell area if cell is 2D periodic or area of enveloping volume on 1D.
        """
        pass

    @abstractmethod
    def getLength(self) -> float:
        """
        :return: unit cell length if cell is 1D periodic or height of enveloping volume in 2D.
        """
        pass

    @abstractmethod
    def getRadius(self) -> float:
        """
        :return: unit cell envelope radius if cell is 1D periodic or non-periodic (0D).
        """
        pass

    @abstractmethod
    def getCornersCoordinates(self) -> VectorN3:
        """
        :return: 8*3 array of absolute coordinates of each corner of unit cell.
        """
        pass

    @abstractmethod
    def getMaxNumSlabs(self, axis: int, N: int) -> int:
        """
        Calculates maximal number of choices of slabs origins in given direction.
        :param axis: direction axis.
        :param N: number of atoms/molecules in cell.
        :return: maximal number of choices of slabs origins.
        """
        pass

    @abstractmethod
    def cartesianToFractional(self, coordinates: VectorN3) -> VectorN3:
        """
        Converts cartesian coordinates to fractional.

        :param coordinates: N*3 array of cartesian coordinates

        :return: N*3 array of fractional coordinates.
        """
        pass

    @abstractmethod
    def fractionalToCartesian(self, coordinates: VectorN3) -> VectorN3:
        """
        Converts fractional coordinates to cartesian.

        :param coordinates: N*3 array of fractional coordinates.

        :return: N*3 array of cartesian coordinates
        """
        pass

    @abstractmethod
    def fractionalToCartesianOperator(self, operator: Operation4) -> 'Transformation':
        """
        Сonvert some operator matrix defined in fractional space to transformation object in cartesian space.

        :param operator: 4*4 matrix defining operator in fractional space.

        :return:
        **Transformation** object defining transformation in cartesian space.
        """
        pass

    @abstractmethod
    def getWrapedCartesianCoordinates(self, coordinates: VectorN3) -> VectorN3:
        """
        Translates all coordinates to their image within unit cell via cell vectors in cartesian space.

        :param coordinates: cartesian coordinates

        :return: cartesian coordinates wrapped to unit cell.
        """
        pass

    @abstractmethod
    def getWrapedFractionalCoordinates(self, coordinates: VectorN3) -> VectorN3:
        """
        Translates all coordinates to their image within unit cell via cell vectors in fractional space.

        :param coordinates: fractional coordinates

        :return: fractional coordinates wrapped to unit cell.
        """
        pass

    @abstractmethod
    def center(self, coordinates: VectorN3, affectedDims=None) -> VectorN3:
        """
        Center atoms in unit cell.

        Centers the coordinates in the unit cell, so there is the same
        amount of vacuum along all cellvectors, specified in affectedDims.

        :param coordinates: list of coordinates of N atoms (Nx3 np.array)
        :param affectedDims: iterable of int/bool/float, specifying the dimensions to act on.
            Default behavior is to center along all vectors, for which pbc is 0

        :return:
        """
        pass

    @abstractmethod
    def decomposeCell(self, other: 'Cell') -> Operation3:  # TODO
        """
        Decompose cell vectors of given unit cell as linear composition of cell vectors of this unit cell.

        :param other: unit cell to decompose.

        :return: 3*3 matrix of decomposition coefficients.
        """
        pass

    @abstractmethod
    def isClose(self, other: 'Cell', tol: float = 5e-2) -> bool:
        """
        Checks if cell vectors of the given cell are close to ones of the other cell

        :param other: unit cell to compare with.

        :return: True or False.
        """
        pass

    @abstractmethod
    def getTrigonalizeTransform(self) -> 'Transformation':
        pass

    @abstractmethod
    def randomTransformation(self) -> 'Transformation':
        """
        Creates transformation object which respects unit cell. I.e. origin is moved only in periodic directions.
        And structure axis if exists is not moved.
        Structure axis is periodic axis in 1D case and axis orthogonal to two periodic axes in 2D case,

        :return: transformation object.
        """
        pass

    @abstractmethod
    def getFittedTransformations(self, initialCoordinates: Vector3, cell: 'Cell') -> list['Transformation']:
        """
        For given set of coordinates and cell object, calculates transformations
        which translate each position into its image within this unit cell along periodic cell vectors of given unit cell.

        :param initialCoordinates: 3 vector of coordinates which need to be translated.
        :param cell: cell object containing periodic vectors along which translation should be done.

        :return: list of **Transformation** objects.
        """
        pass

    @staticmethod
    @abstractmethod
    def getPrincipalAxes(coordinates: VectorN3) -> tuple[Vector3, Operation3]:
        """
        :return: 3x3 matrix of principal axes, main axes of inertia tensor (with all atom masses set to be equal).
        """
