import numpy as np
from copy import copy
from scipy.spatial.transform import Rotation
from scipy.linalg import orthogonal_procrustes

from ..Transformation import Transformation

class Cell:
    """
    Class representing unit cell of atomic structure.
    """

    def __init__(self, cellVectors, pbc):
        """

        :param cellVectors: 3*3 array of cell vectors.
        :param pbc: periodic boundary conditions in each direction.

        """
        self._pbc = tuple(pbc)
        self._antipbc = tuple((~np.asarray(pbc, dtype=bool)).tolist())
        self.dim = sum(pbc)
        self._cellVectors = np.asarray(cellVectors, dtype=float)
        assert self._cellVectors.shape == (3, 3)

    @staticmethod
    def initFromCellVectors(pbc, cellVectors=()):
        """
        Alternative constructor using cell vectors only for periodic directions.

        :param pbc: periodic boundary conditions in each direction.
        :param cellVectors: cell vectors only for periodic directions.
        :return: **Cell** object with appropriate parameters.
        """
        dim = sum(pbc)
        assert len(cellVectors) == dim, f"Provided {len(cellVectors)} cell vectors when dim is {dim}."
        if dim == 3:
            return Cell(cellVectors, pbc)
        elif dim == 2:
            vec1, vec2 = cellVectors
            axis = np.cross(vec1, vec2)
            axis /= np.linalg.norm(axis)
            if pbc == (1, 1, 0):
                cellVectors = np.asarray([vec1, vec2, axis], dtype=float)
            elif pbc == (1, 0, 1):
                cellVectors = np.asarray([vec2, axis, vec1], dtype=float)
            elif pbc == (0, 1, 1):
                cellVectors = np.asarray([axis, vec1, vec2], dtype=float)
            else:
                raise RuntimeError("Impossible!")
            return Cell(cellVectors, pbc)
        elif dim == 1:
            axis, = cellVectors
            a = np.linalg.norm(axis)
            axis /= a
            return Cell.initFromCellParameters(pbc, a, axis)
        elif dim == 0:
            return Cell(np.eye(3), pbc)
        else:
            raise RuntimeError(f"Wrong pbc {pbc}.")

    @staticmethod
    def initFromCellParameters(pbc, *args, a=None, b=None, c=None, alpha=None, beta=None, gamma=None, axis=None):
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
        dim = sum(pbc)
        if len(args) > 0:
            if dim == 3:
                assert len(args) == 6
                assert a is None and b is None and c is None and alpha is None and beta is None and gamma is None
                a, b, c, alpha, beta, gamma = args
            elif dim == 2:
                assert len(args) == 4
                assert axis is None and a is None and b is None and alpha is None
                a, b, alpha, axis = args
            elif dim == 1:
                assert len(args) == 2
                assert axis is None and a is None
                a, axis = args
            else:
                raise RuntimeError(f"Wrong pbc {pbc}.")
        if dim == 3:
            assert axis is None and a is not None and b is not None and c is not None and \
                   alpha is not None and beta is not None and gamma is not None
            alpha, beta, gamma = np.pi / 180 * np.asarray((alpha, beta, gamma), dtype=float)
            va = np.array([a, 0, 0])
            vb = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
            cx = np.cos(beta)
            cy = (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
            cz = (1. - cx ** 2 - cy ** 2) ** 0.5
            vc = c * np.array([cx, cy, cz])
            return Cell(np.vstack((va, vb, vc)), pbc)
        elif dim == 2:
            assert axis is not None and a is not None and b is not None and c is None and \
                   alpha is not None and beta is None and gamma is None
            alpha= np.pi / 180 * alpha
            axis = np.asarray(axis, dtype=float)
            axis /= np.linalg.norm(axis)
            va = np.array([a, 0, 0])
            vb = np.array([b * np.cos(alpha), b * np.sin(alpha), 0])
            return Cell.initFromCellVectors(pbc, np.vstack((va, vb))).getAlignedCell(axis)
        elif dim == 1:
            assert axis is not None and a is not None and b is None and c is None and \
                   alpha is None and beta is None and gamma is None
            cellVectors = np.eye(3)
            cellVectors[np.nonzero(pbc)] *= a
            return Cell(cellVectors, pbc).getAlignedCell(axis)
        elif dim == 0:
            assert axis is None and a is None and b is None and c is None and \
                   alpha is None and beta is None and gamma is None
            return Cell(np.eye(3), pbc)
        else:
            raise RuntimeError(f"Wrong pbc {pbc}.")

    def getOrthogonallyTransformedCell(self, targetCell):
        """
        Transforms cellVectors closely to the cellVectors of the targetCell
        within orthogonal transformation.

        :param targetCell: targen cell to be aligned with.

        :return: **Cell** object with adjusted parameters.
        """

        cellVectors = self.getCellVectors()
        targetCellVectors = targetCell.getCellVectors()
        transformationMatrix, _ = orthogonal_procrustes(cellVectors, targetCellVectors)
        newCellVectors = (transformationMatrix.T @ cellVectors.T).T
        newCell = type(self)(newCellVectors, pbc=self.getPBC())
        return newCell

    def getAlignedCell(self, axis):
        """
        Creates cell with same parameters with given one but aligned along given axis.
        :param axis: peropdic axis (1D) or non-periodic axis (2D).
        :raises RuntimeError: if used on 0D or 3D structure.
        :return: Cell with lattice vectors with non-periodic (for 2D) or periodic (1D) aligned along axis.
        """
        if axis is None:
            return self
        if (self.dim == 2) or (self.dim == 1):
            assert np.linalg.norm(axis) >= 1e-7
            a = self.getCellVectorsAntiPBC()[0] if self.dim == 2 else self.getCellVectorsPBC()[0]
            b = np.asarray(axis, dtype=float)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            c = np.dot(a, b)
            v = np.cross(a, b)
            s = np.linalg.norm(v)
            eps = 1e-7
            if s < eps:
                v = np.cross((0, 0, 1), b)
                if np.linalg.norm(v) < eps:
                    v = np.cross((1, 0, 0), b)
                    assert np.linalg.norm(v) >= eps
            elif s > 0:
                v /= s
            cellVectors = self.getCellVectors()
            return Cell((c * cellVectors - np.cross(cellVectors, s * v) + np.outer(np.dot(cellVectors, v), v - c * v)),
                        self._pbc)
        else:
            raise RuntimeError(f"Wrong dim {self.dim}.")

    def getIntrinsicCell(self, coordinates):
        """
        :return: **Cell** object depending on dimensionality.

            0d: Cell made of unit principal eigenvectors

            1d: Keep periodic vector from original cell, The other two are perpendicular to it,
            directed along principal directions of
            a structure, flatten along periodic vector

            2d: Keep periodic vectors from original cell. The third is a unity vector perpendicular to those two.

            3d: Returns original Cell
        """


        coordinates = np.asarray(coordinates, dtype=float)
        periodicVecs = self.getCellVectorsPBC()
        nonperiodicVecs = self.getCellVectorsAntiPBC()

        if self.dim == 0:
            vectors = self.getPrincipalAxes(coordinates)[1].T
        elif self.dim == 1:
            periodicUnit = periodicVecs[0] / np.linalg.norm(periodicVecs[0])
            orthogPancake = coordinates - np.dot(coordinates, periodicUnit).reshape(-1, 1) * periodicUnit
            val, vectors = self.getPrincipalAxes(orthogPancake)
            vectors = vectors.T
            if val[0] < 1e-5:  # Check if inertia tensor has a singular matrix
                if np.dot(vectors[0], periodicUnit) == 1:
                    vectors[0] = vectors[1]
                vectors[0] -= np.dot(vectors[0], periodicUnit) * periodicUnit
                vectors[0] /= np.linalg.norm(vectors[0])
                vectors[1] = np.cross(periodicUnit, vectors[0])
            vectors[-1] = periodicVecs[0] # any 2D shape has a maximum inertia moment corresponding to orth direction
            vectors = np.roll(vectors, np.where(self._pbc)[0][0] - 2, axis=0)
        elif self.dim == 2:
            normalvector = np.cross(periodicVecs[0], periodicVecs[1])
            normalvector *= np.sign(np.dot(normalvector, nonperiodicVecs[0]))
            vectors = copy(self._cellVectors)
            vectors[np.flatnonzero(self._antipbc)] = normalvector
        elif self.dim == 3:
            return copy(self)
        else:
            raise ValueError(f'Incorrect dim: {self.dim}')

        newCell = Cell(vectors, self._pbc)
        return newCell

    def getEnvelopeCell(self, coordinates=None, vacuumSize: float = 0.0, intrinsic=False):
        """
        :param coordinates: cartesian atomic coordinates
        :param vacuumSize: vacuum distance added along cell vector

        :return:  new cell object, corresponding to
        """
        cell = self.getIntrinsicCell(coordinates) if intrinsic and coordinates is not None else self
        newCellVectors = []
        for vector, isPeriodic in zip(cell.getCellVectors(), self._pbc):
            if isPeriodic:
                newCellVectors.append(vector)
            else:
                vector = vector / np.linalg.norm(vector)
                if coordinates is not None:
                    proj = np.dot(coordinates, vector)
                    newCellVectors.append((np.max(proj) - np.min(proj) + vacuumSize) * vector)
                else:
                    newCellVectors.append(vacuumSize * vector)
        return Cell(np.asarray(newCellVectors, dtype=float), self._pbc)

    def getOptimizedCell(self):
        """
        Idea is as follows - if any lattice vector has projection onto any other lattice vector
        that is greater than half of length of this vector, we can reoptimize the shape
        i. e. if |a*b|/|b| > |b|/2 then a_new = a - ceil(|a*b|/|b|^2)*sign(a*b)*b

        :return: **Cell** object with optimized lattice vectors.
        """

        def reoptimizeVector(v1: np.ndarray, v2: np.ndarray, flag: int):
            """
            The function reoptimizes a vector against another vector. Needs better description.

            :param v1: vector to reoptimize.
            :param v2: vector against which we reoptimize v1.
            :param flag: flag that is raised if a new v1 has been found with norm less than the original v1.
            :return: (v1, flag)
            """

            v = np.copy(v1)

            dot_v1_v2 = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)

            if abs(dot_v1_v2) > norm_v2 ** 2 / 2:  # corrected norm_v2 / 2 -> norm_v2 ** 2 / 2 by V. Baturin 09.11.18
                v1_trial = v1 - np.ceil(abs(dot_v1_v2) / (norm_v2 ** 2)) * np.sign(dot_v1_v2) * v2

                if np.linalg.norm(v1_trial) < norm_v1:
                    v = v1_trial
                    flag = 1

            return v, flag

        if self.dim == 3:
            v1, v2, v3 = self._cellVectors
            flag = 1
            step = 0
            while flag and step < 100:
                flag = 0
                v1, flag = reoptimizeVector(v1, v2, flag)
                v1, flag = reoptimizeVector(v1, v3, flag)
                v2, flag = reoptimizeVector(v2, v1, flag)
                v2, flag = reoptimizeVector(v2, v3, flag)
                v3, flag = reoptimizeVector(v3, v1, flag)
                v3, flag = reoptimizeVector(v3, v2, flag)
                v1, flag = reoptimizeVector(v1, v2 + v3, flag)
                v2, flag = reoptimizeVector(v2, v1 + v3, flag)
                v3, flag = reoptimizeVector(v3, v1 + v2, flag)
                step += 1
            return Cell(np.array((v1, v2, v3), dtype=float), self._pbc)
        elif self.dim == 2:
            v1, v2 = self.getCellVectorsPBC()
            thickness = self.getLength()
            flag = 1
            step = 0
            while flag and step < 100:
                flag = 0
                v1, flag = reoptimizeVector(v1, v2, flag)
                v2, flag = reoptimizeVector(v2, v1, flag)
                step += 1
            return Cell.initFromCellVectors(self._pbc, np.array((v1, v2), dtype=float)).getEnvelopeCell \
                (vacuumSize=thickness)
        else:
            return Cell(self.getCellVectors(), self._pbc)

    def getPBC(self):
        """
        :return: periodic boundary conditions in each direction.
        """
        return self._pbc

    def getAntiPBC(self):
        """
        :return: logic not of periodic boundary conditions in each direction.
        """
        return self._antipbc

    def getCellVectors(self):
        """
        :return: 3*3 array of cell vectors.
        """
        return np.copy(self._cellVectors)

    def getCellVectorsPBC(self):
        """
        :return: x*3 array of periodic cell vectors, where 0 <= x <= 3.
        """
        return np.copy(self._cellVectors)[np.nonzero(self._pbc)]

    def getCellVectorsAntiPBC(self):
        """
        :return: x*3 array of non-periodic cell vectors, where 0 <= x <= 3.
        """
        return np.copy(self._cellVectors)[np.nonzero(self._antipbc)]

    def getCellParameters(self):
        """
        :return:  tuple of cell parameters: a, b, c, alpha, beta, gamma
        """
        if self.dim == 3:
            a = np.linalg.norm(self._cellVectors[0, :])
            b = np.linalg.norm(self._cellVectors[1, :])
            c = np.linalg.norm(self._cellVectors[2, :])
            alpha = 180 / np.pi * np.arccos(np.dot(self._cellVectors[1, :], self._cellVectors[2, :]) / (b * c))
            beta = 180 / np.pi * np.arccos(np.dot(self._cellVectors[0, :], self._cellVectors[2, :]) / (a * c))
            gamma = 180 / np.pi * np.arccos(np.dot(self._cellVectors[0, :], self._cellVectors[1, :]) / (a * b))
            return a, b, c, alpha, beta, gamma
        elif self.dim == 2:
            cellVectors = self.getCellVectorsPBC()
            axis, = self.getCellVectorsAntiPBC()
            axis /= np.linalg.norm(axis)
            a = np.linalg.norm(cellVectors[0, :])
            b = np.linalg.norm(cellVectors[1, :])
            alpha = 180 / np.pi * np.arccos(np.dot(cellVectors[0, :], cellVectors[1, :]) / (a * b))
            return a, b, alpha, axis
        elif self.dim == 1:
            axis, = self.getCellVectorsPBC()
            a = np.linalg.norm(axis)
            axis /= a
            return a, axis
        elif self.dim == 0:
            return ()
        else:
            raise RuntimeError(f"Wrong dim {self.dim}.")

    def __eq__(self, other):
        if self.dim == 0 or self.dim == 3:
            return np.allclose(self.getCellParameters(), other.getCellParameters())
        elif self.dim == 1:
            a1, axis1 = self.getCellParameters()
            a2, axis2 = other.getCellParameters()
            return np.allclose(a1 * axis1, a2 * axis2)
        elif self.dim == 2:
            a1, b1, alpha1, axis1 = self.getCellParameters()
            a2, b2, alpha2, axis2 = other.getCellParameters()
            return np.allclose([a1, b1, alpha1, *axis1], [a2, b2, alpha2, *axis2])
        else:
            raise RuntimeError(f"Wrong dim {self.dim}.")

    def getVolume(self):
        """
        :return: unit cell volume if cell is 3D periodic.
        """
        assert self.dim == 3 or self.dim == 0
        return np.abs(np.linalg.det(self._cellVectors))

    def getArea(self):
        """
        :return: unit cell area if cell is 2D periodic or area of enveloping volume on 1D.
        """
        assert self.dim == 2 or self.dim == 1
        if self.dim == 2:
            nonzeroPBC = np.nonzero(self._pbc)[0]
            nonzeroCellVectors = self._cellVectors[nonzeroPBC]
        else:
            nonzeroPBC = np.nonzero(self._antipbc)[0]
            nonzeroCellVectors = self._cellVectors[nonzeroPBC]
        return np.linalg.norm(np.cross(nonzeroCellVectors[0], nonzeroCellVectors[1]))

    def getLength(self):
        """
        :return: unit cell length if cell is 1D periodic or height of enveloping volume in 2D.
        """
        assert self.dim == 2 or self.dim == 1
        if self.dim == 1:
            nonzeroPBC = np.nonzero(self._pbc)[0]
            nonzeroCellVectors = self._cellVectors[nonzeroPBC]
        else:
            nonzeroPBC = np.nonzero(self._antipbc)[0]
            nonzeroCellVectors = self._cellVectors[nonzeroPBC]
        return np.linalg.norm(nonzeroCellVectors[0])

    def getRadius(self):
        """
        :return: unit cell envelope radius if cell is 1D periodic or non-periodic (0D).
        """
        assert self.dim <= 1
        return np.linalg.norm(self._cellVectors[np.nonzero(self._antipbc)]) / 2.0

    def getCornersCoordinates(self):
        """
        :return: 8*3 array of absolute coordinates of each corner of unit cell.
        """
        coordinates = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    coordinates.append( i *self._cellVectors[0] + j* self._cellVectors[1] + k * self._cellVectors[2])
        return np.asarray(coordinates, dtype=float)

    def getMaxNumSlabs(self, axis, N):
        """
        Calculates maximal number of choices of slabs origins in given direction.
        :param axis: direction axis.
        :param N: number of atoms/molecules in cell.
        :return: maximal number of choices of slabs origins.
        """
        if self.dim == 3:
            volume = self.getVolume()
            if axis == 0:
                L = volume / np.linalg.norm(np.cross(self._cellVectors[1, :], self._cellVectors[2, :]))
            elif axis == 1:
                L = volume / np.linalg.norm(np.cross(self._cellVectors[0, :], self._cellVectors[2, :]))
            elif axis == 2:
                L = volume / np.linalg.norm(np.cross(self._cellVectors[0, :], self._cellVectors[1, :]))
            else:
                raise ValueError(f"Wrong axis {axis}.")
            Lchar = 0.5 * np.power((volume / N), (1 / 3))  # average 'radius' of a molecule in the cell
            Nmax = L / Lchar
        elif self.dim == 2:
            if self._pbc[axis] == 0:
                L = self.getLength()
                Lchar = 0.5 * np.power((L * self.getArea() / N), (1 / 3))
                Nmax = L / Lchar
            else:
                volume = self.getArea() * self.getLength()
                if axis == 0:
                    L = volume / np.linalg.norm(np.cross(self._cellVectors[1, :], self._cellVectors[2, :]))
                elif axis == 1:
                    L = volume / np.linalg.norm(np.cross(self._cellVectors[0, :], self._cellVectors[2, :]))
                elif axis == 2:
                    L = volume / np.linalg.norm(np.cross(self._cellVectors[0, :], self._cellVectors[1, :]))
                else:
                    raise ValueError(f"Wrong axis {axis}.")
                Lchar = 0.5 * np.power((volume / N), (1 / 3))
                Nmax = L / Lchar
        elif self.dim == 1:
            if self._pbc[axis] == 1:
                L = self.getLength()
                Lchar = 0.5 * np.power((L * self.getArea() / N), (1 / 3))
                Nmax = L / Lchar
            else:
                area = self.getArea()
                L = 2 * np.sqrt(area / np.pi)
                Lchar = 0.5 * np.power((area * self.getLength() / N), (1 / 3))
                Nmax = L / Lchar
        elif self.dim == 0:
            Nmax = 0.5 * np.power((6 * np.pi ** 2 * N), (1 / 3))
        else:
            raise RuntimeError(f"Wrong dim {self.dim}.")
        return Nmax

    def cartesianToFractional(self, coordinates):
        """
        Converts cartesian coordinates to fractional.

        :param coordinates: N*3 array of cartesian coordinates

        :return: N*3 array of fractional coordinates.
        """
        return np.linalg.solve(self._cellVectors.T, coordinates.T).T

    def fractionalToCartesian(self, coordinates):
        """
        Converts fractional coordinates to cartesian.

        :param coordinates: N*3 array of fractional coordinates.

        :return: N*3 array of cartesian coordinates
        """
        return np.dot(self._cellVectors.T, coordinates.T).T

    def fractionalToCartesianOperator(self, operator):
        """
        Сonvert some operator matrix defined in fractional space to transformation object in cartesian space.

        :param operator: 4*4 matrix defining operator in fractional space.

        :return:
        **Transformation** object defining transformation in cartesian space.
        """
        rotMatrixFrac = np.asarray(operator)[:3, :3]
        rotMatrixCart = np.dot(self._cellVectors.T, np.dot(rotMatrixFrac, np.linalg.inv(self._cellVectors.T)))
        if np.allclose(np.dot(rotMatrixCart, rotMatrixCart.T), np.eye(3, dtype=float), atol=1e-03):
            transVecFrac = np.asarray(operator)[:3, 3]
            return Transformation(rotMatrixCart, self.fractionalToCartesian(transVecFrac))
        else:
            raise ValueError('rotation matrix is not unitary')

    def getWrapedCartesianCoordinates(self, coordinates):
        """
        Translates all coordinates to their image within unit cell via cell vectors in cartesian space.

        :param coordinates: cartesian coordinates

        :return: cartesian coordinates wrapped to unit cell.
        """
        return self.fractionalToCartesian(self.getWrapedFractionalCoordinates(self.cartesianToFractional(coordinates)))

    def getWrapedFractionalCoordinates(self, coordinates):
        """
        Translates all coordinates to their image within unit cell via cell vectors in fractional space.

        :param coordinates: fractional coordinates

        :return: fractional coordinates wrapped to unit cell.
        """
        wrapedCoordinates = np.copy(coordinates).reshape((-1, 3))
        inds = np.nonzero(self._pbc)
        wrapedCoordinates[:, inds] = np.divmod(wrapedCoordinates[:, inds], 1)[1]
        return wrapedCoordinates.reshape(coordinates.shape)

    def center(self, coordinates, affectedDims=None):
        """
        Center atoms in unit cell.

        Centers the coordinates in the unit cell, so there is the same
        amount of vacuum along all cellvectors, specified in affectedDims.

        :param coordinates: list of coordinates of N atoms (Nx3 np.array)
        :param affectedDims: iterable of int/bool/float, specifying the dimensions to act on.
            Default behavior is to center along all vectors, for which pbc is 0

        :return:
        """
        coordinates = np.asarray(coordinates, dtype=float)
        if affectedDims is None:
            affectedDims = self.getAntiPBC()
        affectedDims = np.array(affectedDims).reshape((1, 3))
        fracCoords = self.cartesianToFractional(coordinates)
        shift = np.array([0.5, 0.5, 0.5]) - 0.5 * (np.min(fracCoords, axis=0) + np.max(fracCoords, axis=0))
        newFrac = fracCoords + shift * affectedDims
        return self.fractionalToCartesian(newFrac)

    def decomposeCell(self, other):  # TODO
        """
        Decompose cell vectors of given unit cell as linear composition of cell vectors of this unit cell.

        :param other: unit cell to decompose.

        :return: 3*3 matrix of decomposition coefficients.
        """
        if self._pbc == other.getPBC():
            matrix = np.eye(3)
            inds = np.nonzero(self._pbc)
            matrix[inds] = np.linalg.solve(self.getCellVectors().T, other.getCellVectors().T).T[inds]
            return matrix
        else:
            return np.eye(3)

    def isClose(self, other, tol=5e-2):
        """
        Checks if cell vectors of the given cell are close to ones of the other cell

        :param other: unit cell to compare with.

        :return: True or False.
        """
        decompositionMatrix = self.decomposeCell(other)
        return np.isclose(np.linalg.norm(decompositionMatrix, axis=1).mean(), 1.0, atol=tol)

    def getTrigonalizeTransform(self):
        normCellVectors = copy(self._cellVectors)
        normCellVectors /= np.linalg.norm(normCellVectors, axis=1).reshape((-1, 1))
        normCellParameters = Cell(normCellVectors, self._pbc).getCellParameters()
        standardCellVectors = Cell.initFromCellParameters(self._pbc, *normCellParameters).getCellVectors()
        matrix = np.linalg.solve(standardCellVectors, normCellVectors)
        return Transformation.fromMatrix(matrix, np.array([0.0, 0.0, 0.0]))

    def randomTransformation(self):
        """
        Creates transformation object which respects unit cell. I.e. origin is moved only in periodic directions.
        And structure axis if exists is not moved.
        Structure axis is periodic axis in 1D case and axis orthogonal to two periodic axes in 2D case,

        :return: transformation object.
        """
        pbcVectorsCart = self.getCellVectorsPBC()
        pbcSum = len(pbcVectorsCart)
        transVec = np.dot(np.random.rand(pbcSum), pbcVectorsCart)
        if pbcSum == 0:
            dir = Rotation.random().as_rotvec()
            norm = np.linalg.norm(dir)
            dir /= norm
            refMatrix = np.eye(3)
            if norm > np.pi:
                refMatrix -= 2.0 * np.outer(dir, dir)
            rotMatrix = refMatrix.dot(Rotation.random().as_matrix())
        elif pbcSum == 1:
            axis = pbcVectorsCart[0]
            rotVec = np.pi * np.random.random() * axis / np.linalg.norm(axis)
            rotMatrix = Rotation.from_rotvec(rotVec).as_matrix()
        elif pbcSum == 2:
            axis = np.cross(pbcVectorsCart[0, :], pbcVectorsCart[1, :])
            rotVec = np.pi * np.random.random() * axis / np.linalg.norm(axis)
            rotMatrix = Rotation.from_rotvec(rotVec).as_matrix()
        else:
            rotMatrix = Rotation.random().as_matrix()
        centerCellVec = self.fractionalToCartesian(np.array([0.5, 0.5, 0.5]))
        transVec = transVec - centerCellVec + np.dot(rotMatrix.T, centerCellVec)
        return Transformation.fromMatrix(rotMatrix, np.dot(rotMatrix, transVec))

    def getFittedTransformations(self, initialCoordinates, cell):
        """
        For given set of coordinates and cell object, calculates transformations
        which translate each position into its image within this unit cell along periodic cell vectors of given unit cell.

        :param initialCoordinates: 3 vector of coordinates which need to be translated.
        :param cell: cell object containing periodic vectors along which translation should be done.

        :return: list of **Transformation** objects.
        """
        inds = np.nonzero(cell.getPBC())
        vectors = cell.getCellVectors()[inds]
        minAndMax = np.asarray([(np.min(coords), np.max(coords))
                                for coords in cell.cartesianToFractional(self.getCornersCoordinates()).T[inds]],
                               dtype=float) \
                    - cell.cartesianToFractional(initialCoordinates)[inds]
        minAndMax = np.asarray(np.ceil(minAndMax), dtype=int).reshape((-1, 2))
        if np.all(minAndMax[:, 1] > minAndMax[:, 0]):
            closeShifts = []
            for minCoordinate, maxCoordinate in minAndMax:
                if closeShifts:
                    newCloseShifts = []
                    for shift in closeShifts:
                        newCloseShifts.extend([shift + (i,) for i in range(minCoordinate, maxCoordinate)])
                    closeShifts = newCloseShifts
                else:
                    closeShifts = [(i,) for i in range(minCoordinate, maxCoordinate)]
            closeShifts = np.asarray(closeShifts, dtype=int)
            closeCoordinates = initialCoordinates.reshape((1, 3)) + np.dot(closeShifts, vectors)
        else:
            closeCoordinates = []

        fittedCoordinates = [coord for coord in closeCoordinates
                             if (np.all(0. <= self.cartesianToFractional(coord)[inds]) and
                                 np.all(self.cartesianToFractional(coord)[inds] < 1.))]

        return [Transformation.fromRotVector([0., 0., 0.], finalCoordinates - initialCoordinates)
                for finalCoordinates in fittedCoordinates]

    @staticmethod
    def getPrincipalAxes(coordinates):
        """
        :return: 3x3 matrix of principal axes, main axes of inertia tensor (with all atom masses set to be equal).
        """
        coordinates = np.asarray(coordinates, dtype=float)
        coordinates = coordinates - coordinates.mean(axis=0)
        val, vectors = np.linalg.eigh(np.eye(3) * np.sum(coordinates ** 2) - np.dot(coordinates.T, coordinates))
        if np.linalg.det(vectors) < 0:
            vectors[:, -1] = - vectors[:, -1]
        return val, vectors
