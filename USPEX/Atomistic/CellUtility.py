import logging
logger = logging.getLogger(__name__)


import numpy as np
import spglib
from typing import Union, List, Tuple
from copy import copy
from scipy.spatial.transform import Rotation

from .Transformation import Transformation


_DEFAULT_SYMMETRY_TOLERANCE = 0.05


class CellUtility:
    """
    Utility for working with unit cells of atomic structures.
    """

    def __init__(self, pbc, cellVectors = None, cellParameters = None, cellVolume = None,
                 reconstructionDegree = None, symTolerance=None, axis=None, debug = False):
        """

        :param pbc: periodic boundary conditions in each direction.
        :param cellVectors: for fixed cell calculation 3*3 array of cell vectors.
        :param cellParameters: for fixed cell calculation dictionary
        {'a': <float>, 'b': <float>, 'c': <float>, 'alpha': <float>, 'beta': <float>, 'gamma': <float>}
        with cell parameters.
        :param cellVolume: for fixed volume calculation cell volume.
        :param reconstructionDegree: int or list of int with allowed supercell sizes.
        :param symTolerance: allowed misplacements of atoms when determining symmetry of structure.
        :param axis: for 1D periodic calculations vector along periodic axis,
        for 2D periodic calculations vector orthogonal to two periodic axes.
        :param debug: switch between two levels of logging. True for debug level, false for INFO level.
        """
        self._pbc = pbc
        if cellVectors is not None:
            self._cell = Cell(cellVectors, pbc)
            self._volume = self._cell.getVolume()
            assert cellParameters is None and cellVolume is None
        elif cellParameters is not None:
            self._cell = Cell.initFromCellParameters(**cellParameters, pbc = pbc)
            self._volume = self._cell.getVolume()
            assert cellVolume is None
        elif cellVolume is not None:
            self._cell = None
            self._volume = cellVolume
        else:
            self._cell = None
            self._volume = None

        if reconstructionDegree is not None:
            self._listOfReconstructions = self._getListOfReconstructions(reconstructionDegree)
        else:
            self._listOfReconstructions = []
        self.dim = sum(self._pbc)
        self._axis = axis
        assert not (self._axis is not None and self._cell is not None)
        if self._cell is not None:
            if self.dim == 2:
                vec1, vec2 = self._cell.getCellVectorsPBC()
                self._axis = np.cross(vec1, vec2)
                self._axis /= np.linalg.norm(self._axis)
            elif self.dim == 1:
                self._axis, = self._cell.getCellVectorsPBC()
                self._axis /= np.linalg.norm(self._axis)

        if symTolerance is not None:
            if isinstance(symTolerance, str):
                if 'high' in symTolerance:
                    self.symTolerance = 0.05
                elif 'medium' in symTolerance:
                    self.symTolerance = 0.1
                elif 'low' in symTolerance:
                    self.symTolerance = 0.2
                else:
                    self.symTolerance = _DEFAULT_SYMMETRY_TOLERANCE
            elif isinstance(symTolerance, (float, int)):
                self.symTolerance = float(symTolerance)
            else:
                self.symTolerance = _DEFAULT_SYMMETRY_TOLERANCE
        else:
            self.symTolerance = _DEFAULT_SYMMETRY_TOLERANCE

        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def getCell(self):
        """

        :return: for fixed cell calculations reference to the unit cell object.
        If cell in calculation is not fixed raises RuntimeError.
        """
        if self._cell is not None:
            return self._cell
        else:
            raise RuntimeError('Cell is not fixed.')

    def getRandomCell(self, composition, conditions):
        """
        For given composition and conditions creates random unit cell with appropriate size and periodic boundary conditions.
        :param composition: dictionary like object defining composition.
        :param conditions: external conditions.
        :return: **Cell** object with appropriate parameters.
        """
        try:
            return self.getCell()
        except RuntimeError:
            volume = self.getCellVolume(composition, conditions)
            a, b, c = np.random.random(3) + 0.5
            x = 0.0
            while x < 0.3:
                alpha, beta, gamma = (np.random.random(3) * 4 + 1) * np.pi / 6
                x = 1. - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 2.*np.cos(alpha)*np.cos(beta)*np.cos(gamma)

            r2d = 180/np.pi
            cell = Cell.initFromCellParameters(a, b, c, alpha*r2d, beta*r2d, gamma*r2d, pbc=self._pbc)
            return Cell(cellVectors=cell.getCellVectors()*((volume / cell.getVolume()) ** (1. / 3.)), pbc = self._pbc)

    def getCellVolume(self, composition, conditions):
        """
        Either return predefined volume for fixed volume calculation or calculates it using provided conditions utility.
        :param composition: dictionary like object defining composition.
        :param conditions: external conditions.
        :return: volume of unit cell.
        """
        return self._volume if self._volume is not None else conditions.calcCompositionVolume(composition)

    def _getListOfReconstructions(self, reconstructionDegree: Union[int, List, Tuple]):
        assert sum(self._pbc) == 2 # For surfaces only 
        if isinstance(reconstructionDegree, int):
            minReconstruction = reconstructionDegree
            maxReconstruction = reconstructionDegree+1
        elif isinstance(reconstructionDegree, (list, tuple)):
            assert len(reconstructionDegree) == 2
            minReconstruction = reconstructionDegree[0]
            maxReconstruction = reconstructionDegree[1]
        listOfReconstructions = []
        nonzeroPBC = np.nonzero(self._pbc)[0]
        for i in range(1, maxReconstruction + 1):
            for j in range(maxReconstruction + 1):
                for k in range(maxReconstruction + 1):
                    for l in range(1, maxReconstruction + 1):
                        M = np.eye(3)
                        rows = np.array([nonzeroPBC, nonzeroPBC])
                        cols = rows.T
                        M[rows, cols] = np.array([[i, -j], [k, l]])
                        if minReconstruction <= np.round(np.linalg.det(M)) < maxReconstruction and M[nonzeroPBC[0]].dot(M[nonzeroPBC[1]]) == 0:
                            listOfReconstructions.append(M)
        return listOfReconstructions
    
    def _getRandomReconstruction(self, factor=None):
        if self._listOfReconstructions:
            if factor:
                mask = np.linalg.det(self._listOfReconstructions) > factor
            else:
                mask = np.ones(len(self._listOfReconstructions), dtype=bool)
            assert sum(mask) # check if any suitable reconstruction found for this composition
            idx = np.random.choice(np.arange(len(self._listOfReconstructions))[mask])
            return self._listOfReconstructions[idx]
        else:
            return np.eye(3)

    def adjustCell(self, cell, composition, conditions):
        """
        Adjust given unit cell according calculation parameters, provided composition and conditions.
        If cell in calculation is fixed returns the fixed cell. Otherwise scale input unit cell to have specific volume.
        If calculation is done with fixed volume then unit cell is scaled to it.
        Otherwise to estimated volume calculated using provided conditions utility.
        Periodic boundary conditions of output unit cell set up same as in utility.
        :param cell: input unit cell to be adjusted.
        :param composition: dictionary like object defining composition.
        :param conditions: external conditions.
        :return: **Cell** object with adjusted parameters.
        """
        if self._cell is not None:
            if self._listOfReconstructions:
                factor = self.getCellVolume(composition, conditions) / self._cell.getVolume()
                reconstruction = self._getRandomReconstruction(factor)
                cell = Cell(cellVectors=reconstruction.dot(self._cell.getCellVectors()), pbc=self._pbc)
            else:
                cell = self._cell
        else:
            cell = Cell(cellVectors = cell, pbc = self._pbc)
            factor = np.power(self.getCellVolume(composition, conditions) / cell.getVolume(), 1.0 / 3.0)
            cell = Cell(cellVectors = cell.getCellVectors() * factor, pbc = self._pbc)
        return cell

    def getHybridCell(self, cell1, cell2, fraction):
        """
        Creates hybrid of two unit cells.
        TODO better to average parameters rather than vectors.
        :param cell1:
        :param cell2:
        :param fraction:
        :return:
        """
        try:
            return self.getCell()
        except RuntimeError:
            assert 0 <= fraction <= 1
            if self._listOfReconstructions:
                matrix1 = np.round(self._cell.decomposeCell(cell1))
                matrix2 = np.round(self._cell.decomposeCell(cell2))
                idx = np.linalg.det([matrix1, matrix2]).argmax()
                cellVectors = [cell1, cell2][idx].getCellVectors()
            else:
                vectors = fraction * cell1.getCellVectors() + (1 - fraction) * cell2.getCellVectors()
                vectors /= np.power(np.linalg.det(vectors), 1./3.)
                volume = fraction * cell1.getVolume() + (1 - fraction) * cell2.getVolume()
                cellVectors = vectors * np.power(volume, 1./3.)
            return Cell(cellVectors, self._pbc)

    @staticmethod
    def volume(system: dict):
        """
        For using in **Fitness** infrastructure
        :param system: dictionary describing system.
        :return: calculated volume of system.
        """
        return system['cell'].getVolume()

    @staticmethod
    def area(system: dict):
        """
        For using in **Fitness** infrastructure
        :param system: dictionary describing system.
        :return: calculated area of system.
        """
        return system['cell'].getArea()
    
    def symmetry(self, system: dict):
        """
        For using in **Fitness** infrastructure
        :param system: dictionary describing system.
        :return: calculated symmetry of system.
        """
        cell = system['cell']
        molecules = system['molecules']
        structure, disassembler = type(molecules[0]).assemble(molecules, cell)
        lattice = cell.getCellVectors()
        coordinates = structure.getFractionalCoordinates()
        numbers = [el.z for el in structure.getAtomTypes()]
        spacegroup = spglib.get_spacegroup((lattice, coordinates, numbers), symprec=self.symTolerance)
        if cell.getPBC() == (1, 1, 1) and spacegroup is not None:
            symmetry = '{:7s} {:4s}'.format(*[str(x) for x in spacegroup.split()])
        else:
            symmetry = None
        return symmetry


class Cell:
    """
    Class representing unit cell of atomic structure.
    """

    def __init__(self, cellVectors, pbc):
        """
        :param cellVectors: 3*3 array of cell vectors.
        :param pbc: periodic boundary conditions in each direction.
        """
        self._cellVectors = np.asarray(cellVectors)
        self._pbc = pbc

    @staticmethod
    def initFromCellParameters(a, b, c, alpha, beta, gamma, pbc):
        """
        Alternative constructor using cell parameters.
        :param a:
        :param b:
        :param c:
        :param alpha:
        :param beta:
        :param gamma:
        :param pbc:
        :return:
        """

        alpha, beta, gamma = np.pi / 180 * np.asarray((alpha, beta, gamma), dtype=float)
        va = np.array([a, 0, 0])
        vb = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
        cx = np.cos(beta)
        cy = (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
        cz = (1. - cx ** 2 - cy ** 2) ** 0.5
        vc = c * np.array([cx, cy, cz])
        return Cell(np.vstack((va, vb, vc)), pbc)

    def getCellVectors(self):
        """
        :return: 3*3 array of cell vectors.
        """
        return copy(self._cellVectors)

    def getCellVectorsPBC(self):
        """
        :return: x*3 array of periodic cell vectors, where 0 <= x <= 3.
        """
        return copy(self._cellVectors)[np.nonzero(self._pbc)]

    def getPBC(self):
        """
        :return: periodic boundary conditions in each direction.
        """
        return self._pbc

    def getCellParameters(self):
        """
        :return:  tuple of cell parameters: a, b, c, alpha, beta, gamma
        """
        a = np.linalg.norm(self._cellVectors[0, :])
        b = np.linalg.norm(self._cellVectors[1, :])
        c = np.linalg.norm(self._cellVectors[2, :])
        alpha = 180 / np.pi * np.arccos(np.dot(self._cellVectors[1, :], self._cellVectors[2, :]) / (b * c))
        beta = 180 / np.pi * np.arccos(np.dot(self._cellVectors[0, :], self._cellVectors[2, :]) / (a * c))
        gamma = 180 / np.pi * np.arccos(np.dot(self._cellVectors[0, :], self._cellVectors[1, :]) / (a * b))
        return a, b, c, alpha, beta, gamma

    def getVolume(self):
        """
        :return: unit cell volume
        """
        return np.abs(np.linalg.det(self._cellVectors))

    def getArea(self):
        """
        :return: uinit cell area if cell is 2D periodic.
        """
        assert sum(self._pbc) >= 2
        nonzeroPBC = np.nonzero(self._pbc)[0]
        nonzeroCellVectors = self._cellVectors[nonzeroPBC][:, nonzeroPBC]
        return np.abs(np.cross(nonzeroCellVectors[0], nonzeroCellVectors[1]))

    def getAltitudes(self):
        """
        :return: altitudes calculates as volume divided by face area for each face.
        """
        volime = self.getVolume()
        l0 = volime / np.linalg.norm(np.cross(self._cellVectors[1, :], self._cellVectors[2, :]))
        l1 = volime / np.linalg.norm(np.cross(self._cellVectors[0, :], self._cellVectors[2, :]))
        l2 = volime / np.linalg.norm(np.cross(self._cellVectors[0, :], self._cellVectors[1, :]))
        return np.array([l0, l1, l2])

    def getCornersCoordinates(self):
        """
        :return: 8*3 array of absolute coordinates of each corner of unit cell.
        """
        coordinates = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    coordinates.append(i*self._cellVectors[0] + j*self._cellVectors[1] + k*self._cellVectors[2])
        return np.asarray(coordinates, dtype = float)

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
        TODO convert some operator matrix defined in fractional space to transformation object in cartesian space.
        :param operator:
        :return:
        """
        return operator 

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
        return np.divmod(coordinates, 1/np.asarray(self._pbc, dtype=float))[1]

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
            rotMatrix = Rotation.random().as_matrix()
        elif pbcSum == 1:
            axis = pbcVectorsCart[0]
            rotVec = np.pi * np.random.random() * axis / np.linalg.norm(axis)
            rotMatrix = Rotation.from_rotvec(rotVec).as_matrix()
        elif pbcSum == 2:
            axis = np.cross(pbcVectorsCart[0,:], pbcVectorsCart[1,:])
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
        :param initialCoordinates: N*3 array of coordinates which need to be translated.
        :param cell: cell object containing periodic vectors along which translation should be done.
        :return: list of **Transformation** objects.
        """
        inds = np.nonzero(cell.getPBC())
        vectors = cell.getCellVectors()[inds]
        minAndMax = np.asarray([(np.min(coords), np.max(coords))
                                for coords in cell.cartesianToFractional(self.getCornersCoordinates()).T[inds]],
                               dtype = float)\
                    - cell.cartesianToFractional(initialCoordinates)[inds].reshape((-1,1))
        minAndMax = np.asarray(np.ceil(minAndMax), dtype=int).reshape((-1,1))
        if np.all(minAndMax[:1] > minAndMax[:0]):
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
            closeCoordinates = initialCoordinates.reshape((1,3)) + np.dot(closeShifts, vectors)
        else:
            closeCoordinates = []

        fittedCoordinates = [coord for coord in closeCoordinates if (np.all(0. <= self.cartesianToFractional(coord)) and
                                                                     np.all(self.cartesianToFractional(coord) < 1.))]

        return [Transformation.fromRotVector([0.,0.,0.], finalCoordinates - initialCoordinates)
                for finalCoordinates in fittedCoordinates]

    def getEnvelopeCell(self, coordinates, vacuumSize=0):
        """
        @param coordinates: cartesian atomic coordinates
        @param vacuumSize: vacuum distance added along cell vector
        @return:  new cell object, corresponding to
        """
        newCellVectors = []
        for vector, isPeriodic in zip(self._cellVectors, self._pbc):
            if isPeriodic:
                newCellVectors.append(vector)
            else:
                vector = vector / np.linalg.norm(vector)
                proj = np.dot(coordinates, vector)
                newCellVectors.append((np.max(proj) - np.min(proj) + vacuumSize) * vector)
        return Cell(np.asarray(newCellVectors, dtype=float), self._pbc)

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
        if affectedDims is None:
            affectedDims = 1 - np.asarray(self._pbc, dtype=int)
        affectedDims = np.array(affectedDims).reshape((1, 3))
        fracCoords = self.cartesianToFractional(coordinates)
        shift = np.array([0.5, 0.5, 0.5]) - 0.5 * (np.min(fracCoords, axis=0) + np.max(fracCoords, axis=0))
        newFrac = fracCoords + shift * affectedDims
        return self.fractionalToCartesian(newFrac)

    def decomposeCell(self, other):
        """
        Decompose cell vectors of given unit cell as linear composition of cell vectors of this unit cell.
        :param other: unit cell to decompose.
        :return: 3*3 matrix of decomposition coefficients.
        """
        assert self._pbc == other.getPBC()
        matrix = np.eye(3)
        inds = np.nonzero(self._pbc)
        matrix[inds] = np.linalg.solve(self.getCellVectors().T,other.getCellVectors().T).T[inds]
        return matrix