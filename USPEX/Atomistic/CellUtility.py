import logging
logger = logging.getLogger(__name__)


import numpy as np
import spglib
from scipy.spatial.transform import Rotation

from .Transformation import Transformation


_DEFAULT_SYMMETRY_TOLERANCE = 0.05


class CellUtility:

    def __init__(self, pbc, cellVectors = None, cellParameters = None, cellVolume = None, symTolerance=None, axis=None,
                 debug = False):
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
        if self._cell is not None:
            return  self._cell
        else:
            raise RuntimeError('Cell is not fixed.')

    def getRandomCell(self, composition, conditions):
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
        return self._volume if self._volume is not None else conditions.calcCompositionVolume(composition)

    def adjustCell(self, cell, composition, conditions):
        if self._cell is not None:
            cell = self._cell
        else:
            cell = Cell(cellVectors = cell, pbc = self._pbc)
            factor = np.power(self.getCellVolume(composition, conditions) / cell.getVolume(), 1.0 / 3.0)
            cell = Cell(cellVectors = cell.getCellVectors() * factor, pbc = self._pbc)
        return cell

    def getHybridCell(self, cell1, cell2, fraction):
        assert 0 <= fraction <= 1
        vectors = fraction * cell1.getCellVectors() + (1 - fraction) * cell2.getCellVectors()
        vectors /= np.power(np.linalg.det(vectors), 1./3.)
        volume = fraction * cell1.getVolume() + (1 - fraction) * cell2.getVolume()
        return Cell(vectors * np.power(volume, 1./3.), self._pbc)

    @staticmethod
    def volume(system: dict):
        return system['cell'].getVolume()

    def symmetry(self, system: dict):
        cell = system['cell']
        molecules = system['molecules']
        structure, disassembler = type(molecules[0]).assemble(molecules, cell)
        lattice = cell.getCellVectors()
        coordinates = structure.getFractionalCoordinates()
        numbers = [el.z for el in structure.getAtomTypes()]
        if cell.getPBC() == (1,1,1):
            symmetry = '{:7s} {:4s}'.format(*[str(x) for x in spglib.get_spacegroup((lattice, coordinates, numbers),
                                                                                    symprec=self.symTolerance).split()])
        else:
            symmetry = None
        return symmetry


class Cell:

    def __init__(self, cellVectors, pbc):
        self._cellVectors = cellVectors
        self._pbc = pbc

    def getCellVectors(self):
        return self._cellVectors

    def getCellVectorsPBC(self):
        return self._cellVectors[np.nonzero(self._pbc)]

    def getPBC(self):
        return self._pbc

    def getCellParameters(self):
        a = np.linalg.norm(self._cellVectors[0, :])
        b = np.linalg.norm(self._cellVectors[1, :])
        c = np.linalg.norm(self._cellVectors[2, :])
        alpha = 180 / np.pi * np.arccos(np.dot(self._cellVectors[1, :], self._cellVectors[2, :]) / (b * c))
        beta = 180 / np.pi * np.arccos(np.dot(self._cellVectors[0, :], self._cellVectors[2, :]) / (a * c))
        gamma = 180 / np.pi * np.arccos(np.dot(self._cellVectors[0, :], self._cellVectors[1, :]) / (a * b))
        return a, b, c, alpha, beta, gamma

    def getVolume(self):
        return np.abs(np.linalg.det(self._cellVectors))

    def getAltitudes(self):
        volime = self.getVolume()
        l0 = volime / np.linalg.norm(np.cross(self._cellVectors[1, :], self._cellVectors[2, :]))
        l1 = volime / np.linalg.norm(np.cross(self._cellVectors[0, :], self._cellVectors[2, :]))
        l2 = volime / np.linalg.norm(np.cross(self._cellVectors[0, :], self._cellVectors[1, :]))
        return np.array([l0, l1, l2])

    def getCornersCoordinates(self):
        coordinates = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    coordinates.append(i*self._cellVectors[0] + j*self._cellVectors[1] + k*self._cellVectors[2])
        return np.asarray(coordinates, dtype = float)

    def cartesianToFractional(self, coordinates):
        return np.linalg.solve(self._cellVectors.T, coordinates.T).T

    def fractionalToCartesian(self, coordinates):
        return np.dot(self._cellVectors.T, coordinates.T).T

    def fractionalToCartesianOperator(self, operator):
        return np.linalg.solve(self._cellVectors.T, np.dot(self._cellVectors.T, operator.T).T).T

    def getWrapedCartesianCoordinates(self, coordinates):
        return self.fractionalToCartesian(self.getWrapedFractionalCoordinates(self.cartesianToFractional(coordinates)))

    def getWrapedFractionalCoordinates(self, coordinates):
        return np.divmod(coordinates, 1/np.asarray(self._pbc, dtype=float))[1]

    def randomTransformation(self):
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

    @staticmethod
    def initFromCellParameters(a, b, c, alpha, beta, gamma, pbc):
        alpha, beta, gamma = np.pi / 180 * np.asarray((alpha, beta, gamma), dtype=float)
        va = np.array([a, 0, 0])
        vb = np.array([b*np.cos(gamma), b*np.sin(gamma), 0])
        cx = np.cos(beta)
        cy = (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
        cz = (1. - cx ** 2 - cy ** 2) ** 0.5
        vc = c * np.array([cx, cy, cz])
        return Cell(np.vstack((va, vb, vc)), pbc)

    def addVacuum(self, coordinates, vacuumSize):
        """
        @param structure: cartesian atomic coordinates
        @param vacuumSize: ordered container of vacuum distances along cartesian coordinates
        @return:  new cell parameters
        structure = AtomicStructure(structure.getAtomTypes(), newCoords, cell=cell)
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

    def center(self, coordinates, affectedDims=None, toPrincipalAxes=True):
        """
        Center atoms in unit cell.

        Centers the coordinates in the unit cell, so there is the same
        amount of vacuum on all sides specified by affectedDims.

        :param coordinates: list of coordinates of N atoms (Nx3 np.array)
        :param affectedDims: iterable of floats or ints dimensions to act on. Default - act on all dimensions (1,1,1)
        :param toPrincipalAxes: Applies rotation to principal axes, but limited to affectedDims. E.g. for 1d structures
                                affectedDims=(1,1,0) since we don't want to touch z-direction. That means only two
                                principal axes will be found for (x_i, y_i) coordinates and the structure will be turned
                                around z-axis accordingly
        :return:
        """
        if affectedDims is None:
            affectedDims = 1 - np.asarray(self.getPBC(), dtype=int)
        affectedDims = np.array(affectedDims).reshape((1, 3))
        centerCellVec = self.fractionalToCartesian(np.array([0.5, 0.5, 0.5]))
        coordinates -= coordinates.mean(axis=0) * affectedDims
        return coordinates + centerCellVec*affectedDims

    def getPerfectCell(self):
        dim = sum(self._pbc)
        if dim == 3:
            cell = Cell(self.getCellVectors(), self.getPBC())
        elif dim == 2:
            cell = None
        elif dim == 1:
            cell = None
        elif dim == 0:
            cell = Cell(np.eye(3), self.getPBC())
        else:
            raise ValueError(f'Incorrect dim {dim}')
        return cell