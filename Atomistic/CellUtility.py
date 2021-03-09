import numpy as np
from scipy.spatial.transform import Rotation

from .Transformation import Transformation


class CellUtility:

    def __init__(self, pbc, cellVectors = None, cellParameters = None, cellVolume = None):
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


class Cell:

    def __init__(self, cellVectors, pbc):
        self._cellVectors = cellVectors
        self._pbc = pbc

    def getCellVectors(self):
        return self._cellVectors

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
        return np.linalg.solve(self._cellVectors, coordinates.T).T

    def fractionalToCartesian(self, coordinates):
        return np.dot(self._cellVectors, coordinates.T).T

    def getWrapedCartesianCoordinates(self, coordinates):
        return self.fractionalToCartesian(self.getWrapedFractionalCoordinates(self.cartesianToFractional(coordinates)))

    def getWrapedFractionalCoordinates(self, coordinates):
        return np.divmod(coordinates, 1/np.asarray(self._pbc, dtype=float))[1]

    def randomTransformation(self):
        pbcVec = np.array(list(self.getPBC()))
        pbcSum = np.sum(pbcVec)
        matrixDirToCart = np.array(self.getCellVectors()).T
        if pbcSum == 0 or 3:
            rotMatrix = Rotation.random().as_matrix()
            if pbcSum == 0:
                transVec = np.zeros(3)
            else:
                transVec = np.dot(matrixDirToCart, np.random.rand(3))
        else:
            if pbcSum == 1:
                rotAngle = np.pi * np.random.random()
                pbcVecCart = np.dot(matrixDirToCart, pbcVec)
                rotVec = rotAngle * pbcVecCart / np.linalg.norm(pbcVecCart)
                transVec = np.random.random() * pbcVecCart  # pbcVec without norm here
            else:
                rotAngle = np.pi * np.random.random()
                pbcVectorsCart = self._cellVectors[np.nonzero(pbcVec)]
                rotVecWithoutNorm = np.cross(pbcVectorsCart[0,:], pbcVectorsCart[1,:])
                rotVec = rotAngle * rotVecWithoutNorm/np.linalg.norm(rotVecWithoutNorm)
                transVec = np.dot(np.random.rand(2) * pbcVectorsCart)
            rotMatrix = Rotation.from_rotvec(rotVec).as_matrix()
        centerCellVec = np.dot(matrixDirToCart, np.array([0.5, 0.5, 0.5]))
        transVec = transVec - np.dot(rotMatrix, centerCellVec)
        return Transformation.fromMatrix(rotMatrix, np.dot(rotMatrix, transVec))

    def getFittedTransformations(self, initialCoordinates, cell):
        minAndMax = [(np.min(coords), np.max(coords))
                     for coords in cell.cartesianToFractional(self.getCornersCoordinates()).T[np.nonzero(cell.getPBC())]]
        vectors = cell.getCellVectors()[np.nonzero(cell.getPBC())]
        closeCoordinates = [cell.getWrapedCartesianCoordinates(initialCoordinates)]
        N = len(closeCoordinates)
        while True:
            for probeCoordinates in closeCoordinates[:]:
                for vector, (minCoordinate, maxCoordinate) in zip(vectors, minAndMax):
                    closeCoordinates.extend(_findClose(probeCoordinates + vector,  vector,  minCoordinate,  maxCoordinate))
                    closeCoordinates.extend(_findClose(probeCoordinates - vector, -vector, -maxCoordinate, -minCoordinate))
            closeCoordinates = _removeDuplicates(closeCoordinates)
            if len(closeCoordinates) == N:
                break
            else:
                N = len(closeCoordinates)

        fittedCoordinates = [coord for coord in closeCoordinates if (np.all(0. <= self.cartesianToFractional(coord)) and
                                                                     np.all(self.cartesianToFractional(coord) < 1.))]

        return [Transformation.fromRotVector([0.,0.,0.], finalCoordinates - initialCoordinates)
                for finalCoordinates in fittedCoordinates]

    @staticmethod
    def initFromCellParameters(a, b, c, alpha, beta, gamma, pbc):
        va = np.array([a, 0, 0])
        vb = np.array([b*np.cos(gamma), b*np.sin(gamma), 0])
        cx = np.cos(beta)
        cy = (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
        cz = (1. - cx ** 2 - cy ** 2) ** 0.5
        vc = c * np.array([cx, cy, cz])
        return Cell(np.vstack((va, vb, vc)), pbc)


def _findClose(coordinates, vector, minCoordinate, maxCoordinate):
    coordinate = np.dot(coordinates, vector) / (np.linalg.norm(vector) ** 2)
    initialCoordinate = coordinate
    closeCoordinates = []
    oldDiff = np.inf
    while True:
        diff = min(abs(coordinate - minCoordinate), abs(coordinate - maxCoordinate))
        if minCoordinate <= coordinate <= maxCoordinate:
            closeCoordinates.append(coordinates + vector * (coordinate - initialCoordinate))
        elif diff <= oldDiff:
            oldDiff = diff
        else:
            break
        coordinate += 1
    return closeCoordinates

def _removeDuplicates(coordinates):
    cleanedCoordinates = []
    for probe in coordinates:
        isDuplicate = False
        for ref in cleanedCoordinates:
            if np.allclose(probe, ref):
                isDuplicate = True
                break
        if not isDuplicate:
            cleanedCoordinates.append(probe)
    return cleanedCoordinates