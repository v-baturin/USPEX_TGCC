import numpy as np


class CellUtility:

    def __init__(self):
        pass

    def getRandomCell(self):
        pass

    def getHybridCell(self, cell1, cell2):
        pass


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
        return abs(np.dot(self._cellVectors[0], np.cross(self._cellVectors[1, :], self._cellVectors[2, :])))

    def getAltitudes(self):
        volime = self.getVolume()
        l0 = volime / np.linalg.norm(np.cross(self._cellVectors()[1, :], self._cellVectors()[2, :]))
        l1 = volime / np.linalg.norm(np.cross(self._cellVectors()[0, :], self._cellVectors()[2, :]))
        l2 = volime / np.linalg.norm(np.cross(self._cellVectors()[0, :], self._cellVectors()[1, :]))
        return np.array([l0, l1, l2])

    def cartesianToFractional(self, coordinates):
        pass

    def fractionalToCartesian(self, coordinates):
        pass

    def getWrapedCartesianCoordinates(self, coordinates):
        return self.fractionalToCartesian(self.getWrapedFractionalCoordinates(self.cartesianToFractional(coordinates)))

    @staticmethod
    def getWrapedFractionalCoordinates(coordinates):
        return np.divmod(coordinates, (1,1,1))[1]

    def randomTransformation(self):
        pass

    @staticmethod
    def initFromCellParameters(a, b, c, alpha, beta, gamma, pbc):
        va = np.array([a, 0, 0])
        vb = np.array([b*np.cos(gamma), b*np.sin(gamma), 0])
        cx = np.cos(beta)
        cy = (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
        cz = (1. - cx ** 2 - cy ** 2) ** 0.5
        vc = c * np.array([cx, cy, cz])
        return Cell(np.vstack((va, vb, vc)), pbc)
