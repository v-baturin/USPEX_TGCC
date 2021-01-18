import numpy as np


class CellProcessing:

    def __init__(self):
        pass

    def getRandomCell(self):
        pass

    def getHybridCell(self, cell1, cell2):
        pass


class Cell:

    def __init__(self, cellVectors, pbc):
        self.cellVectors = cellVectors
        self.pbc = pbc

    def getCellVectors(self):
        return self.cellVectors

    def getPBC(self):
        return self.pbc

    def getCellParameters(self):
        a = np.linalg.norm(self.cellVectors[0, :])
        b = np.linalg.norm(self.cellVectors[1, :])
        c = np.linalg.norm(self.cellVectors[2, :])
        alpha = 180 / np.pi * np.arccos(np.dot(self.cellVectors[1, :], self.cellVectors[2, :]) / (b * c))
        beta = 180 / np.pi * np.arccos(np.dot(self.cellVectors[0, :], self.cellVectors[2, :]) / (a * c))
        gamma = 180 / np.pi * np.arccos(np.dot(self.cellVectors[0, :], self.cellVectors[1, :]) / (a * b))
        return a, b, c, alpha, beta, gamma

    def getVolume(self): 
        return abs(np.dot(self.cellVectors[0], np.cross(self.cellVectors[1, :], self.cellVectors[2, :])))

    def getAltitudes(self):
        l0 = self.getVolume() / np.linalg.norm(np.cross(self.getCellVectors()[1, :], self.getCellVectors()[2, :]))
        l1 = self.getVolume() / np.linalg.norm(np.cross(self.getCellVectors()[0, :], self.getCellVectors()[2, :]))
        l2 = self.getVolume() / np.linalg.norm(np.cross(self.getCellVectors()[0, :], self.getCellVectors()[1, :]))
        return np.array([l0, l1, l2])

    def cartesianToFractional(self, coordinates):
        pass

    def fractionalToCartesian(self, coordinates):
        pass

    def wrapVector(self, vector):
        # vector is in fractional format
        while vector.any() > 1:
            vector[vector > 1] -= 1
        while vector.any() < -1:
            vector[vector < -1] += 1

    def wrapStructure(self, structure):
        return self.wrapVector(structure)

    def randomOrientation(self):
        pass

    def randomOrigin(self):
        pass

    @staticmethod
    def initFromCellParameters(a, b, c, alpha, beta, gamma):
        va = np.array([a, 0, 0])
        vb = np.array([b*np.cos(gamma), b*np.sin(gamma), 0])
        cx = np.cos(beta)
        cy = (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
        cz = (1. - cx ** 2 - cy ** 2) ** 0.5
        vc = np.array([cx, cy, cz])
        return np.vstack((va, vb, vc))
