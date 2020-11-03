

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
        pass

    def getVolume(self):
        pass

    def getAltitudes(self):
        pass
        # if dimension == 0:
        #     L = abs(system.get_volume() / np.linalg.norm(np.cross(lat[1, :], lat[2, :])))
        # elif dimension == 1:
        #     L = abs(system.get_volume() / np.linalg.norm(np.cross(lat[0, :], lat[2, :])))
        # else:
        #     L = abs(system.get_volume() / np.linalg.norm(np.cross(lat[0, :], lat[1, :])))


    def cartesianToFractional(self, coordinates):
        pass

    def fractionalToCartesian(self, coordinates):
        pass

    def wrapVector(self, vector):
        pass

    def wrapStructure(self, structure):
        pass

    def randomOrientation(self):
        pass

    def randomOrigin(self):
        pass

    @staticmethod
    def initFromCellParameters(a, b, c, alpha, beta, gama):
        pass
