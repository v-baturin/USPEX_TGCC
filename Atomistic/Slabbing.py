import numpy as np
from copy import copy

class Slab:

    def __init__(self, indices: list, transformations: list):
        self.indices = indices
        self.transformations = transformations

    @staticmethod
    def getSlabs(molecules, inputCell, outputCell, axis, gaugesOfSlabs, origin, orientation):
        assert inputCell is outputCell # for now we support only this case
        origin = inputCell.wrapVector(origin)
        slabs = tuple(Slab([],[]) for i in gaugesOfSlabs)
        coordinateBounds = np.cumsum(gaugesOfSlabs)/np.sum(gaugesOfSlabs)
        for i, molecule in enumerate(molecules):
            molecule = inputCell.wrapStructure(molecule)
            molecule = molecule.getTransformedStructure(origin, orientation)
            molecule = outputCell.wrapStructure(molecule)
            coordinate = outputCell.cartesianToFractional(molecule.getCenterOfMassCartesianCoordinates())[axis]
            for j, upperBoundCoordinate in enumerate(coordinateBounds):
                if coordinate <= upperBoundCoordinate:
                    offset = molecule.getCenterOfMassCartesianCoordinates() - molecules[i].getCenterOfMassCartesianCoordinates()
                    slabs[j].indices.append(i)
                    slabs[j].transformations.append((offset, orientation))
                    break
        return slabs
