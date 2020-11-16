import numpy as np
from copy import copy

class Slab:

    def __init__(self, indices: list, transformations: list):
        self.indices = indices
        self.transformations = transformations

    @staticmethod
    def getSlabs(molecules, inputCell, outputCell, axis, gaugesOfSlabs, transformation):
        assert inputCell is outputCell # for now we support only this case
        origin = transformation.origin
        orientation = transformation.oriantation
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

    @staticmethod
    def getRandomSlabs(molecules, inputCell, outputCell, axis, gaugesOfSlabs, order, correlation, parity):
        L = inputCell.getAltitudes()[axis]
        Lchar = 0.5 * (inputCell.getVolume() / len(molecules)) ** (1 / 3) # average 'radius' of a molecule in the cell
        N = int(round(L / (Lchar + (L - Lchar) * (np.cos(correlation * np.pi / 2)) ** 2)))
        slabsCandidates = [Slab.getSlabs(molecules = molecules, inputCell = inputCell, outputCell = outputCell,
                                axis = axis, gaugesOfSlabs = gaugesOfSlabs,
                                transformation = randomTransformation)
                  for randomTransformation in inputCell.randomTransformations(N)]
        candidatesCharacteristic = np.argsort(sum(order[slab.indices].sum()*(i%2+parity) for i, slab in enumerate(slabs))
                                              for slabs in slabsCandidates)
        return slabsCandidates[candidatesCharacteristic[0]] if correlation > 0 else slabsCandidates[candidatesCharacteristic[-1]]
