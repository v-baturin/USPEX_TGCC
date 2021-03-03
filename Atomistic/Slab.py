import numpy as np

class Slab:

    def __init__(self, indices, depths, transformations):
        self.indices = np.asarray(indices, dtype=int)
        self.depths = np.asarray(depths, dtype=float)
        self.transformations = np.asarray(transformations)

    @staticmethod
    def getSlabs(molecules, inputCell, outputCell, axis, gaugesOfSlabs, transformation):
        slabs = tuple(([],[],[]) for i in gaugesOfSlabs)
        coordinateBounds = np.cumsum(gaugesOfSlabs)/np.sum(gaugesOfSlabs)
        fittedTransformations = outputCell.getFittedTransformations(inputCell)
        for i, molecule in enumerate(molecules):
            centerOfMassCoordinates = molecule.getCenterOfMassCartesianCoordinates()
            centerOfMassCoordinates = inputCell.getWrapedCartesianCoordinates(centerOfMassCoordinates)
            for fittedTransformation in fittedTransformations:
                compositTransformation = transformation.composition(fittedTransformation)
                centerOfMassCoordinates = compositTransformation.getTransformedCoordinates(centerOfMassCoordinates)
                centerOfMassCoordinates = outputCell.getWrapedCartesianCoordinates(centerOfMassCoordinates)
                coordinate = outputCell.cartesianToFractional(centerOfMassCoordinates)[axis]
                for j, upperBoundCoordinate in enumerate(coordinateBounds):
                    if coordinate <= upperBoundCoordinate:
                        indices, depths, transformations = slabs[j]
                        indices.append(i)
                        depths.append(np.min((upperBoundCoordinate - coordinate, coordinate - coordinateBounds[j-1])))
                        transformations.append(compositTransformation)
                        break
        return (Slab(indices, depths, transformation) for indices, depths, transformation in slabs)

    @staticmethod
    def getRandomSlabs(molecules, inputCell, outputCell, axis, gaugesOfSlabs, order, correlation, parity: int):
        L = inputCell.getAltitudes()[axis]
        Lchar = 0.5 * (inputCell.getVolume() / len(molecules)) ** (1 / 3) # average 'radius' of a molecule in the cell
        N = int(round(L / (Lchar + (L - Lchar) * (np.cos(correlation * np.pi / 2)) ** 2)))
        slabsCandidates = [Slab.getSlabs(molecules, inputCell, outputCell, axis, gaugesOfSlabs,
                                         transformation = inputCell.randomTransformation()) for i in range(N)]
        candidatesCharacteristic = np.argsort(sum(order[slab.indices].sum()*((i+parity)%2) for i, slab in enumerate(slabs))
                                              for slabs in slabsCandidates)
        return slabsCandidates[candidatesCharacteristic[0]] if correlation > 0 else slabsCandidates[candidatesCharacteristic[-1]]
