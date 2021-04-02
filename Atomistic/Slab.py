import numpy as np

class Slab:

    def __init__(self, indices, depths, molecules):
        self.indices = np.asarray(indices, dtype=int)
        self.depths = np.asarray(depths, dtype=float)
        self.molecules = molecules

    @staticmethod
    def getSlabs(molecules, inputCell, outputCell, axis, gaugesOfSlabs, transformation):
        cellTransformation = type(transformation).fromMatrix(transformation.rotMatrix, [0.,0.,0.])
        inputCell = cellTransformation.transformCell(inputCell)
        slabs = tuple(([],[],[]) for i in gaugesOfSlabs)
        coordinateBounds = np.cumsum(gaugesOfSlabs)/np.sum(gaugesOfSlabs)
        for i, molecule in enumerate(molecules):
            centerOfMassCoordinatesInitial = molecule.getCenterOfMassCartesianCoordinates()
            centerOfMassCoordinates = transformation.getTransformedCoordinates(centerOfMassCoordinatesInitial)
            # for fittedTransformation in outputCell.getFittedTransformations(centerOfMassCoordinates, inputCell):
            #     coordinates = outputCell.cartesianToFractional(fittedTransformation.getTransformedCoordinates(centerOfMassCoordinates))
            #     assert np.all(0. <= coordinates) and np.all(coordinates < 1.)
            #     coordinate = coordinates[axis]
            #     for j, upperBoundCoordinate in enumerate(coordinateBounds):
            #         if coordinate <= upperBoundCoordinate:
            #             lowerBoundCoordinate = 0 if j < 1 else coordinateBounds[j-1]
            #             indices, depths, mols = slabs[j]
            #             indices.append(i)
            #             depths.append(np.min((upperBoundCoordinate - coordinate, coordinate - lowerBoundCoordinate)))
            #             mols.append((fittedTransformation * transformation).transform(molecule))
            #             break
            coordinates = inputCell.cartesianToFractional(centerOfMassCoordinates)
            coordinates = inputCell.getWrapedFractionalCoordinates(coordinates)
            coordinate = coordinates[axis]
            for j, upperBoundCoordinate in enumerate(coordinateBounds):
                if coordinate <= upperBoundCoordinate:
                    lowerBoundCoordinate = 0 if j < 1 else coordinateBounds[j-1]
                    indices, depths, mols = slabs[j]
                    indices.append(i)
                    depths.append(np.min((upperBoundCoordinate - coordinate, coordinate - lowerBoundCoordinate)))
                    transVector = outputCell.fractionalToCartesian(coordinates) - \
                                  np.dot(transformation.rotMatrix, centerOfMassCoordinatesInitial)
                    finalTransformation = type(transformation).fromMatrix(transformation.rotMatrix, transVector)
                    mols.append(finalTransformation.transform(molecule))
                    break
        return (Slab(indices, depths, molecules) for indices, depths, molecules in slabs)

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
