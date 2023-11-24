"""
USPEX.Atomistic.Slab
====================
"""

import numpy as np


class Slab:

    def __init__(self, indices, depths, molecules):
        self.indices = np.asarray(indices, dtype=int)
        self.depths = np.asarray(depths, dtype=float)
        self.molecules = molecules

    @staticmethod
    def getSlabs(molecules, inputCell, outputCell, axis, gaugesOfSlabs, transformation):
        assert inputCell.getPBC() == outputCell.getPBC()
        if inputCell.dim == 1:
            inputAxis = inputCell.getCellVectorsPBC()
            inputAxis /= np.linalg.norm(inputAxis)
            outputAxis = outputCell.getCellVectorsPBC()
            outputAxis /= np.linalg.norm(outputAxis)
            assert np.allclose(inputAxis, outputAxis)
        elif inputCell.dim == 2:
            inputAxis = inputCell.getCellVectorsAntiPBC()
            inputAxis /= np.linalg.norm(inputAxis)
            outputAxis = outputCell.getCellVectorsAntiPBC()
            outputAxis /= np.linalg.norm(outputAxis)
            assert np.allclose(inputAxis, outputAxis)
        centerShift = (outputCell.getCellVectors() - inputCell.getCellVectors()).sum(axis=0)/2
        inputCell = transformation.transformCell(inputCell)
        slabs = tuple(([],[],[]) for i in gaugesOfSlabs)
        coordinateBounds = np.cumsum(gaugesOfSlabs)/np.sum(gaugesOfSlabs)
        for i, molecule in enumerate(molecules):
            centerOfMassCoordinatesInitial = molecule.getCenterOfMassCartesianCoordinates()
            centerOfMassCoordinates = transformation.transformCoordinates(centerOfMassCoordinatesInitial)
            pbc = outputCell.getPBC()
            dimensionality = np.sum(pbc)
            if dimensionality == 1 and pbc[axis]:
                for fittedTransformation in outputCell.getFittedTransformations(centerOfMassCoordinates, inputCell):
                    coordinates = outputCell.cartesianToFractional(fittedTransformation.transformCoordinates(centerOfMassCoordinates))
                    coordinate = coordinates[axis]
                    for j, upperBoundCoordinate in enumerate(coordinateBounds):
                        if coordinate <= upperBoundCoordinate:
                            lowerBoundCoordinate = 0 if j < 1 else coordinateBounds[j-1]
                            indices, depths, mols = slabs[j]
                            indices.append(i)
                            depths.append(np.min((upperBoundCoordinate - coordinate, coordinate - lowerBoundCoordinate)))
                            mols.append((fittedTransformation * transformation).transform(molecule))
                            break
            else:
                coordinates = inputCell.cartesianToFractional(centerOfMassCoordinates)
                coordinates = inputCell.getWrapedFractionalCoordinates(coordinates)
                inds = np.nonzero(outputCell.getAntiPBC())
                coordinates[inds] = outputCell.cartesianToFractional(centerOfMassCoordinates + centerShift)[inds]
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
        Nmax = inputCell.getMaxNumSlabs(axis, len(molecules))
        N = int(round(Nmax / (1 + (Nmax - 1) * (np.cos(correlation * np.pi / 2)) ** 2)))
        slabsCandidates = [Slab.getSlabs(molecules, inputCell, outputCell, axis, gaugesOfSlabs,
                                         transformation=inputCell.randomTransformation()) for i in range(N)]
        candidatesCharacteristic = np.argsort(sum(order[slab.indices].sum()*((i+parity)%2) for i, slab in enumerate(slabs))
                                              for slabs in slabsCandidates)
        return slabsCandidates[candidatesCharacteristic[0]] if correlation > 0 else slabsCandidates[candidatesCharacteristic[-1]]
