"""
USPEX.Atomistic.EnvironmentUtility
==================================
"""

import numpy as np
from copy import copy
import logging

from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import get_2d_transform, Deformation

from USPEX.Atomistic.AtomicPrimitives import AtomicStructure

logger = logging.getLogger(__name__)

DEFAULT_SUBSTRATE_GAP = 2.0
DEFAULT_MAX_MISFIT_STRAIN = 5E-3
DEFAULT_MAX_ENVIRONMENT_AREA = 1000
DEFAULT_VACUUM = 1e-3

class Substrate:
    """
    Class representing part of structure which is not being altered via variation operators.
    I.e. it acts as environment for individual.
    """

    def __init__(self, structure, bufferThickness: float = None, offsetVector = None, gap: float = None,
                 maxMisfitStrain: float = None, maxEnvironmentArea: float = None, **kwargs): # TODO: add tiling (maybe)
        """
        :param structure: atomic structure associated with this environment.
        :param gap: a gap between the film and substrate
        :param offsetVector: vector to be added to molecules centers when assemble whole structure.
            If not provided such vector will be calculated on demand.
        """
        self._structure = structure
        self._thickness = bufferThickness  # comes from input
        self._offsetVector = offsetVector
        self._gap = gap if gap else DEFAULT_SUBSTRATE_GAP
        self._maxEnvironmentArea = maxEnvironmentArea if maxEnvironmentArea else DEFAULT_MAX_ENVIRONMENT_AREA
        self._maxMisfitStrain = maxMisfitStrain if maxMisfitStrain else DEFAULT_MAX_MISFIT_STRAIN
        antiPBC = self._structure.getCell().getAntiPBC()
        assert sum(antiPBC) == 1
        self._ind = np.flatnonzero(antiPBC)[0]
        coordinates = self._structure.getCartesianCoordinates()[:, self._ind]
        upperBound = coordinates.max() - self._thickness if self._thickness is not None else coordinates.min()
        self._indices = np.flatnonzero(coordinates < upperBound)

    def getThickness(self):
        """
        Get thickness of the substrate.
        """
        return self._thickness

    def calculateOffset(self, molecules, syscell=None):
        """
        Calculate or retrieve vector to be added to each molecule when assemble whole structure.
        If such vector is not predefined for this environment it will be calculated basing on minimal atomic coordinates
        in nonperiodic direction.

        :param molecules: list of molecules for which the offset is being calculated.
        :param cell: TODO

        :return: offset vector.
        """
        if self._offsetVector is not None:
            offsetVector = np.asarray(self._offsetVector, dtype=float)
        else:
            cell = self._structure.getCell()
            frac_coords = AtomicStructure.assemble(molecules, cell)[0].getFractionalCoordinates()
            sysPBC = syscell.getPBC()
            offsetVector = np.zeros(3)
            for idx in range(3):
                curr_axis = cell.getCellVectors()[idx]
                if idx == self._ind:
                    fracEnvCoordinates = cell.cartesianToFractional(self._structure.getCartesianCoordinates())
                    offsetVector += curr_axis * (self._gap / np.linalg.norm(curr_axis)
                                   + fracEnvCoordinates[:, idx].max() - frac_coords[:, idx].min())
                elif not sysPBC[idx]:
                    offsetVector += curr_axis * (0.5 - 0.5 * (frac_coords[:, idx].min() + frac_coords[:, idx].max()))
        return offsetVector

    def getUpdatedEnvironment(self, atomTypes, coordinates, cell, envStructure):
        environment = dict(
            structure = envStructure,
            bufferThickness = self._thickness, 
            offsetVector = np.min(coordinates, axis=0),
            gap = self._gap, 
            maxEnvironmentArea = self._maxEnvironmentArea,
            maxMisfitStrain = self._maxMisfitStrain
        )
        return Substrate(**environment)
        
    def assemble(self, atomTypes, coordinates, cell):
        """
        Assembles the system with environment

        :return: atomTypes, coordinates, assembledCell
        """
        structure = self.getStructure()
        assembledCell = structure.getCell()
        atomTypes.extend(structure.getAtomTypes())
        coordinates.extend(structure.getCartesianCoordinates())
        return atomTypes, coordinates, assembledCell

    def getStructure(self):
        """
        Retrieve atomic structure associated with environment.

        :return: atomic structure.
        """
        return self._structure

    def getFixedIndices(self):
        """
        Get indices of atoms in substrate positions of which are fixed.
        """
        return self._indices

    @classmethod
    def adjustSystem(cls, molecules, cell, environment):
        """
        Adjusts the cells of the environment and the structure to fit each other
        """
        envStructure = environment.getStructure()
        axis = environment._ind
        maxEnvironmentArea = environment._maxEnvironmentArea
        maxMisfitStrain = environment._maxMisfitStrain
        newMolecules, newCell, newEnvStructure = adjustSystem(molecules, cell, envStructure, axis, maxEnvironmentArea, maxMisfitStrain)
        newEnvironmentDict = dict(
            structure = newEnvStructure, 
            bufferThickness = environment._thickness, 
            offsetVector = environment._offsetVector, 
            gap = environment._gap, 
            maxMisfitStrain = environment._maxMisfitStrain,
            maxEnvironmentArea = environment._maxEnvironmentArea
        )
        newEnvironment = type(environment)(**newEnvironmentDict)
        return newMolecules, newCell, newEnvironment


class Interface:
    """
    Class representing part of structure which is not being altered via variation operators.
    I.e. it acts as environment for individual.
    """

    def __init__(self, lowerStructure, upperStructure, bufferThickness: float = None, offsetVector = None, internalOffsetVector = None, gap: float = None,
                 maxMisfitStrain: float = None, maxEnvironmentArea: float = None, **kwargs):
        """
        :param structures: atomic structures associated with this environment.
        :param gap: a gap between the film and substrate
        :param offsetVector: vector to be added to molecules centers when assemble whole structure.
            If not provided such vector will be calculated on demand.
        """
        self._structures = (lowerStructure, upperStructure)
        self._offsetVector = offsetVector
        self._internalOffsetVector = internalOffsetVector
        self._thickness = bufferThickness  # comes from input
        self._gap = gap if gap else DEFAULT_SUBSTRATE_GAP
        self._maxEnvironmentArea = maxEnvironmentArea if maxEnvironmentArea else DEFAULT_MAX_ENVIRONMENT_AREA
        self._maxMisfitStrain = maxMisfitStrain if maxMisfitStrain else DEFAULT_MAX_MISFIT_STRAIN
        self._inds = []
        self._indices = []
        for i, structure in enumerate(self._structures):
            antiPBC = structure.getCell().getAntiPBC()
            assert sum(antiPBC) == 1
            ind = np.flatnonzero(antiPBC)[0]
            coordinates = structure.getCartesianCoordinates()[:, ind]
            if i == 0:
                upperBound = coordinates.max() - self._thickness if self._thickness is not None else coordinates.min()
                indices = np.flatnonzero(coordinates < upperBound)
            elif i == 1:
                lowerBound = coordinates.min() + self._thickness if self._thickness is not None else coordinates.max()
                indices = np.flatnonzero(coordinates > lowerBound)
            self._inds.append(ind)
            self._indices.append(indices)

    def getThickness(self):
        """
        Get thickness of the substrate.
        """
        return self._thickness

    def _calculateInternalOffset(self, molecules=None, syscell=None):
        """
        Calculate the internal offset vector of the upper part of the interface 
        w.r.t. the lower part. 

        :param molecules: 
        :param syscell: 

        """
        if self._internalOffsetVector is not None:
            internalOffsetVector = np.asarray(self._internalOffsetVector, dtype=float)
        else:
            lowerStructure, ind = self._structures[0], self._inds[0]
            upperStructure = self._structures[1]
            cell = lowerStructure.getCell()
            internalOffsetVector = np.zeros(3)
            if molecules is None and syscell is None:
                thickness = self._thickness if self._thickness else self._gap * 2.0
            else:
                coords = AtomicStructure.assemble(molecules, cell)[0].getCartesianCoordinates()
                thickness = coords[:, ind].max() - coords[:, ind].min()
            for idx in range(3):
                curr_axis = cell.getCellVectors()[idx]
                if idx == ind:
                    fracLowerEnvCoordinates = cell.cartesianToFractional(lowerStructure.getCartesianCoordinates())
                    fracUpperEnvCoordinates = cell.cartesianToFractional(upperStructure.getCartesianCoordinates())
                    lowerEnvShift = fracLowerEnvCoordinates[:, idx].max()
                    upperEnvShift = fracUpperEnvCoordinates[:, idx].min()
                    internalOffsetVector += curr_axis * ((self._gap * 2.0 + thickness) / np.linalg.norm(curr_axis) + lowerEnvShift - upperEnvShift)
        return internalOffsetVector

    def calculateOffset(self, molecules, syscell=None):
        """
        Calculate or retrieve vector to be added to each molecule when assemble whole structure.
        If such vector is not predefined for this environment it will be calculated basing on minimal atomic coordinates
        in nonperiodic direction.

        :param molecules: 
        :param cell: 

        :return: offset vector.
        """
        if self._offsetVector is not None:
            offsetVector = np.asarray(self._offsetVector, dtype=float)
        else:
            structure, ind = self._structures[0], self._inds[0]
            cell = structure.getCell()
            frac_coords = AtomicStructure.assemble(molecules, cell)[0].getFractionalCoordinates()
            sysPBC = syscell.getPBC()
            offsetVector = np.zeros(3)
            for idx in range(3):
                curr_axis = cell.getCellVectors()[idx]
                if idx == ind:
                    fracEnvCoordinates = cell.cartesianToFractional(structure.getCartesianCoordinates())
                    offsetVector += curr_axis * (self._gap / np.linalg.norm(curr_axis)
                                   + fracEnvCoordinates[:, idx].max() - frac_coords[:, idx].min())
                elif not sysPBC[idx]:
                    offsetVector += curr_axis * (0.5 - 0.5 * (frac_coords[:, idx].min() + frac_coords[:, idx].max()))
        self._internalOffsetVector = self._calculateInternalOffset(molecules, syscell)
        return offsetVector

    def assemble(self, atomTypes, coordinates, cell):
        """
        Assembles the system with environment

        :return: atomTypes, coordinates, assembledCell
        """
        for i, structure in enumerate(self._structures):
            atomTypes.extend(structure.getAtomTypes())
            tmpCoords = structure.getCartesianCoordinates()
            if i == 1:
                tmpCoords = list(np.asarray(tmpCoords, dtype = float) + self._internalOffsetVector)
            coordinates.extend(tmpCoords)
        assembledCell = self._structures[0].getCell()
        structure = AtomicStructure(atomTypes, coordinates, assembledCell)
        assembledCell = structure.getRectifiedCell().getEnvelopeCell(coordinates, vacuumSize=DEFAULT_VACUUM)
        coordinates = assembledCell.center(structure.getCartesianCoordinates())
        return atomTypes, coordinates, assembledCell

    def getUpdatedEnvironment(self, atomTypes, coordinates, cell, envStructure):
        envAtomTypes = envStructure.getAtomTypes()
        envCoordinates = envStructure.getCartesianCoordinates()
        envCell = envStructure.getCell()
        offsetVector = np.zeros(3)
        internalOffsetVector = np.zeros(3)
        ind = self._inds[0]
        for idx in range(3):
            if idx == ind:
                offsetVector[idx] = coordinates[:, idx].min()
                internalOffsetVector[idx] = coordinates[:, idx].max()
        lowerEnvIndices = list(range(len(self.getLowerStructure())))
        upperEnvIndices = list(range(len(self.getLowerStructure()), len(self.getLowerStructure()) + len(self.getUpperStructure())))
        lowerEnvStructure = AtomicStructure(envAtomTypes[lowerEnvIndices], envCoordinates[lowerEnvIndices], envCell)
        upperEnvStructure = AtomicStructure(envAtomTypes[upperEnvIndices], envCoordinates[upperEnvIndices] - internalOffsetVector, envCell)
        environment = dict(
            lowerStructure = lowerEnvStructure,
            upperStructure = upperEnvStructure,
            bufferThickness = self._thickness, 
            offsetVector = offsetVector,
            internalOffsetVector = internalOffsetVector, 
            gap = self._gap, 
            maxEnvironmentArea = self._maxEnvironmentArea,
            maxMisfitStrain = self._maxMisfitStrain
        )
        return Interface(**environment)

    def getLowerStructure(self):
        """
        Retrieve the atomic structure associated with lower part of the environment.

        :return: atomic structure.
        """
        structure = self._structures[0]
        coordinates = structure.getCartesianCoordinates()
        cell = structure.getRectifiedCell().getEnvelopeCell(coordinates, DEFAULT_VACUUM)
        coordinates = cell.center(coordinates)
        return AtomicStructure(structure.getAtomTypes(), coordinates, cell)

    def getUpperStructure(self):
        """
        Retrieve the atomic structure associated with upper part of the environment.

        :return: atomic structure.
        """
        structure = self._structures[1]
        coordinates = structure.getCartesianCoordinates()
        cell = structure.getRectifiedCell().getEnvelopeCell(coordinates, DEFAULT_VACUUM)
        coordinates = cell.center(coordinates)
        return AtomicStructure(structure.getAtomTypes(), coordinates, cell)   

    def getStructure(self):
        """
        Combines the two parts of interface in a single structure and returns it.

        :return: atomic structure.
        """
        atomTypes, coordinates, assembledCell = self.assemble(atomTypes=[], coordinates=[], cell=None)
        return AtomicStructure(atomTypes, coordinates, assembledCell)

    @classmethod
    def adjustSystem(cls, molecules, cell, environment):
        """
        Adjusts the cells of the environment and the structure to fit each other
        """
        maxMisfitStrain = environment._maxMisfitStrain
        maxEnvironmentArea = environment._maxEnvironmentArea
        lowerEnvStructure, upperEnvStructure = environment._structures
        lowerAxis, upperAxis = environment._inds
        lowerCell = lowerEnvStructure.getCell()
        upperCell = upperEnvStructure.getCell()
        logger.debug('Starting adjustment of the Interface')

        envCellsAreClose = lowerCell.isClose(upperCell)
        firstStageMaxArea = maxEnvironmentArea if envCellsAreClose else maxEnvironmentArea * 0.6

        # Stage 1. We adjust initial molecules and cell of our system to the lower part of the environment
        newMolecules, newCell, newLowerEnvStructure, supercellMatrices = adjustSystem(molecules, cell,
                                                                                      lowerEnvStructure, lowerAxis,
                                                                                      firstStageMaxArea,
                                                                                      maxMisfitStrain,
                                                                                      returnSupercellMatrices=True)
        logger.debug('Adjustment Stage 1 is done')

        if envCellsAreClose:
            logger.debug('Stage 2 is skipped due to proximity of lower and upper environment cells.')
            # Stage 3. We finally adjust the upper part of the environment to fit the lower part from stage 1
            envSupercellMatrix = supercellMatrices[0]
            newUpperEnvStructure = upperEnvStructure.makeSupercell(envSupercellMatrix)
            newUpperEnvStructure = alignStructure(newUpperEnvStructure, newLowerEnvStructure)
        else:
            # Stage 2. We adjuct the molecules and cell from the previous stage to fit the upper part of the environment
            newMolecules, newCell, newUpperEnvStructure, supercellMatrices = adjustSystem(newMolecules, newCell,
                                                                                          upperEnvStructure, upperAxis,
                                                                                          maxEnvironmentArea,
                                                                                          maxMisfitStrain,
                                                                                          returnSupercellMatrices=True)
            logger.debug('Adjustment Stage 2 is done')
            # Stage 3. We finally adjust the lower part of the environment from stage 1 to fit the new upper part from stage 2
            envSupercellMatrix = supercellMatrices[1]
            newLowerEnvStructure = newLowerEnvStructure.makeSupercell(envSupercellMatrix)
            newLowerEnvStructure = alignStructure(newLowerEnvStructure, newUpperEnvStructure)

        logger.debug('Adjustment Stage 3 is done')
        logger.debug('Adjustment of the Interface is finished')

        assert len(newLowerEnvStructure) % len(lowerEnvStructure) == 0
        assert len(newUpperEnvStructure) % len(upperEnvStructure) == 0

        newEnvironmentDict = dict(
            lowerStructure=newLowerEnvStructure,
            upperStructure=newUpperEnvStructure,
            bufferThickness=environment._thickness,
            offsetVector=environment._offsetVector,
            internalOffsetVector=environment._internalOffsetVector,
            gap=environment._gap,
            maxEnvironmentArea=environment._maxEnvironmentArea,
            maxMisfitStrain=environment._maxMisfitStrain
        )

        newEnvironment = type(environment)(**newEnvironmentDict)
        return newMolecules, newCell, newEnvironment


class Bulk:

    def __init__(self, structure, isFixed: bool = True, **kwargs):
        self._structure = structure
        self.isFixed = isFixed
        if self.isFixed:
            self._indices = np.arange(len(structure))
        else:
            self._indices = np.array([], dtype=int)

    def calculateOffset(self, molecules, syscell=None):
        return np.array([0.0, 0.0, 0.0])

    def assemble(self, atomTypes, coordinates, cell):
        """
        Assembles the system with environment

        :return: atomTypes, coordinates, assembledCell
        """
        structure = self.getStructure()
        assembledCell = structure.getCell()
        atomTypes.extend(structure.getAtomTypes())
        coordinates.extend(structure.getCartesianCoordinates())
        return atomTypes, coordinates, assembledCell

    def getUpdatedEnvironment(self, atomTypes, coordinates, cell, envStructure):
        return Bulk(envStructure, self.isFixed)

    def getStructure(self):
        """
        Retrieve atomic structure associated with environment.

        :return: atomic structure.
        """
        return self._structure

    def getFixedIndices(self):
        """
        Get indices of atoms in substrate positions of which are fixed.
        """
        return self._indices

class EnvironmentUtility:
    """
    Class representing utility which generates possible environmemnts for calculation.
    """
    structureRepresentation = None
    supportedEnvironments = {
        'interface': Interface,
        'substrate': Substrate,
        'bulk': Bulk
    }

    @classmethod
    def setRepresentation(cls, representation):
        cls.structureRepresentation = representation

    def __init__(self, environments: list = None):
        """

        :param files: files with structures for possile environments.
        :param pbc: periodic boundary conditions of environment structures.

        """
        self._environments = []
        if environments is not None:
            for environment in environments:
                pbc = environment.pop('pbc')
                if environment['type'].lower() == 'substrate' or environment['type'].lower() == 'bulk':
                    file = environment.pop('file')
                    environment.setdefault('structure', self.structureRepresentation.readAtomicStructureRaw(file, pbc))
                elif environment['type'].lower() == 'interface':
                    lowerFile = environment.pop('lowerFile')
                    upperFile = environment.pop('upperFile')
                    environment.setdefault('lowerStructure', self.structureRepresentation.readAtomicStructureRaw(lowerFile, pbc))
                    environment.setdefault('upperStructure', self.structureRepresentation.readAtomicStructureRaw(upperFile, pbc))
                self._environments.append(environment)

    def hasEnvironment(self):
        return len(self._environments) > 0

    def putEnvironment(self, system, environment=None):
        """
        Put environment in dictionary representing system.

        :param system: system dictionary.

        """
        if environment is not None:
            system['environment'] = environment
        elif self._environments:
            environment = copy(np.random.choice(self._environments))
            name = environment.pop('name')
            envType = environment.pop('type').lower()
            envConstructor = EnvironmentUtility.supportedEnvironments.get(envType)
            if envConstructor is not None:
                system['environment'] = envConstructor(**environment)
            else:
                raise ValueError(f"Unknown environment type {envType}.")


def adjustSystem(molecules, cell, envStructure, axis, maxSubstrateArea, maxMisfitStrain, returnSupercellMatrices=False):
    """
    Adjusts the cells of the environment and the structure to make them fit each other
    """
    logger.debug(f'Got the system with {len(molecules)} atoms')
    logger.debug(f'Got the envStructure with {len(envStructure)} atoms')
    structure, disassembler = AtomicStructure.assemble(molecules, cell)
    newEnvStructure, newStructure, supercellMatrices = adjustStructures(envStructure, structure, axis=axis, 
                                                    maxMisfitStrain=maxMisfitStrain, maxSubstrateArea=maxSubstrateArea)
    disassembler = type(disassembler)(indices=[[i] for i in range(len(newStructure))], cell=newStructure.getCell(), environment=None)
    newSystem = disassembler.disassemble(newStructure)
    newMolecules, newCell = newSystem['molecules'], newSystem['cell']
    logger.debug(f'Got the final system with {len(newMolecules)} atoms')
    logger.debug(f'Got the final newEnvStructure with {len(newEnvStructure)} atoms')
    if returnSupercellMatrices:
        return newMolecules, newCell, newEnvStructure, supercellMatrices
    else:
        return newMolecules, newCell, newEnvStructure

def adjustStructures(lowerStructure, upperStructure, axis, maxSubstrateArea, maxMisfitStrain):
    """
    Adjusts unit cells of two structures along the given axis to make them fit each other
    """
    lowerCellVectors = lowerStructure.getCell().getCellVectors()
    upperCellVectors = upperStructure.getCell().getCellVectors()
    lowerSupercellMatrix, upperSupercellMatrix = calculateSupercellMatrices(lowerCellVectors, upperCellVectors, axis=axis, 
                                                                     maxMisfitStrain=maxMisfitStrain, maxSubstrateArea=maxSubstrateArea)
    logger.debug(f'Supercell Matrices: {lowerSupercellMatrix} (lower), {upperSupercellMatrix} (upper)')
    newLowerStructure = lowerStructure.makeSupercell(lowerSupercellMatrix)
    newUpperStructure = upperStructure.makeSupercell(upperSupercellMatrix)
    newUpperStructure = alignStructure(newUpperStructure, newLowerStructure)
    assert len(newLowerStructure) % len(lowerStructure) == 0
    assert len(newUpperStructure) % len(upperStructure) == 0
    supercellMatrices = (lowerSupercellMatrix, upperSupercellMatrix)
    return newLowerStructure, newUpperStructure, supercellMatrices

def calculateSupercellMatrices(filmCellVectors, substrateCellVectors, axis, maxMisfitStrain, maxSubstrateArea):
    """
    Calculates 2D supercell matrices with non-periodic specified axis for both film and substrate 
    to make them fit each other in terms of the minimal resulting supercell misfit strain
    """
    logger.debug('Starting the calculation of supercell matrices')
    generator = ZSLGenerator(max_area=maxSubstrateArea)
    logger.debug('ZSLGenerator initialized')
    filmPlaneCellVectors = np.delete(filmCellVectors, axis, axis=0)
    substratePlaneCellVectors = np.delete(substrateCellVectors, axis, axis=0)
    matches = list(generator(filmPlaneCellVectors, substratePlaneCellVectors))
    logger.debug(f'Found {len(matches)} matches for current maxArea of {maxSubstrateArea}')
    if len(matches) > 0:
        metrics, matrices = [], [] 
        for match in matches:
            M1 = np.round(get_2d_transform(filmPlaneCellVectors, match.film_sl_vectors)).astype(int)
            M2 = np.round(get_2d_transform(substratePlaneCellVectors, match.substrate_sl_vectors)).astype(int)
            strain = Deformation(match.match_transformation).green_lagrange_strain
            metrics.append([np.max(strain), match.match_area])
            matrices.append([M1, M2])
        metrics = np.array(metrics)
        matrices = np.array(matrices)
        if len(matrices) > 0:
            goodStrainIndices = np.where(metrics[:, 0] <= maxMisfitStrain)[0]
            logger.debug(f'{len(goodStrainIndices)} matches satisfy maxMisfitStrain of {maxMisfitStrain:.3e} ')
            if len(goodStrainIndices) > 0:
                goodMatricesIndex = metrics[goodStrainIndices][:, 1].argmin()
                goodMetrics = metrics[goodStrainIndices][goodMatricesIndex]
                M1, M2 = matrices[goodStrainIndices][goodMatricesIndex]
                filmSupercellMatrix = convert2DMatrixTo3D(M1, axis=axis)
                substrateSupercellMatrix = convert2DMatrixTo3D(M2, axis=axis)
                logger.debug(f'Successfully found supercell matrices')
                logger.debug(f'Misfit strain: {goodMetrics[0]:.4e}, Area: {int(goodMetrics[1])}')
                logger.debug('Calculation of supercell matrices is finished')
                return filmSupercellMatrix, substrateSupercellMatrix
    else:
        raise RuntimeError('ZSLGenerator failed.')

def alignStructure(structure, targetStructure):
    """
    Transforms the cellVectors of a given structure to map them closely to 
    cellVectors of the targetStructure.
    """
    cell = structure.getCell()
    targetCell = targetStructure.getCell()
    newCell = cell.getOrthogonallyTransformedCell(targetCell)
    newStructure = AtomicStructure.initFromFractionalCoordinates(structure.getAtomTypes(), 
                                                                structure.getFractionalCoordinates(), newCell)
    return newStructure

def convert2DMatrixTo3D(matrix, axis):
    """
    Expands 2D matrix to 3D by filling the diagonal element on the specified axis with 1.0
    and offdiagonal elements with zeros
    """
    M = np.diag((1.0, 1.0, 1.0))
    rows = np.delete(np.arange(3), axis)
    M[np.ix_(rows, rows)] = matrix
    return M

def convert3DMatrixTo2D(matrix, axis):
    """
    Crops 3D matrix and makes 2D one by deleting the row and column corresponding to the specified axis
    """
    rows = np.delete(np.arange(3), axis)
    return matrix[np.ix_(rows, rows)]
