"""
USPEX.Atomistic.EnvironmentUtility
==================================
"""

from shutil import register_unpack_format
from matplotlib.pyplot import axis
import numpy as np
from scipy.linalg import orthogonal_procrustes
from copy import copy
import logging

from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import get_2d_transform, Deformation
from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
from pymatgen.core.surface import SlabGenerator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.transformations.standard_transformations import RotationTransformation

from USPEX.Atomistic.CellUtility import Cell
from USPEX.Atomistic.AtomicPrimitives import AtomicStructure
from USPEX.Atomistic.Element import Element

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
        return Substrate(envStructure, self._thickness, np.min(coordinates, axis=0))
        
    def assemble(self, molecules, cell):
        """
        Assembles the system with environment

        :return: atomTypes, coordinates, assembledCell
        """
        atomTypes = []
        coordinates = []
        for molecule in molecules:
            atomTypes.extend(molecule.getAtomTypes())
            coordinates.extend(molecule.getCartesianCoordinates())
        coordinates = list(np.asarray(coordinates, dtype = float) + self.calculateOffset(molecules, cell))
        assembledCell = self.getStructure().getCell()
        atomTypes.extend(self.getStructure().getAtomTypes())
        coordinates.extend(self.getStructure().getCartesianCoordinates())
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

    # def getUpdatedEnvironment(self, atomTypes, coordinates, cell, indices, envIndices):
    #     sys_allcoords = []
    #     for idx in indices:
    #         sys_allcoords.extend(coordinates[idx])
    #     offsetVector = np.min(sys_allcoords, axis=0)
    #     envStructure = AtomicStructure(atomTypes[envIndices], coordinates[envIndices], cell)
    #     environment = dict(
    #         structure = envStructure,
    #         bufferThickness = self._thickness,
    #         offsetVector = offsetVector,
    #         gap = self._gap,
    #         maxEnvironmentArea = self._maxEnvironmentArea,
    #         maxMisfitStrain = self._maxMisfitStrain
    #     )
    #     return type(self)(**environment)

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
                 maxMisfitStrain: float = None, maxEnvironmentArea: float = None, **kwargs): # TODO: add tiling (maybe)
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
        if molecules is not None and syscell is not None:    
            self._internalOffsetVector = internalOffsetVector
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
        return offsetVector

    def assemble(self, molecules=None, cell=None):
        """
        Assembles the system with environment

        :param molecules: 
        :param cell: 

        :return: atomTypes, coordinates, assembledCell
        """
        atomTypes = []
        coordinates = []
        if molecules is not None:
            for molecule in molecules:
                atomTypes.extend(molecule.getAtomTypes())
                coordinates.extend(molecule.getCartesianCoordinates())
            coordinates = list(np.asarray(coordinates, dtype = float) + self.calculateOffset(molecules, cell))
        for i, structure in enumerate(self._structures):
            atomTypes.extend(structure.getAtomTypes())
            tmpCoords = structure.getCartesianCoordinates()
            if i == 1:
                tmpCoords = list(np.asarray(tmpCoords, dtype = float) + self._calculateInternalOffset(molecules, cell))
            coordinates.extend(tmpCoords)
        assembledCell = self._structures[0].getCell()
        structure = AtomicStructure(atomTypes, coordinates, assembledCell)
        assembledCell = structure.getRectifiedCell().getEnvelopeCell(coordinates, vacuumSize=DEFAULT_VACUUM)
        coordinates = assembledCell.center(structure.getCartesianCoordinates())
        return atomTypes, coordinates, assembledCell

    def getUpdatedEnvironment(self, atomTypes, coordinates, cell, indices, envIndices):
        sys_allcoords = []
        for index in indices:
            sys_allcoords.extend(coordinates[index])
        sys_allcoords = np.asarray(sys_allcoords)
        offsetVector = np.zeros(3)
        internalOffsetVector = np.zeros(3)
        ind = self._inds[0]
        for idx in range(3):
            if idx == ind:
                offsetVector[idx] = sys_allcoords[:, idx].min()
                internalOffsetVector[idx] = sys_allcoords[:, idx].max()
        lowerEnvIndices = envIndices[:len(self.getLowerStructure())]
        upperEnvIndices = envIndices[len(self.getLowerStructure()):]
        newLowerEnvStructure = AtomicStructure(atomTypes[lowerEnvIndices], coordinates[lowerEnvIndices], cell)
        newUpperEnvStructure = AtomicStructure(atomTypes[upperEnvIndices], coordinates[upperEnvIndices] - internalOffsetVector, cell)
        environment = dict(
            lowerStructure = newLowerEnvStructure,
            upperStructure = newUpperEnvStructure,
            bufferThickness = self._thickness, 
            offsetVector = offsetVector,
            internalOffsetVector = internalOffsetVector, 
            gap = self._gap, 
            maxEnvironmentArea = self._maxEnvironmentArea,
            maxMisfitStrain = self._maxMisfitStrain
        )
        return type(self)(**environment)

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
        atomTypes, coordinates, assembledCell = self.assemble()
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

    def __init__(self, structure, isFixed: bool = True):
        self._structure = structure
        self.isFixed = isFixed
        if self.isFixed:
            self._indices = np.arange(len(structure))
        else:
            self._indices = np.array([], dtype=int)

    def calculateOffset(self, molecules, syscell=None):
        return np.array([0.0, 0.0, 0.0])

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
                build = environment.pop('build', False)
                if build:
                    logger.debug(f'Environment build option was enabled')
                    environment = self.buildEnvironment(environment)
                else:
                    pbc = environment.pop('pbc')
                    if environment['type'].lower() == 'substrate':
                        file = environment.pop('file')
                        environment['structure'] = self.structureRepresentation.readAtomicStructureRaw(file, pbc)
                    elif environment['type'].lower() == 'interface':
                        lowerFile = environment.pop('lowerFile')
                        upperFile = environment.pop('upperFile')
                        environment['lowerStructure'] = self.structureRepresentation.readAtomicStructureRaw(lowerFile, pbc)
                        environment['upperStructure'] = self.structureRepresentation.readAtomicStructureRaw(upperFile, pbc)
                self._environments.append(environment)

    def hasEnvironment(self):
        return len(self._environments) > 0

    def getRandomEnvironment(self):
        if self._environments:
            environmentDict = copy(np.random.choice(self._environments))
            envType = environmentDict.pop('type').lower()
            assert envType in ('substrate', 'interface')
            del environmentDict['name']
            if envType == 'substrate':
                environment = Substrate(**environmentDict)
            elif envType == 'interface':
                environment = Interface(**environmentDict)
            return environment

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
            envType = environment.pop('type')
            structure = environment.pop('structure')
            environment['structure'] = structure.makeSupercell(np.round(structure.getCell().decomposeCell(system['cell'])))
            if envType == 'substrate':
                system['environment'] = Substrate(**environment)
            elif envType == 'bulk':
                system['environment'] = Bulk(**environment)
            else:
                raise ValueError(f"Unknown environment type {envType}.")

        # else:
        #     system['environment'] = self.getRandomEnvironment()



    def buildEnvironment(self, environment: dict):
        pbc = tuple(environment.pop('pbc'))
        envType = environment.get('type')
        logger.debug(f'Selected environment type is {envType}')
        if envType == 'interface':
            lowerFile = environment.pop('lowerFile')
            upperFile = environment.pop('upperFile')
            if lowerFile == upperFile and 'sigma' in environment:
                logger.debug(f'Proceeding with Grain Boundary mode')
                initStructure = self.structureRepresentation.readAtomicStructureRaw(lowerFile, pbc=(1,1,1))
                sigma = environment.pop('sigma')
                plane = environment.pop('plane')
                rotAxis = environment.pop('rotAxis')
                slabThickness = environment.pop('slabThickness')
                lowerStructure, upperStructure = constructGrainsSlabs(initStructure, pbc, sigma, plane, rotAxis, slabThickness)
                logger.debug('Grains are successfully created')
            else:
                logger.debug(f'Proceeding with Heterostructure mode')
                initLowerStructure = self.structureRepresentation.readAtomicStructureRaw(lowerFile)
                initUpperStructure = self.structureRepresentation.readAtomicStructureRaw(upperFile)
                lowerPlane = environment.pop('lowerPlane')
                upperPlane = environment.pop('upperPlane')
                slabThickness = environment.pop('slabThickness')
                lowerStructure = constructSurfaceSlab(initLowerStructure, pbc, lowerPlane, slabThickness)
                upperStructure = constructSurfaceSlab(initUpperStructure, pbc, upperPlane, slabThickness)
                logger.debug('Surface Slabs are successfully created')
            adjust = environment.pop('adjust', False)
            if adjust:
                logger.debug(f'Auto-adjustment of interfacial slabs was enabled')
                antiPBC = tuple((~np.asarray(pbc, dtype=bool)).tolist())
                nonPBCAxis = np.flatnonzero(antiPBC)[0]
                maxMisfitStrain = environment.get('maxMisfitStrain', DEFAULT_MAX_MISFIT_STRAIN)
                maxEnvironmentArea = 0.6 * environment.get('maxEnvironmentArea', DEFAULT_MAX_ENVIRONMENT_AREA)
                lowerStructure, upperStructure, _ = adjustStructures(lowerStructure, upperStructure, axis=nonPBCAxis, 
                                                                    maxMisfitStrain=maxMisfitStrain,
                                                                    maxSubstrateArea=maxEnvironmentArea)
                logger.debug(f'Interfacial slabs are successfully adjusted')
            environment['lowerStructure'] = lowerStructure
            environment['upperStructure'] = upperStructure
        elif environment['type'] == 'substrate':
            file = environment.pop('file')
            initStructure = self.structureRepresentation.readAtomicStructureRaw(file)
            plane = environment.pop('plane')
            slabThickness = environment.pop('slabThickness')
            environment['structure'] = constructSurfaceSlab(initStructure, pbc, plane, slabThickness)
        return environment

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

def convertFromPymatgen(pmgStructure, pbc):
    """
    Converts the pymatgen Structure object to the USPEX AtomicStructure object
    """
    cellVectors = pmgStructure.lattice.matrix[np.flatnonzero(pbc)]
    cell = Cell.initFromCellVectors(pbc, cellVectors)
    species = [Element(specie.name) for specie in pmgStructure.species]
    coordinates = pmgStructure.cart_coords
    structure = AtomicStructure(species, coordinates, cell)
    return structure

def convertToPymatgen(structure):
    """
    Converts the USPEX AtomicStructure object to the pymatgen Structure object 
    """
    cellVectors = structure.getRectifiedCell().getCellVectors()
    coordinates = structure.getCartesianCoordinates()
    species = [el.short_name for el in structure.getAtomTypes()]
    pmgStructure = Structure(cellVectors, species, coordinates, coords_are_cartesian=True)
    return pmgStructure

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

def constructSurfaceSlab(structure, pbc, plane, slabThickness, **kwargs):
    """
    Creates a surface slab with given plane Miller indices and slabThickness 
    """
    logger.debug(f'Surface Slab Constructor is initialized')
    pmgStructure = convertToPymatgen(structure)
    logger.debug('Creating slab with parameters:')
    logger.debug(f'Plane: {plane}')
    slabGenerator = SlabGenerator(pmgStructure, miller_index=plane, min_slab_size=slabThickness, 
                                  min_vacuum_size=1e-3, lll_reduce=True, center_slab=True, **kwargs)
    slab = slabGenerator.get_slab()
    slabStructure = convertFromPymatgen(slab, pbc)
    return slabStructure

def constructGrainsSlabs(structure, pbc, sigma, plane, rotAxis, slabThickness, **kwargs):
    """
    Creates two grain slabs with given plane Miller indices, Sigma value and rotation axis
    within CSL Model.
    """
    logger.debug(f'Grains Constructor is initialized')
    antiPBC = tuple((~np.asarray(pbc, dtype=bool)).tolist())
    nonPBCAxis = np.flatnonzero(antiPBC)[0]
    pmgStructure = convertToPymatgen(structure)
    spgAnalzyer = SpacegroupAnalyzer(pmgStructure)
    crystalSystem = spgAnalzyer.get_crystal_system()
    logger.debug(f'Grains crystal system was determined as {crystalSystem}')
    gbGenerator = GrainBoundaryGenerator(initial_structure=pmgStructure)
    angle = min(gbGenerator.get_rotation_angle_from_sigma(sigma, rotAxis, lat_type=crystalSystem[0]))
    logger.debug('Creating graines with parameters:')
    logger.debug(f'Sigma: {sigma}, Angle: {angle:.3f}, Plane: {plane}, Axis: {rotAxis}')
    gb = gbGenerator.gb_from_parameters(rotAxis, angle, plane=plane, expand_times=1, rm_ratio=0.5)
    unitSlabThickness = np.linalg.norm(gb.lattice.matrix[nonPBCAxis]) / 2
    expandTimes = int(np.ceil(slabThickness / unitSlabThickness))
    gb = gbGenerator.gb_from_parameters(rotAxis, angle, plane=plane, expand_times=expandTimes, rm_ratio=0.5)
    assert None not in gb.site_properties['grain_label']
    lowerGrain = SpacegroupAnalyzer(gb.bottom_grain).get_refined_structure()
    upperGrain = SpacegroupAnalyzer(gb.top_grain).get_refined_structure()
    grainThickness = lowerGrain.cart_coords[:, nonPBCAxis].max() - lowerGrain.cart_coords[:, nonPBCAxis].min()
    upperShiftVector = np.zeros(3)
    lowerShiftVector = np.zeros(3)
    for i in range(3):
        if i == nonPBCAxis:
            upperShiftVector[i] = grainThickness / 2 + 1e-5
            lowerShiftVector[i] = - lowerGrain.cart_coords[:, i].min()
    upperGrain.translate_sites(indices=list(range(len(upperGrain))), vector=upperShiftVector, frac_coords=False)
    lowerGrain.translate_sites(indices=list(range(len(lowerGrain))), vector=lowerShiftVector, frac_coords=False)
    lowerGrainStructure = convertFromPymatgen(lowerGrain, pbc)
    upperGrainStructure = convertFromPymatgen(upperGrain, pbc)
    lowerGrainStructure = centerAndEnvelope(lowerGrainStructure)
    upperGrainStructure = centerAndEnvelope(upperGrainStructure)
    return lowerGrainStructure, upperGrainStructure

def centerAndEnvelope(structure):
    cell = structure.getRectifiedCell()
    coordinates = structure.getCartesianCoordinates()
    coordinates = cell.center(coordinates)
    structure._cell = cell
    structure._coordinates = coordinates
    return structure
