"""
USPEX.Atomistic.EnvironmentUtility
==================================
"""

import numpy as np
import logging

from pymatgen.analysis.interfaces.zsl import ZSLGenerator
from pymatgen.analysis.interfaces.coherent_interfaces import get_2d_transform, Deformation

from pymatgen.analysis.gb.grain import GrainBoundaryGenerator
from pymatgen.core.surface import SlabGenerator
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import alphashape

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

    class Assembler:

        def __init__(self, structure, bufferThickness: float = None, gap: float = None,
                     maxMisfitStrain: float = None, maxEnvironmentArea: float = None, **kwargs):
            self._structure = structure
            self._thickness = bufferThickness  # comes from input
            self._gap = gap if gap else DEFAULT_SUBSTRATE_GAP
            self.maxEnvironmentArea = maxEnvironmentArea if maxEnvironmentArea else DEFAULT_MAX_ENVIRONMENT_AREA
            self.maxMisfitStrain = maxMisfitStrain if maxMisfitStrain else DEFAULT_MAX_MISFIT_STRAIN
            antiPBC = self._structure.getCell().getAntiPBC()
            assert sum(antiPBC) == 1
            self._axis = np.flatnonzero(antiPBC)[0]

        def getCell(self):
            return self._structure.getCell()

        def assemble(self, molecules, cell, structure=None, **kwargs):
            structure = structure if structure is not None else self._structure
            sysStructure, disassembler = EnvironmentUtility.atomicDisassemblerType.assemble(molecules, cell)
            intermediateStructure = structure.makeSupercell(structure.getCell().decomposeCell(sysStructure.getCell()))
            envCell = intermediateStructure.getCell()
            fracCoordinates = envCell.cartesianToFractional(sysStructure.getCartesianCoordinates())
            envCoordinates = intermediateStructure.getCartesianCoordinates()
            fracEnvCoordinates = envCell.cartesianToFractional(envCoordinates)
            offset = np.zeros(3)
            for idx in range(3):
                currAxis = envCell.getCellVectors()[idx]
                if idx == self._axis:
                    offset += currAxis * (self._gap / np.linalg.norm(currAxis)
                                          + fracEnvCoordinates[:, idx].max() - fracCoordinates[:, idx].min())
                elif not cell.getPBC()[idx]:
                    offset += currAxis * (1 - (fracCoordinates[:, idx].min() + fracCoordinates[:, idx].max())) / 2

            finalStructure = EnvironmentUtility.structureType(intermediateStructure.getAtomTypes(),
                                                              envCoordinates - offset, envCell)
            coordinates = finalStructure.getCartesianCoordinates()[:, self._axis]
            upperBound = coordinates.max() - self._thickness if self._thickness is not None else coordinates.min()
            indices = np.flatnonzero(coordinates < upperBound)
            return Substrate(finalStructure, indices, self)

        @staticmethod
        def build(file, pbc, build=False, plane=None, slabThickness=None, **kwargs):
            """
            Builds the environment objects for a given description.
            """
            structure = EnvironmentUtility.structureRepresentation.readPOSCAR(file, pbc)
            if build:
                structure = constructSurfaceSlab(structure, pbc, plane, slabThickness)
            environment = dict(
                structure=structure,
            )
            return environment

    processingStyles = {
        'onlyEnvironment': 'getStructure'
    }

    def __init__(self, structure, indices, assembler):
        """
        :param structure: atomic structure associated with this environment.
        :param gap: a gap between the film and substrate
        :param offsetVector: vector to be added to molecules centers when assemble whole structure.
            If not provided such vector will be calculated on demand.
        """
        self._structure = structure
        self._indices = indices
        self._assembler = assembler

    @staticmethod
    def fromIndices(structure, all, fixed, pbc):
        all = np.asarray(all, dtype=int)
        fixed = np.where(np.in1d(all, fixed))[0]
        envStructure = EnvironmentUtility.structureType(structure.getAtomTypes()[all],
                                          structure.getCartesianCoordinates()[all],
                                          EnvironmentUtility.cellType(structure.getCell().getCellVectors(), pbc=pbc))
        return Substrate(envStructure, fixed, None), all

    def getUpdatedEnvironment(self, envStructure):
        return Substrate(envStructure, self._indices, self._assembler)
        
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

    def adjustSystem(self, molecules, cell):
        """
        Adjusts the cells of the environment and the structure to fit each other
        """
        antiPBC = self._structure.getCell().getAntiPBC()
        assert sum(antiPBC) == 1
        axis = np.flatnonzero(antiPBC)[0]
        maxEnvironmentArea = self._assembler.maxEnvironmentArea
        maxMisfitStrain = self._assembler.maxMisfitStrain
        newMolecules, newCell, newEnvStructure = adjustSystem(molecules, cell, self._structure, axis, maxEnvironmentArea, maxMisfitStrain)
        newEnvironment = self._assembler.assemble(molecules, cell, structure=newEnvStructure)
        return newMolecules, newCell, newEnvironment


class Interface:
    """
    Class representing part of structure which is not being altered via variation operators.
    I.e. it acts as environment for individual.
    """

    class Assembler:

        def __init__(self, lowerStructure, upperStructure, bufferThickness: float = None, gap: float = None,
                     maxMisfitStrain: float = None, maxEnvironmentArea: float = None, **kwargs):
            """
            :param structures: atomic structures associated with this environment.
            :param gap: a gap between the film and substrate
            :param offsetVector: vector to be added to molecules centers when assemble whole structure.
                If not provided such vector will be calculated on demand.
            """
            self._lowerStructure = lowerStructure
            self._upperStructure = upperStructure
            self._structures = (lowerStructure, upperStructure)
            self._thickness = bufferThickness  # comes from input
            self._gap = gap if gap else DEFAULT_SUBSTRATE_GAP
            self.maxEnvironmentArea = maxEnvironmentArea if maxEnvironmentArea else DEFAULT_MAX_ENVIRONMENT_AREA
            self.maxMisfitStrain = maxMisfitStrain if maxMisfitStrain else DEFAULT_MAX_MISFIT_STRAIN
            antiPBC = lowerStructure.getCell().getAntiPBC()
            assert sum(antiPBC) == 1
            lowerAxis = np.flatnonzero(antiPBC)[0]
            antiPBC = upperStructure.getCell().getAntiPBC()
            assert sum(antiPBC) == 1
            upperAxis = np.flatnonzero(antiPBC)[0]
            assert lowerAxis == upperAxis
            self._axis = lowerAxis

        def getCell(self):
            return None

        def _calculateUpperOffset(self, coords, sysPBC=None):
            """
            Calculate the internal offset vector of the upper part of the interface
            w.r.t. the lower part.

            :param molecules:
            :param syscell:

            """
            cell = self._lowerStructure.getCell()
            internalOffsetVector = np.zeros(3)
            thickness = coords[:, self._axis].max() - coords[:, self._axis].min()
            for idx in range(3):
                curr_axis = cell.getCellVectors()[idx]
                if idx == self._axis:
                    fracLowerEnvCoordinates = cell.cartesianToFractional(self._lowerStructure.getCartesianCoordinates())
                    fracUpperEnvCoordinates = cell.cartesianToFractional(self._upperStructure.getCartesianCoordinates())
                    lowerEnvShift = fracLowerEnvCoordinates[:, idx].max()
                    upperEnvShift = fracUpperEnvCoordinates[:, idx].min()
                    internalOffsetVector += curr_axis * ((self._gap * 2.0 + thickness) / np.linalg.norm(
                        curr_axis) + lowerEnvShift - upperEnvShift)
            return internalOffsetVector

        def _calculateLowerOffset(self, frac_coords, sysPBC=None):
            """
            Calculate or retrieve vector to be added to each molecule when assemble whole structure.
            If such vector is not predefined for this environment it will be calculated basing on minimal atomic coordinates
            in nonperiodic direction.

            :param molecules:
            :param cell:

            :return: offset vector.
            """
            cell = self._lowerStructure.getCell()
            offsetVector = np.zeros(3)
            for idx in range(3):
                curr_axis = cell.getCellVectors()[idx]
                if idx == self._axis:
                    fracEnvCoordinates = cell.cartesianToFractional(self._lowerStructure.getCartesianCoordinates())
                    offsetVector += curr_axis * (self._gap / np.linalg.norm(curr_axis)
                                                 + fracEnvCoordinates[:, idx].max() - frac_coords[:, idx].min())
                elif not sysPBC[idx]:
                    offsetVector += curr_axis * (0.5 - 0.5 * (frac_coords[:, idx].min() + frac_coords[:, idx].max()))
            return offsetVector

        def assemble(self, molecules, cell, lowerStructure=None, upperStructure=None, **kwargs):
            """
            Assembles the system with environment

            :return: atomTypes, coordinates, assembledCell
            """
            lowerStructure = lowerStructure if lowerStructure is not None else self._structures[0]
            upperStructure = upperStructure if upperStructure is not None else self._structures[1]
            structure, disassembler = EnvironmentUtility.atomicDisassemblerType.assemble(molecules, cell)
            lowerOffset = self._calculateLowerOffset(structure.getFractionalCoordinates(), cell.getPBC())
            upperOffset = self._calculateUpperOffset(structure.getCartesianCoordinates(), cell.getPBC()) - lowerOffset
            lowerStructure = EnvironmentUtility.structureType(lowerStructure.getAtomTypes(),
                                                              lowerStructure.getCartesianCoordinates() - lowerOffset,
                                                              lowerStructure.getCell())
            upperStructure = EnvironmentUtility.structureType(upperStructure.getAtomTypes(),
                                                              upperStructure.getCartesianCoordinates() + upperOffset,
                                                              upperStructure.getCell())
            coordinates = lowerStructure.getCartesianCoordinates()[:, self._axis]
            upperBound = coordinates.max() - self._thickness if self._thickness is not None else coordinates.min()
            lowerIndices = np.flatnonzero(coordinates < upperBound)
            coordinates = upperStructure.getCartesianCoordinates()[:, self._axis]
            lowerBound = coordinates.min() + self._thickness if self._thickness is not None else coordinates.max()
            upperIndices = np.flatnonzero(coordinates > lowerBound)
            indices = np.concatenate((lowerIndices, upperIndices))
            return Interface(lowerStructure, upperStructure, indices, self)

        @staticmethod
        def build(lowerFile, upperFile, pbc, slabThickness, adjust=False, sigma=None, plane=None, rotAxis=None,
                    lowerPlane=None, upperPlane=None, maxMisfitStrain=None, maxEnvironmentArea=None, **kwagrs):
            """
            Builds the environment objects for a given description.
            """
            if lowerFile == upperFile and sigma is not None:
                logger.debug(f'Proceeding with Grain Boundary mode')
                initStructure = EnvironmentUtility.structureRepresentation.readPOSCAR(lowerFile, pbc=(1, 1, 1))
                lowerStructure, upperStructure = constructGrainsSlabs(initStructure, pbc, sigma, plane, rotAxis, slabThickness)
                logger.debug('Grains are successfully created')
            else:
                logger.debug(f'Proceeding with Heterostructure mode')
                initLowerStructure = EnvironmentUtility.structureRepresentation.readPOSCAR(lowerFile, pbc=(1, 1, 1))
                initUpperStructure = EnvironmentUtility.structureRepresentation.readPOSCAR(upperFile, pbc=(1, 1, 1))
                lowerStructure = constructSurfaceSlab(initLowerStructure, pbc, lowerPlane, slabThickness)
                upperStructure = constructSurfaceSlab(initUpperStructure, pbc, upperPlane, slabThickness)
                logger.debug('Surface Slabs are successfully created')
            if adjust:
                logger.debug(f'Auto-adjustment of interfacial slabs was enabled')
                antiPBC = tuple((~np.asarray(pbc, dtype=bool)).tolist())
                nonPBCAxis = np.flatnonzero(antiPBC)[0]
                maxMisfitStrain = maxMisfitStrain if maxMisfitStrain else DEFAULT_MAX_MISFIT_STRAIN
                maxEnvironmentArea = maxEnvironmentArea if maxEnvironmentArea else DEFAULT_MAX_ENVIRONMENT_AREA
                lowerStructure, upperStructure, _ = adjustStructures(lowerStructure, upperStructure, axis=nonPBCAxis,
                                                                        maxMisfitStrain=maxMisfitStrain,
                                                                        maxSubstrateArea=maxEnvironmentArea)
                logger.debug(f'Interfacial slabs are successfully adjusted')
            environment = dict(
                lowerStructure=lowerStructure,
                upperStructure=upperStructure,
            )
            return environment

    processingStyles = {
        'onlyLowerEnvironment': 'getLowerStructure',
        'onlyUpperEnvironment': 'getUpperStructure'
    }

    def __init__(self, lowerStructure, upperStructure, indices, assembler):
        """
        :param structures: atomic structures associated with this environment.
        :param gap: a gap between the film and substrate
        :param offsetVector: vector to be added to molecules centers when assemble whole structure.
            If not provided such vector will be calculated on demand.
        """
        self._structures = (lowerStructure, upperStructure)
        self._indices = indices
        self._assembler = assembler

    @staticmethod
    def fromIndices(structure, lowerSlab, upperSlab, fixed, pbc):
        all = np.asarray(lowerSlab + upperSlab, dtype=int)
        lowerSlab = np.asarray(lowerSlab, dtype=int)
        upperSlab = np.asarray(upperSlab, dtype=int)
        fixed = np.where(np.in1d(all, fixed))[0]
        cell = EnvironmentUtility.cellType(structure.getCell().getCellVectors(), pbc=pbc)
        lowerEnvStructure = EnvironmentUtility.structureType(structure.getAtomTypes()[lowerSlab],
                                                             structure.getCartesianCoordinates()[lowerSlab],
                                                             cell)
        upperEnvStructure = EnvironmentUtility.structureType(structure.getAtomTypes()[upperSlab],
                                                             structure.getCartesianCoordinates()[upperSlab],
                                                             cell)
        return Interface(lowerEnvStructure, upperEnvStructure, fixed, None), all

    def getUpdatedEnvironment(self, envStructure):
        envAtomTypes = envStructure.getAtomTypes()
        envCoordinates = envStructure.getCartesianCoordinates()
        envCell = envStructure.getCell()
        N = len(self._structures[0])
        lowerEnvStructure = EnvironmentUtility.structureType(envAtomTypes[:N], envCoordinates[:N], envCell)
        upperEnvStructure = EnvironmentUtility.structureType(envAtomTypes[N:], envCoordinates[N:], envCell)
        return Interface(lowerEnvStructure, upperEnvStructure, self._indices, self._assembler)

    def getLowerStructure(self):
        """
        Retrieve the atomic structure associated with lower part of the environment.

        :return: atomic structure.
        """
        structure = self._structures[0]
        coordinates = structure.getCartesianCoordinates()
        cell = structure.getRectifiedCell().getEnvelopeCell(coordinates, DEFAULT_VACUUM)
        coordinates = cell.center(coordinates)
        return EnvironmentUtility.structureType(structure.getAtomTypes(), coordinates, cell)

    def getUpperStructure(self):
        """
        Retrieve the atomic structure associated with upper part of the environment.

        :return: atomic structure.
        """
        structure = self._structures[1]
        coordinates = structure.getCartesianCoordinates()
        cell = structure.getRectifiedCell().getEnvelopeCell(coordinates, DEFAULT_VACUUM)
        coordinates = cell.center(coordinates)
        return EnvironmentUtility.structureType(structure.getAtomTypes(), coordinates, cell)   

    def getStructure(self):
        """
        Combines the two parts of interface in a single structure and returns it.

        :return: atomic structure.
        """
        atomTypes = []
        coordinates = []
        for i, structure in enumerate(self._structures):
            atomTypes.extend(structure.getAtomTypes())
            coordinates.extend(structure.getCartesianCoordinates())
        assembledCell = self._structures[0].getCell()
        structure = EnvironmentUtility.structureType(atomTypes, coordinates, assembledCell)
        assembledCell = structure.getRectifiedCell().getEnvelopeCell(coordinates, vacuumSize=DEFAULT_VACUUM)
        coordinates = assembledCell.center(structure.getCartesianCoordinates())
        return EnvironmentUtility.structureType(atomTypes, coordinates, assembledCell)

    def getFixedIndices(self):
        """
        Get indices of atoms in substrate positions of which are fixed.
        """
        return self._indices

    def adjustSystem(self, molecules, cell):
        """
        Adjusts the cells of the environment and the structure to fit each other
        """
        maxMisfitStrain = self._assembler.maxMisfitStrain
        maxEnvironmentArea = self._assembler.maxEnvironmentArea
        lowerEnvStructure, upperEnvStructure = self._structures
        lowerCell = lowerEnvStructure.getCell()
        upperCell = upperEnvStructure.getCell()
        axis = np.flatnonzero(lowerCell.getAntiPBC())[0]

        logger.debug('Starting adjustment of the Interface')

        envCellsAreClose = lowerCell.isClose(upperCell)
        firstStageMaxArea = maxEnvironmentArea if envCellsAreClose else maxEnvironmentArea * 0.6

        # Stage 1. We adjust initial molecules and cell of our system to the lower part of the environment
        newMolecules, newCell, newLowerEnvStructure, supercellMatrices = adjustSystem(molecules, cell,
                                                                                      lowerEnvStructure, axis,
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
                                                                                          upperEnvStructure, axis,
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

        newEnvironment = self._assembler.assemble(molecules, cell,
                                                  lowerEnvStructure=newLowerEnvStructure,
                                                  upperStructure=newUpperEnvStructure)
        return newMolecules, newCell, newEnvironment


class Bulk:

    class Assembler:

        def __init__(self, structure, isFixed: bool = True, **kwargs):
            self._structure = structure
            self.isFixed = isFixed
            if self.isFixed:
                self._indices = np.arange(len(structure))
            else:
                self._indices = np.array([], dtype=int)

        def getCell(self):
            return self._structure.getCell()

        def assemble(self, molecules, cell, **kwargs):
            return Bulk(self._structure, self._indices, self)

        @staticmethod
        def build(file, **kwargs):
            """
            Builds the environment objects for a given description.
            """
            structure = EnvironmentUtility.structureRepresentation.readPOSCAR(file)
            environment = dict(
                structure=structure,
            )
            return environment

    processingStyles = {
        'onlyEnvironment': 'getStructure'
    }

    def __init__(self, structure, indices, assembler):
        self._structure = structure
        self._indices = indices
        self._assembler = assembler

    @staticmethod
    def fromIndices(structure, all, fixed, pbc):
        all = np.asarray(all, dtype=int)
        fixed = np.where(np.in1d(all, fixed))[0]
        envStructure = EnvironmentUtility.structureType(structure.getAtomTypes()[all],
                                          structure.getCartesianCoordinates()[all],
                                          EnvironmentUtility.cellType(structure.getCell().getCellVectors(), pbc=pbc))
        return Bulk(envStructure, fixed, None), all

    def getUpdatedEnvironment(self, envStructure):
        return Bulk(envStructure, self._indices, self)

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

class NanoparticleCore:

    class Assembler:
        def __init__(self, structure):
            self.structure = structure
            self.connectors = {}

        def assemble(self, alpha, **kwargs):
            return NanoparticleCore(self.structure, alpha)

        def getConnectors(self, whichConnectors):
            if whichConnectors in self.connectors:
                return self.connectors[whichConnectors]
            elif hasattr(whichConnectors, 'label') and whichConnectors.label in ('FACE', 'EDGE', 'VERTEX'):
                return self.calc_alphashape_connectors(whichConnectors)


        def calc_alphashape_connectors(self, whichConnectors):
            # determine active centers + normal vectors self.activeCenters = [(xyz, normal), ...],
            if whichConnectors.connectorParam is None:
                alpha = 0.
            else:
                alpha = whichConnectors.connectorParam
            alpha_shape = alphashape.alphashape(self.structure.getCartesianCoordinates(), alpha)
            self.connectors[type(whichConnectors)("FACE", alpha)] =\
                [{'mount_point': m, 'orientation': v}
                 for m, v in zip(alpha_shape.triangles_center, alpha_shape.face_normals)]
            self.connectors[type(whichConnectors)("VERTEX", alpha)] =\
                [{'mount_point': m, 'orientation': v}
                 for m, v in zip(alpha_shape.vertices, alpha_shape.vertex_normals)]
            edge_normals = []
            for adj_e, adj_f in zip(alpha_shape.face_adjacency_edges, alpha_shape.face_adjacency):
                origin = 0.5 * (alpha_shape.vertices[adj_e[0]] + alpha_shape.vertices[adj_e[1]])
                normal = alpha_shape.face_normals[adj_f[0]] + alpha_shape.face_normals[adj_f[1]]
                normal /= np.linalg.norm(normal)
                edge_normals.append({'mount_point': origin, 'orientation': normal})
            self.connectors[type(whichConnectors)("EDGE", alpha)] = edge_normals
            return self.connectors[whichConnectors]

        @staticmethod
        def build(filename, **kwargs):
            structure = EnvironmentUtility.structureRepresentation.readXYZ(filename)
            environment = dict(
                structure=structure,
            )
            return environment



    def __init__(self, structure, links):
        self.structure = structure







    pass


class EnvironmentUtility:
    """
    Class representing utility which generates possible environments for calculation.
    """

    structureRepresentation = None
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None
    supportedEnvironments = {
        'interface': Interface,
        'substrate': Substrate,
        'bulk': Bulk,
        'nanoparticle_core': NanoparticleCore
    }
   

    @classmethod
    def setRepresentation(cls, representation):
        cls.structureRepresentation = representation
    
    @classmethod
    def registerTypes(cls, structureType, atomType, cellType, atomicDisassemblerType):
        """
        Register types used by this utility.

        :param structureType: type representing atomic structure.
        :param atomType: type representing chemical element.
        :param cellType: type representing unit cell.
        :param atomicDisassemblerType: type representing utility used for disassembling structure into molecules.
        """
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType

    @classmethod
    def build(cls, type, **description):
        return EnvironmentUtility.supportedEnvironments.get(type).Assembler.build(**description)

    def __init__(self, environments: list = None):
        """

        """
        self.assemblers = [self.supportedEnvironments.get(environment['type']).Assembler(**environment)
                           for environment in (environments if environments is not None else [])]


# TODO create a unittest for all environment types
def adjustSystem(molecules, cell, envStructure, axis, maxSubstrateArea, maxMisfitStrain, returnSupercellMatrices=False):
    """
    Adjusts the cells of the environment and the structure to make them fit each other
    """
    logger.debug(f'Got the system with {len(molecules)} atoms')
    logger.debug(f'Got the envStructure with {len(envStructure)} atoms')
    structure, disassembler = EnvironmentUtility.structureType.assemble(molecules, cell)
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
    newStructure = EnvironmentUtility.structureType.initFromFractionalCoordinates(structure.getAtomTypes(), 
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

def convertFromPymatgen(pmgStructure, pbc):
    """
    Converts the pymatgen Structure object to the USPEX EnvironmentUtility.structureType object
    """
    cellVectors = pmgStructure.lattice.matrix[np.flatnonzero(pbc)]
    cell = EnvironmentUtility.cellType.initFromCellVectors(pbc, cellVectors)
    species = [EnvironmentUtility.atomType(specie.name) for specie in pmgStructure.species]
    coordinates = pmgStructure.cart_coords
    structure = EnvironmentUtility.structureType(species, coordinates, cell)
    return structure

def convertToPymatgen(structure):
    """
    Converts the USPEX EnvironmentUtility.structureType object to the pymatgen Structure object 
    """
    cellVectors = structure.getRectifiedCell().getCellVectors()
    coordinates = structure.getCartesianCoordinates()
    species = [el.short_name for el in structure.getAtomTypes()]
    pmgStructure = Structure(cellVectors, species, coordinates, coords_are_cartesian=True)
    return pmgStructure

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
