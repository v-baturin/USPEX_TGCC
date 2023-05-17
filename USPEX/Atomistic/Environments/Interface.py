import numpy as np
import logging


from .slabFunctions import adjustSystem, adjustStructures, constructSurfaceSlab, constructGrainsSlabs, alignStructure


logger = logging.getLogger(__name__)


DEFAULT_SUBSTRATE_GAP = 2.0
DEFAULT_MAX_MISFIT_STRAIN = 5E-3
DEFAULT_MAX_ENVIRONMENT_AREA = 1000
DEFAULT_VACUUM = 1e-3


class Interface:
    """
    Class representing part of structure which is not being altered via variation operators.
    I.e. it acts as environment for individual.
    """
    structureRepresentation = None
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    @classmethod
    def registerTypes(cls,representationType, structureType, atomType, cellType, atomicDisassemblerType):
        """
        Register types used by this utility.

        :param structureType: type representing atomic structure.
        :param atomType: type representing chemical element.
        :param cellType: type representing unit cell.
        :param atomicDisassemblerType: type representing utility used for disassembling structure into molecules.
        """
        cls.structureRepresentation = representationType
        cls.structureType = structureType
        cls.atomType = atomType
        cls.cellType = cellType
        cls.atomicDisassemblerType = atomicDisassemblerType

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
            structure, disassembler = Interface.atomicDisassemblerType.assemble(molecules, cell)
            lowerOffset = self._calculateLowerOffset(structure.getFractionalCoordinates(), cell.getPBC())
            upperOffset = self._calculateUpperOffset(structure.getCartesianCoordinates(), cell.getPBC()) - lowerOffset
            lowerStructure = Interface.structureType(lowerStructure.getAtomTypes(),
                                                              lowerStructure.getCartesianCoordinates() - lowerOffset,
                                                              lowerStructure.getCell())
            upperStructure = Interface.structureType(upperStructure.getAtomTypes(),
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
                initStructure = Interface.structureRepresentation.readPOSCAR(lowerFile, pbc=(1, 1, 1))
                lowerStructure, upperStructure = constructGrainsSlabs(initStructure, pbc, sigma, plane, rotAxis, slabThickness)
                logger.debug('Grains are successfully created')
            else:
                logger.debug(f'Proceeding with Heterostructure mode')
                initLowerStructure = Interface.structureRepresentation.readPOSCAR(lowerFile, pbc=(1, 1, 1))
                initUpperStructure = Interface.structureRepresentation.readPOSCAR(upperFile, pbc=(1, 1, 1))
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
        cell = Interface.cellType(structure.getCell().getCellVectors(), pbc=pbc)
        lowerEnvStructure = Interface.structureType(structure.getAtomTypes()[lowerSlab],
                                                             structure.getCartesianCoordinates()[lowerSlab],
                                                             cell)
        upperEnvStructure = Interface.structureType(structure.getAtomTypes()[upperSlab],
                                                             structure.getCartesianCoordinates()[upperSlab],
                                                             cell)
        return Interface(lowerEnvStructure, upperEnvStructure, fixed, None), all

    def getUpdatedEnvironment(self, envStructure):
        envAtomTypes = envStructure.getAtomTypes()
        envCoordinates = envStructure.getCartesianCoordinates()
        envCell = envStructure.getCell()
        N = len(self._structures[0])
        lowerEnvStructure = Interface.structureType(envAtomTypes[:N], envCoordinates[:N], envCell)
        upperEnvStructure = Interface.structureType(envAtomTypes[N:], envCoordinates[N:], envCell)
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
        return Interface.structureType(structure.getAtomTypes(), coordinates, cell)

    def getUpperStructure(self):
        """
        Retrieve the atomic structure associated with upper part of the environment.

        :return: atomic structure.
        """
        structure = self._structures[1]
        coordinates = structure.getCartesianCoordinates()
        cell = structure.getRectifiedCell().getEnvelopeCell(coordinates, DEFAULT_VACUUM)
        coordinates = cell.center(coordinates)
        return Interface.structureType(structure.getAtomTypes(), coordinates, cell)

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
        structure = Interface.structureType(atomTypes, coordinates, assembledCell)
        assembledCell = structure.getRectifiedCell().getEnvelopeCell(coordinates, vacuumSize=DEFAULT_VACUUM)
        coordinates = assembledCell.center(structure.getCartesianCoordinates())
        return Interface.structureType(atomTypes, coordinates, assembledCell)

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
