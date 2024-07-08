import numpy as np
import logging


from ...Semantics.Atomistic.Environment import Environment as EnvironmentSemantics
from .slabFunctions import constructSurfaceSlab  # adjustSystem


logger = logging.getLogger(__name__)


DEFAULT_SUBSTRATE_GAP = 2.0
DEFAULT_MAX_MISFIT_STRAIN = 5E-3
DEFAULT_MAX_ENVIRONMENT_AREA = 1000


class Substrate(EnvironmentSemantics):
    """
    Class representing part of structure which is not being altered via variation operators.
    I.e. it acts as environment for individual.
    """
    Atomistic = None

    @classmethod
    def registerTypes(cls, Atomistic):
        """
        Register types used by this utility.
        """
        cls.Atomistic = Atomistic

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
        sysStructure, disassembler = Substrate.Atomistic.atomicDisassemblerType.assemble({'atomistic.molecules': molecules,
                                                                                'atomistic.cell': cell})
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

        finalStructure = Substrate.Atomistic.structureType(intermediateStructure.getAtomTypes(),
                                                          envCoordinates - offset, envCell)
        coordinates = finalStructure.getCartesianCoordinates()[:, self._axis]
        upperBound = coordinates.max() - self._thickness if self._thickness is not None else coordinates.min()
        indices = np.flatnonzero(coordinates < upperBound)
        return [(finalStructure, indices)]

    @staticmethod
    def build(file, pbc, build=False, plane=None, slabThickness=None, **kwargs):
        """
        Builds the environment objects for a given description.
        """
        structure = Substrate.Atomistic.AtomicStructureRepresentation.readPOSCAR(file, pbc)
        if build:
            structure = constructSurfaceSlab(structure, pbc, plane, slabThickness)
        environment = dict(
            structure=structure,
        )
        return environment

# def adjustSystem(self, molecules, cell):
#     """
#     Adjusts the cells of the environment and the structure to fit each other
#     """
#     antiPBC = self._structure.getCell().getAntiPBC()
#     assert sum(antiPBC) == 1
#     axis = np.flatnonzero(antiPBC)[0]
#     maxEnvironmentArea = self._assembler.maxEnvironmentArea
#     maxMisfitStrain = self._assembler.maxMisfitStrain
#     newMolecules, newCell, newEnvStructure = adjustSystem(molecules, cell, self._structure, axis,
#                                                           maxEnvironmentArea, maxMisfitStrain)
#     newEnvironment = self._assembler.assemble(molecules, cell, structure=newEnvStructure)
#     return newMolecules, newCell, newEnvironment
