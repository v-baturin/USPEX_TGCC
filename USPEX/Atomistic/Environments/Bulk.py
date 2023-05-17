import numpy as np
import logging


logger = logging.getLogger(__name__)


class Bulk:
    structureRepresentation = None
    structureType = None
    atomType = None
    cellType = None
    atomicDisassemblerType = None

    processingStyles = {
        'onlyEnvironment': 'getStructure'
    }

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
        return [(self._structure, self._indices)]

    @staticmethod
    def build(file, **kwargs):
        """
        Builds the environment objects for a given description.
        """
        structure = Bulk.structureRepresentation.readPOSCAR(file)
        environment = dict(
            structure=structure,
        )
        return environment

def fromIndices(structure, all, fixed, pbc):
    all = np.asarray(all, dtype=int)
    fixed = np.where(np.in1d(all, fixed))[0]
    envStructure = Bulk.structureType(structure.getAtomTypes()[all],
                                      structure.getCartesianCoordinates()[all],
                                      Bulk.cellType(structure.getCell().getCellVectors(), pbc=pbc))
    return envStructure, fixed,  all

