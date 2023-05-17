import numpy as np
import logging


logger = logging.getLogger(__name__)


class Bulk:
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
            structure = Bulk.structureRepresentation.readPOSCAR(file)
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
        envStructure = Bulk.structureType(structure.getAtomTypes()[all],
                                          structure.getCartesianCoordinates()[all],
                                          Bulk.cellType(structure.getCell().getCellVectors(), pbc=pbc))
        return Bulk(envStructure, fixed, None), all

    def getUpdatedEnvironment(self, envStructure):
        return Bulk(envStructure, self._indices, self._assembler)

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
