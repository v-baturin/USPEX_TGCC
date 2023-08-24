import numpy as np
import logging


logger = logging.getLogger(__name__)


class Bulk:
    Atomistic = None

    @classmethod
    def registerTypes(cls, Atomistic):
        """
        Register types used by this utility.

        :param structureType: type representing atomic structure.
        :param atomType: type representing chemical element.
        :param cellType: type representing unit cell.
        :param atomicDisassemblerType: type representing utility used for disassembling structure into molecules.
        """
        cls.Atomistic = Atomistic

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
        structure = Bulk.Atomistic.AtomicStructureRepresentation.readPOSCAR(file)
        environment = dict(
            structure=structure,
        )
        return environment
