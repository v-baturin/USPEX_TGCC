from typing import Union
from .Primitives.Generics import Tiling3, Vector3
from .Primitives.Cell import Cell


class CellUtility:
    """
    Utility for working with unit cells of atomic structures.
    """

    def getDim(self) -> int:
        """
        :return: dimensionality set for the **CellUtility** instance.
        """
        pass

    def getPBC(self) -> tuple[bool, bool, bool]:
        """
        :return: periodic boundary conditions set for the **CellUtility** instance.
        """
        pass

    def getCell(self) -> Union[Cell, None]:
        """
        :return: for fixed cell calculations copy of the unit cell object otherwise None.
        """
        pass

    def getAxis(self) -> Vector3:
        """
        :return: axis set for the **CellUtility** instance.
        In 1D it is the periodic axis, in 2D it is normal to the periodic plane.
        """
        pass

    def getThickness(self) -> float:
        """
        :return: Thickness of the system.
        """
        pass

    def getCellVolume(self) -> float:
        """
        :return: volume of unit cell if it is set or the cell is fixed, otherwise *None*.
        """
        pass

    def propertyExtension(self):
        return CellFunctions(self)

    def adjustCell(self, cellVectors, estimatedVolume, numAtoms, baseCell=None) -> Cell:
        """
        Adjust given unit cell according calculation parameters, provided volume and number of atoms.
        If cell in calculation is fixed returns the fixed cell.
        Otherwise, scale input unit cell to have specific volume (3D), area (2D) or length (1D).
        TODO here we should describe how we get this parameters.

        :param cellVectors: input unit cell vectors to be adjusted.
        :param estimatedVolume: estimated volume of bulk unit cell.
        :param numAtoms: number of atoms in structure.

        :return: **Cell** object with adjusted parameters.
        """
        pass

    def getRandomCell(self, estimatedVolume, numAtoms, baseCell=None) -> Cell:
        """
        For given volume and number of atoms creates random unit cell with appropriate size and periodic boundary conditions.

        :param estimatedVolume: estimated volume of bulk unit cell.
        :param numAtoms: number of atoms in structure.

        :return: **Cell** object with appropriate parameters.
        """
        pass

    def getHybridCell(self, cell1, cell2, fraction) -> Cell:
        """
        Creates hybrid of two unit cells.
        If cell in calculation is fixed returns the fixed cell.
        Otherwise, takes average cell parameters from two input cells with ratio fraction/(1-fraction).
        Then scale its volume (3D), area (2D) or length (1D) to average one with same ratio.
        And finally takes average of non-periodic direction(s) of two cells with same ratio.

        :param cell1: first input cell.
        :param cell2: second input cell.
        :param fraction: fraction of first cell in resulting cell.

        :return: **Cell** object with appropriate parameters.
        """
        pass

    def getRandomSupercell(self, factor: int = None) -> Tiling3:
        """
        Picks on of possible supercells consisting of specified number of cells.
        :param factor: number of cells which requested supercell should contain.
        :return: supercell matrix (m, n, l).
        """
        pass

    def isGoodCell(self, cell) -> bool:
        """
        Checks if provided cell satisfies constraints.
        :param cell: cell to be checked.
        :return:
        """
        pass