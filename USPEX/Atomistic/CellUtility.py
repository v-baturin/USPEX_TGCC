"""
USPEX.Atomistic.CellUtility
===========================
"""

import logging
import numpy as np
from copy import copy

from ..Expressions.Functions.CellFunctions import CellFunctions
from .Primitives.Cell import Cell

logger = logging.getLogger(__name__)
_DEFAULT_SYMMETRY_TOLERANCE = 0.05


class CellUtility:
    """
    Utility for working with unit cells of atomic structures.
    """

    propertyExtension = CellFunctions

    def __init__(self, dim=None, pbc=None, cellVectors = None, cellParameters = None, cellVolume = None, axis=None,
                 thickness=None, supercellDegree = None, symTolerance=None, debug = False):
        """

        :param dim: dimensionality, i.e. number of periodic directions.
        :param pbc: periodic boundary conditions in each direction.
        :param cellVectors: for fixed cell calculation 3*3 array of cell vectors.
        :param cellParameters: for fixed cell calculation dictionary
            {'a': <float>, 'b': <float>, 'c': <float>, 'alpha': <float>, 'beta': <float>, 'gamma': <float>}
            with cell parameters.
        :param cellVolume: for fixed volume calculation cell volume.
        :param axis: for 1D periodic calculations vector along periodic axis,
            for 2D periodic calculations vector orthogonal to two periodic axes.
        :param thickness: for 2D, 1D and 0D structures constraint on size of containment space.
        :param supercellDegree: int or list of int with allowed supercell sizes.
        :param symTolerance: allowed imperfection of atomic positions when determining symmetry of structure.
        :param debug: switch between two levels of logging. True for debug level, false for INFO level.

        """

        assert not (pbc is not None and dim is not None)
        if pbc is not None:
            self._pbc = pbc
        elif dim is None or dim == 3:
            self._pbc = (1, 1, 1)
        elif dim == 2:
            self._pbc = (1, 1, 0)
        elif dim == 1:
            self._pbc = (1, 0, 0)
        elif dim == 0:
            self._pbc = (0, 0, 0)
        else:
            raise ValueError(f"Incorrect dimensionality {dim}.")

        self._dim = sum(self._pbc)

        if self._dim == 3:
            assert thickness is None and axis is None
            if cellVectors is not None:
                assert cellParameters is None and cellVolume is None
            elif cellParameters is not None:
                assert cellVolume is None
        elif self._dim == 2:
            assert thickness is not None
            if cellVectors is not None:
                assert cellParameters is None and axis is None and cellVolume is None
            elif cellParameters is not None:
                assert axis is not None and cellVolume is None
        elif self._dim == 1:
            assert thickness is not None
            if cellVectors is not None:
                assert cellParameters is None and axis is None and cellVolume is None
            elif cellParameters is not None:
                assert axis is not None and cellVolume is None
        elif self._dim == 0:
            assert cellVectors is None and cellParameters is None and cellVolume is None and axis is None
        else:
            raise RuntimeError(f"Wrong pbc {pbc}.")

        self._axis = np.asarray(axis, dtype=float) if axis is not None else None

        if cellVectors is not None:
            self._cell = Cell.initFromCellVectors(self._pbc, cellVectors)
            if self._dim == 1:
                self._axis = self._cell.getCellVectorsPBC()[0]
                self._axis /= np.linalg.norm(self._axis)
            elif self._dim == 2:
                self._axis = self._cell.getCellVectorsAntiPBC()[0]
                self._axis /= np.linalg.norm(self._axis)
        elif cellParameters is not None:
            self._cell = Cell.initFromCellParameters(self._pbc, **cellParameters, axis = self._axis)
        else:
            self._cell = None

        self._thickness = thickness
        if self._thickness is not None and self._cell is not None:
            self._cell = self._cell.getEnvelopeCell(vacuumSize=self._thickness)

        assert cellVolume is None or self._cell is None
        if cellVolume is not None:
            self._volume = cellVolume
        elif self._cell is not None and supercellDegree is None:
            if self._dim == 3:
                self._volume = self._cell.getVolume()
            elif self._dim == 2:
                self._volume = self._cell.getArea() * self._thickness
            elif self._dim == 1:
                self._volume = self._cell.getLength() * self._thickness ** 2
            else:
                raise ValueError(f"Incorrect dimensionality {dim}.")
        else:
            self._volume = None

        self._supercellDegree = (supercellDegree, supercellDegree + 1) if isinstance(supercellDegree, int) else supercellDegree

        if symTolerance is not None:
            if isinstance(symTolerance, str):
                if 'high' in symTolerance:
                    self.symTolerance = 0.05
                elif 'medium' in symTolerance:
                    self.symTolerance = 0.1
                elif 'low' in symTolerance:
                    self.symTolerance = 0.2
                else:
                    self.symTolerance = _DEFAULT_SYMMETRY_TOLERANCE
            elif isinstance(symTolerance, (float, int)):
                self.symTolerance = float(symTolerance)
            else:
                self.symTolerance = _DEFAULT_SYMMETRY_TOLERANCE
        else:
            self.symTolerance = _DEFAULT_SYMMETRY_TOLERANCE

        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def getDim(self):
        """
        :return: dimensionality set for the **CellUtility** instance.
        """
        return self._dim

    def getPBC(self):
        """
        :return: periodic boundary conditions set for the **CellUtility** instance.
        """
        return self._pbc

    def getCell(self):
        """
        :return: for fixed cell calculations copy of the unit cell object otherwise None.
        """
        return None if self._cell is None else copy(self._cell)

    def getAxis(self):
        """
        :return: axis set for the **CellUtility** instance.
        In 1D it is the periodic axis, in 2D it is normal to the periodic plane.
        """
        return np.copy(self._axis) if self._axis is not None else None

    def getThickness(self):
        """
        :return: Thickness of the system.
        """
        return self._thickness

    def getCellVolume(self):
        """
        :return: volume of unit cell if it is set or the cell is fixed, otherwise *None*.
        """
        return self._volume

    def adjustCell(self, cellVectors, estimatedVolume, numAtoms, baseCell=None):
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

        if self._cell is not None:
            baseCell = self._cell
        elif baseCell is not None:
            baseCell = baseCell.getEnvelopeCell(vacuumSize=self._thickness)
        if self._dim == 1 or self._dim == 2:
            if self._axis is None:
                if self._dim == 1:
                    axis = baseCell.getCellVectorsPBC()[0]
                    axis /= np.linalg.norm(axis)
                else:
                    axis = baseCell.getCellVectorsAntiPBC()[0]
                    axis /= np.linalg.norm(axis)
            else:
                axis = self._axis
            cellVectors = Cell(cellVectors, self._pbc).getAlignedCell(axis).getCellVectors()
        if baseCell is None:
            cell = Cell(cellVectors, self._pbc)
            d = np.power(estimatedVolume / numAtoms, 1.0 / 3.0)
            if self._dim == 3:
                factorMin = factorMax = np.power(estimatedVolume / cell.getVolume(), 1.0 / 3.0)
            elif self._dim == 2:
                estimatedAreaMin = estimatedVolume / self._thickness
                estimatedAreaMax = estimatedVolume / d
                area = cell.getArea()
                factorMin = np.sqrt(estimatedAreaMin / area)
                factorMax = np.sqrt(estimatedAreaMax / area)
            elif self._dim == 1:
                estimatedLengthMin = estimatedVolume / (self._thickness) ** 2
                estimatedLengthMax = estimatedVolume / d ** 2
                length = cell.getLength()
                factorMin = estimatedLengthMin / length
                factorMax = estimatedLengthMax / length
            elif self._dim == 0:
                factorMin = factorMax = 1
            else:
                raise RuntimeError(f"Wrong dim {cell.dim}.")
            if factorMin > 1 or factorMax < 1:
                factor = (factorMax - factorMin)*np.random.random() + factorMin
            else:
                factor = 1
            cellVectors[np.nonzero(self._pbc)] *= factor
            cell = Cell(cellVectors, self._pbc)
        elif self._supercellDegree is not None:
            cell = Cell(cellVectors, self._pbc)
            if self._dim == 3:
                factor = cell.getVolume() / baseCell.getVolume()
            elif self._dim == 2:
                factor = cell.getArea() / baseCell.getArea()
            elif self._dim == 1:
                factor = cell.getLength() / baseCell.getLength()
            else:
                raise RuntimeError(f"Wrong dim {self._dim}.")
            reconstruction = self.getRandomSupercell(int(np.round(factor)))
            cell = Cell(reconstruction.dot(baseCell.getCellVectors()), self._pbc)
        else:
            cell = copy(baseCell)
        return cell

    def getRandomCell(self, estimatedVolume, numAtoms, baseCell=None):
        """
        For given volume and number of atoms creates random unit cell with appropriate size and periodic boundary conditions.

        :param estimatedVolume: estimated volume of bulk unit cell.
        :param numAtoms: number of atoms in structure.

        :return: **Cell** object with appropriate parameters.
        """
        if self._cell is not None:
            baseCell = self._cell
        elif baseCell is not None:
            baseCell = baseCell.getEnvelopeCell(vacuumSize=self._thickness)
        if baseCell is None:
            r2d = 180 / np.pi
            if self._dim == 3:
                a, b, c = np.random.random(3) + 0.5
                def _randomAngles():
                    while True:
                        alpha, beta, gamma = (np.random.random(3) * 4 + 1) * np.pi / 6
                        if 1. - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 2.*np.cos(alpha)*np.cos(beta)*np.cos(gamma) >= 0.3:
                            return r2d * alpha, r2d * beta, r2d * gamma
                cell = Cell.initFromCellParameters(self._pbc, a, b, c, *_randomAngles())
            elif self._dim == 2:
                a, b = np.random.random(2) + 0.5
                alpha = (np.random.random() * 4 + 1) * 30
                cell = Cell.initFromCellParameters(self._pbc, a, b, alpha, self._axis)
            elif self._dim == 1:
                a = np.random.random() + 0.5
                cell = Cell.initFromCellParameters(self._pbc, a, self._axis)
            elif self._dim == 0:
                cell = Cell.initFromCellParameters(self._pbc)
            else:
                raise RuntimeError(f"Wrong pbc {self._pbc}.")
            cell = self.adjustCell(cell.getCellVectors(), estimatedVolume, numAtoms, baseCell)
            if self._thickness is not None:
                cell = cell.getEnvelopeCell(vacuumSize=self._thickness)
        elif self._supercellDegree is not None:
            reconstruction = self.getRandomSupercell()
            cell = Cell(reconstruction.dot(baseCell.getCellVectors()), self._pbc)
        else:
            cell = copy(baseCell)
        return cell

    def getHybridCell(self, cell1, cell2, fraction):
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
        assert 0 <= fraction <= 1
        if self._cell is None:
            cellParameters = fraction * np.asarray(cell1.getCellParameters()) + \
                             (1 - fraction) * np.asarray(cell2.getCellParameters())
            if self._dim == 3:
                cell = Cell.initFromCellParameters(self._pbc, *cellParameters, axis=self._axis)
                factor = np.power((fraction * cell1.getVolume() + (1 - fraction) * cell2.getVolume()) / cell.getVolume(), 1. / 3.)
                thickness = 0
                cellParameters[0:3] *= factor
            elif self._dim == 2:
                a, b, alpha, axis = cellParameters
                cell = Cell.initFromCellParameters(self._pbc, a, b, alpha, self._axis)
                factor = np.sqrt((fraction * cell1.getArea() + (1 - fraction) * cell2.getArea()) / cell.getArea())
                thickness = fraction * cell1.getLength() + (1 - fraction) * cell2.getLength()
                cellParameters = (a * factor, b * factor, alpha, self._axis)
            elif self._dim == 1:
                a, axis = cellParameters
                cell = Cell.initFromCellParameters(self._pbc, a, self._axis)
                factor = (fraction * cell1.getLength() + (1 - fraction) * cell2.getLength()) / cell.getLength()
                thickness = np.sqrt(fraction * cell1.getArea() + (1 - fraction) * cell2.getArea())
                cellParameters = (a * factor, self._axis)
            elif self._dim == 0:
                thickness = np.power(fraction * cell1.getVolume() + (1 - fraction) * cell2.getVolume(), 1.0 / 3.0)
                cellParameters = ()
            else:
                raise RuntimeError(f"Wrong dim {self._dim}.")
            if self._thickness is not None and thickness > self._thickness:
                thickness = self._thickness
            cell = Cell.initFromCellParameters(self._pbc, *cellParameters).getEnvelopeCell(vacuumSize=thickness)
        elif self._supercellDegree is not None:
            if self._dim == 3:
                factor1 = cell1.getVolume() / self._cell.getVolume()
                factor2 = cell2.getVolume() / self._cell.getVolume()
            elif self._dim == 2:
                factor1 = cell1.getArea() / self._cell.getArea()
                factor2 = cell2.getArea() / self._cell.getArea()
            elif self._dim == 1:
                factor1 = cell1.getLength() / self._cell.getLength()
                factor2 = cell2.getLength() / self._cell.getLength()
            else:
                raise RuntimeError(f"Wrong dim {self._dim}.")
            factor = (fraction * factor1 + (1 - fraction) * factor2)
            reconstruction = self.getRandomSupercell(int(np.round(factor)))
            cell = Cell(reconstruction.dot(self._cell.getCellVectors()), self._pbc)
        else:
            cell = self.getCell()
        return cell

    def getRandomSupercell(self, factor: int = None):
        """
        Picks on of possible supercells consisting of specified number of cells.
        :param factor: number of cells which requested supercell should contain.
        :return: supercell matrix (m, n, l).
        """
        matrix = np.eye(3)
        if self._supercellDegree is None:
            return matrix
        minDegree = self._supercellDegree[0]
        maxDegree = self._supercellDegree[1]
        if factor is None:
            factor = np.random.randint(minDegree, maxDegree)
        assert minDegree <= factor < maxDegree, f'No supercells with factor {factor} are allowed.'

        while True:
            if self._dim == 3:
                matrix = np.random.randint(-factor, factor + 1, size=(3, 3))
            elif self._dim == 2:
                inds = np.flatnonzero(self._pbc)
                matrix[tuple(np.meshgrid(inds, inds))] = np.random.randint(-factor, factor + 1, size=(2, 2))
            elif self._dim == 1:
                ind = np.flatnonzero(self._pbc)[0]
                matrix[ind, ind] = factor
            else:
                raise RuntimeError(f"Wrong dim {self._dim}.")
            if np.round(np.linalg.det(matrix)) == factor:
                return matrix

    def isGoodCell(self, cell):
        """
        Checks if provided cell satisfies constraints.
        :param cell: cell to be checked.
        :return:
        """
        isGood = True
        if self._cell is not None and self._supercellDegree is None:
            isGood  = isGood and (cell == self._cell)
        if self._dim == 2:
            isGood = isGood and (cell.getLength() <= self._thickness * 1.0001)
        elif self._dim == 1:
            isGood = isGood and (cell.getRadius() <= self._thickness * 0.7072)
        elif self._thickness is not None:
            isGood = isGood and (cell.getRadius() <= self._thickness * 0.8661)
        return isGood

