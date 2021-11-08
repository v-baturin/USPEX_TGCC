"""
USPEX.Atomistic.CellUtility
===========================
"""

import logging
logger = logging.getLogger(__name__)


import numpy as np
import spglib
from typing import Union, List, Tuple
from copy import copy
from scipy.spatial.transform import Rotation

from .Transformation import Transformation
from .VolumeEstimator import VolumeEstimator

_DEFAULT_SYMMETRY_TOLERANCE = 0.05


class CellUtility:
    """
    Utility for working with unit cells of atomic structures.
    """

    def __init__(self, dim=None, pbc=None, cellVectors = None, cellParameters = None, cellVolume = None,
                 reconstructionDegree = None, symTolerance=None, axis=None, radius=None, thickness=None, volumeType=0,
                 debug = False):
        """

        :param dim: dimensionality, i.e. number of periodic directions.
        :param pbc: periodic boundary conditions in each direction.
        :param cellVectors: for fixed cell calculation 3*3 array of cell vectors.
        :param cellParameters: for fixed cell calculation dictionary
            {'a': <float>, 'b': <float>, 'c': <float>, 'alpha': <float>, 'beta': <float>, 'gamma': <float>}
            with cell parameters.
        :param cellVolume: for fixed volume calculation cell volume.
        :param reconstructionDegree: int or list of int with allowed supercell sizes.
        :param symTolerance: allowed misplacements of atoms when determining symmetry of structure.
        :param axis: for 1D periodic calculations vector along periodic axis,
            for 2D periodic calculations vector orthogonal to two periodic axes.
        :param radius: for 1D and 0D structures constraint on size of containment space.
        :param thickness: for 2D structures constraint on size of containment space.
        :param volumeType:
            range 0 to 1, 0 corresponds to pure atomic environment for
            volume estimation, 1 to pure molecular one.
            Molecular environment is less dense.
            Intermediate value is a coefficient for molecular environment
            in linear combination of the two.
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

        self.dim = sum(self._pbc)

        if self.dim == 3:
            assert radius is None and thickness is None and axis is None
            if cellVectors is not None:
                assert cellParameters is None and cellVolume is None
            elif cellParameters is not None:
                assert cellVolume is None
        elif self.dim == 2:
            assert radius is None and thickness is not None
            if cellVectors is not None:
                assert cellParameters is None and axis is None and cellVolume is None
            elif cellParameters is not None:
                assert cellVolume is None
            else:
                assert axis is not None
        elif self.dim == 1:
            assert thickness is None and radius is not None
            if cellVectors is not None:
                assert cellParameters is None and axis is None and cellVolume is None
            elif cellParameters is not None:
                assert cellVolume is None
            else:
                assert axis is not None
        elif self.dim == 0:
            assert cellVectors is None and cellParameters is None and cellVolume is None and\
                   thickness is None and axis is None
        else:
            raise RuntimeError(f"Wrong pbc {pbc}.")

        self._axis = np.asarray(axis, dtype=float) if axis is not None else None
        self._thickness = thickness
        self._radius = radius

        if cellVectors is not None:
            self._cell = Cell(cellVectors, self._pbc)
        elif cellParameters is not None:
            self._cell = Cell.initFromCellParameters(self._pbc, **cellParameters, axis = self._axis)
        else:
            self._cell = None

        assert cellVolume is None or self._cell is None
        if cellVolume is not None:
            self._volume = cellVolume
        elif self._cell is not None:
            self._volume = self._cell.getVolume() if self.dim == 3 else None
        else:
            self._volume = None

        if reconstructionDegree is not None:
            self._listOfReconstructions = self._getListOfReconstructions(reconstructionDegree)
        else:
            self._listOfReconstructions = []

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

        self.volumeEstimator = VolumeEstimator(volumeType)

        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    def getCell(self):
        """
        :return: for fixed cell calculations copy of the unit cell object otherwise None.
        """
        return None if self._cell is None else copy(self._cell)

    def getRandomCell(self, composition, pressure):
        """
        For given composition and conditions creates random unit cell with appropriate size and periodic boundary conditions.

        :param composition: dictionary like object defining composition.
        :param pressure: external pressure.

        :return: **Cell** object with appropriate parameters.
        """
        if self._cell is None:
            r2d = 180 / np.pi
            if self.dim == 3:
                a, b, c = np.random.random(3) + 0.5
                def _randomAngles():
                    while True:
                        alpha, beta, gamma = (np.random.random(3) * 4 + 1) * np.pi / 6
                        if 1. - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 2.*np.cos(alpha)*np.cos(beta)*np.cos(gamma) >= 0.3:
                            return r2d * alpha, r2d * beta, r2d * gamma
                cell = Cell.initFromCellParameters(self._pbc, a, b, c, *_randomAngles())
            elif self.dim == 2:
                a, b = np.random.random(2) + 0.5
                alpha = (np.random.random() * 4 + 1) * 30
                cell = Cell.initFromCellParameters(self._pbc, a, b, alpha=alpha, axis=self._axis).getEnvelopeCell(vacuumSize=self._thickness)
            elif self.dim == 1:
                a = np.random.random() + 0.5
                cell = Cell.initFromCellParameters(self._pbc, a, axis=self._axis).getEnvelopeCell(vacuumSize=self._radius)
            elif self.dim == 0:
                cell = Cell([], self._pbc)
            else:
                raise RuntimeError(f"Wrong pbc {self._pbc}.")
            cell = self.adjustCell(cell.getCellVectors(), composition, pressure)
        else:
            cell = self.getCell()

        return cell


    def adjustCell(self, cellVectors, composition, pressure):
        """
        Adjust given unit cell according calculation parameters, provided composition and pressure.
        If cell in calculation is fixed returns the fixed cell. Otherwise scale input unit cell to have specific volume.
        If calculation is done with fixed volume then unit cell is scaled to it.
        Otherwise to estimated volume calculated using provided conditions utility.
        Periodic boundary conditions of output unit cell set up same as in utility.

        :param cellVectors: input unit cell vectors to be adjusted.
        :param composition: dictionary like object defining composition.
        :param conditions: external pressure.

        :return: **Cell** object with adjusted parameters.
        """

        cell = self.getCell()
        if cell is None:
            cell = Cell(cellVectors, self._pbc)
            estimatedVolume = self.getCellVolume(composition, pressure)
            d = np.power(estimatedVolume / sum(composition.values()), 1.0 / 3.0)
            if cell.dim == 3:
                factorMin = factorMax = np.power(estimatedVolume / cell.getVolume(), 1.0 / 3.0)
            elif cell.dim == 2:
                estimatedAreaMin = estimatedVolume / self._thickness
                estimatedAreaMax = estimatedVolume / d - self._thickness * d
                area = cell.getArea()
                factorMin = np.sqrt(estimatedAreaMin / area)
                factorMax = np.sqrt(estimatedAreaMax / area)
            elif cell.dim == 1:
                estimatedLengthMin = estimatedVolume / (2 * self._radius) ** 2
                estimatedLengthMax = estimatedVolume / d ** 2
                length = cell.getLength()
                factorMin = estimatedLengthMin / length
                factorMax = estimatedLengthMax / length
            elif cell.dim == 0:
                factorMin = factorMax = 1
            else:
                raise RuntimeError(f"Wrong dim {cell.dim}.")
            if factorMin > 1 or factorMax < 1:
                factor = (factorMax - factorMin)*np.random.random() + factorMin
            else:
                factor = 1
            cellVectors = np.asarray(cellVectors)
            cellVectors[np.nonzero(self._pbc)] *= factor
            cell = Cell(cellVectors, self._pbc)
        elif self._listOfReconstructions:
            factor = self.calcCompositionVolume(composition, pressure) / cell.getVolume()
            reconstruction = self._getRandomReconstruction(factor)
            cell = Cell(reconstruction.dot(cell.getCellVectors()), cell.getPBC())
        return cell

    def getCellVolume(self, composition, pressure):
        """
        Either return predefined volume for fixed volume calculation or calculates it under specific pressure.

        :param composition: dictionary like object defining composition.
        :param pressure: external pressure.

        :return: volume of unit cell.
        """
        return self._volume if self._volume is not None else self.calcCompositionVolume(composition, pressure)

    def calcCompositionVolume(self, composition, pressure):
        """
        The function calculates a volume of the given composition at the target pressure.

        :type composition: Mapping
        :param composition: Composition for which the volume is to be estimated.
        :param pressure: external pressure.

        :rtype: float
        :return: volume
        """
        return sum(self.volumeEstimator.calcAtomVolume(sym, pressure) * amount for sym, amount in composition.items())


    def getHybridCell(self, cell1, cell2, fraction):
        """
        Creates hybrid of two unit cells.

        :param cell1:
        :param cell2:
        :param fraction:

        :return:
        """
        assert 0 <= fraction <= 1
        if self._listOfReconstructions:
            matrix1 = np.round(self._cell.decomposeCell(cell1))
            matrix2 = np.round(self._cell.decomposeCell(cell2))
            idx = np.linalg.det([matrix1, matrix2]).argmax()
            cellVectors = [cell1, cell2][idx].getCellVectors()
            cell = Cell(cellVectors, self._pbc)
        elif self.dim == 0:
            vacuum = np.power(fraction * cell1.getVolume() + (1 - fraction) * cell2.getVolume(), 1.0/3.0)
            cell = Cell([], self._pbc).getEnvelopeCell(vacuumSize=vacuum)
        else:
            cellParameters = fraction * np.asarray(cell1.getCellParameters()) + \
                         (1 - fraction) * np.asarray(cell2.getCellParameters())
            if self.dim == 3:
                cell = Cell.initFromCellParameters(self._pbc, *cellParameters, axis=self._axis)
                factor = np.power((fraction*cell1.getVolume() + (1 - fraction)*cell2.getVolume()) / cell.getVolume(), 1./3.)
                vacuum = 0
                cellParameters[0:3] *= factor
            elif self.dim == 2:
                a, b, alpha = cellParameters
                cell = Cell.initFromCellParameters(self._pbc, a, b, alpha=alpha, axis=self._axis)
                factor = np.sqrt((fraction*cell1.getArea() + (1 - fraction)*cell2.getArea()) / cell.getArea())
                vacuum = fraction * cell1.getLength() + (1 - fraction) * cell2.getLength()
                cellParameters = np.asarray((a*factor, b*factor, None, alpha, None, None))
            elif self.dim == 1:
                a, = cellParameters
                cell = Cell.initFromCellParameters(self._pbc, a, axis=self._axis)
                factor = (fraction*cell1.getLength() + (1 - fraction)*cell2.getLength()) / cell.getLength()
                vacuum = np.sqrt(fraction * cell1.getArea() + (1 - fraction) * cell2.getArea())
                cellParameters = np.asarray((a*factor, None, None, None, None, None))
            else:
                raise RuntimeError(f"Wrong dim {self.dim}.")
            cell = Cell.initFromCellParameters(self._pbc, *cellParameters, axis=self._axis).getEnvelopeCell(vacuumSize=vacuum)
        return cell

    def getThickness(self):
        """
        :return: Thickness of the 2D system.
        """
        return self._thickness

    def getRadius(self):
        """
        :return: Radius of the 1D or 0D system.
        """
        return self._radius

    def alignCell(self, cell_in):
        """
        :return: Cell with lattice vectors with nonperiodic (for 2D) or periodic (1D) aligned along axis.
        """
        cellVectors = cell_in.getCellVectors()
        if (self.dim == 2) or (self.dim == 1):
            if self.dim == 2:
                a = np.copy(cellVectors[self._pbc.index(0), :])
            else:
                a = np.copy(cellVectors[self._pbc.index(1), :])
            b = np.copy(self._axis)
            a /= np.linalg.norm(a)
            b /= np.linalg.norm(b)
            c = np.dot(a, b)
            v = np.cross(a, b)
            s = np.linalg.norm(a)
            eps = 1e-7
            if s < eps:
                v = np.cross((0, 0, 1), b)
                if np.linalg.norm(v) < eps:
                    v = np.cross((1, 0, 0), b)
                    assert np.linalg.norm(v) >= eps
                elif s > 0:
                    v /= s
            cellVectors[:] = (c * cellVectors -
                        np.cross(cellVectors, s * v) +
                        np.outer(np.dot(cellVectors, v), (1.0 - c) * v))
        return Cell(cellVectors, self._pbc)

    def _getListOfReconstructions(self, reconstructionDegree: Union[int, List, Tuple]):
        assert sum(self._pbc) == 2 # For surfaces only 
        if isinstance(reconstructionDegree, int):
            minReconstruction = reconstructionDegree
            maxReconstruction = reconstructionDegree+1
        elif isinstance(reconstructionDegree, (list, tuple)):
            assert len(reconstructionDegree) == 2
            minReconstruction = reconstructionDegree[0]
            maxReconstruction = reconstructionDegree[1]
        listOfReconstructions = []
        nonzeroPBC = np.nonzero(self._pbc)[0]
        for i in range(1, maxReconstruction + 1):
            for j in range(maxReconstruction + 1):
                for k in range(maxReconstruction + 1):
                    for l in range(1, maxReconstruction + 1):
                        M = np.eye(3)
                        rows = np.array([nonzeroPBC, nonzeroPBC])
                        cols = rows.T
                        M[rows, cols] = np.array([[i, -j], [k, l]])
                        if minReconstruction <= np.round(np.linalg.det(M)) < maxReconstruction and M[nonzeroPBC[0]].dot(M[nonzeroPBC[1]]) == 0:
                            listOfReconstructions.append(M)
        return listOfReconstructions
    
    def _getRandomReconstruction(self, factor=None):
        if self._listOfReconstructions:
            if factor:
                mask = np.linalg.det(self._listOfReconstructions) > factor
            else:
                mask = np.ones(len(self._listOfReconstructions), dtype=bool)
            assert sum(mask) # check if any suitable reconstruction found for this composition
            idx = np.random.choice(np.arange(len(self._listOfReconstructions))[mask])
            return self._listOfReconstructions[idx]
        else:
            return np.eye(3)

    @staticmethod
    def volume(system: dict):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculated volume of system.
        """
        return system['cell'].getVolume()

    @staticmethod
    def area(system: dict):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculated area of system.
        """
        return system['cell'].getArea()
    
    @staticmethod
    def length(system: dict):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculated length of system.
        """
        return system['cell'].getLength()


    def symmetry(self, system: dict):
        """
        For using in **Fitness** infrastructure

        :param system: dictionary describing system.

        :return: calculated symmetry of system.
        """
        cell = system['cell']
        molecules = system['molecules']
        structure, disassembler = type(molecules[0]).assemble(molecules, cell)
        lattice = cell.getCellVectors()
        coordinates = structure.getFractionalCoordinates()
        numbers = [el.z for el in structure.getAtomTypes()]
        spacegroup = spglib.get_spacegroup((lattice, coordinates, numbers), symprec=self.symTolerance)
        if cell.getPBC() == (1, 1, 1) and spacegroup is not None:
            symmetry = '{:7s} {:4s}'.format(*[str(x) for x in spacegroup.split()])
        else:
            symmetry = None
        return symmetry


class Cell:
    """
    Class representing unit cell of atomic structure.
    """

    def __init__(self, cellVectors, pbc):
        """

        :param cellVectors: 3*3 array of cell vectors.
        :param pbc: periodic boundary conditions in each direction.

        """
        self._pbc = pbc
        self._antipbc = tuple((~np.asarray(pbc, dtype=bool)).tolist())
        self.dim = sum(pbc)
        cellVectors = np.asarray(cellVectors, dtype=float).reshape((-1,3))
        if len(cellVectors) == 3:
            self._cellVectors = cellVectors
        elif len(cellVectors) == self.dim:
            self._cellVectors = np.eye(3)
            self._cellVectors[np.nonzero(self._pbc)] = cellVectors
            if self.dim == 2:
                vec1, vec2 = cellVectors
                axis = np.cross(vec1, vec2)
                axis /= np.linalg.norm(axis)
                self._cellVectors[np.nonzero(self._antipbc)] = np.asarray([axis], dtype=float)
            elif self.dim == 1:
                axis, = cellVectors
                axis /= np.linalg.norm(axis)
                vec1 = np.random.random(3)
                vec1 -= axis*np.dot(axis, vec1)
                vec1 /= np.linalg.norm(vec1)
                vec2 = np.cross(vec1, axis)
                self._cellVectors[np.nonzero(self._antipbc)] = np.asarray([vec1, vec2], dtype=float)
        else:
            raise ValueError(f"Provided {len(cellVectors)} cell vectors when dim is {self.dim}.")

    @staticmethod
    def initFromCellParameters(pbc, a, b=None, c=None, alpha=None, beta=None, gamma=None, axis=None):
        """
        Alternative constructor using cell parameters.

        :param pbc:
        :param a:
        :param b:
        :param c:
        :param alpha:
        :param beta:
        :param gamma:
        :param axis:

        :return:
        """
        dim = sum(pbc)
        if dim == 3:
            assert axis is None and b is not None and c is not None and alpha is not None and beta is not None and gamma is not None
            alpha, beta, gamma = np.pi / 180 * np.asarray((alpha, beta, gamma), dtype=float)
            va = np.array([a, 0, 0])
            vb = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
            cx = np.cos(beta)
            cy = (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
            cz = (1. - cx ** 2 - cy ** 2) ** 0.5
            vc = c * np.array([cx, cy, cz])
            return Cell(np.vstack((va, vb, vc)), pbc)
        elif dim == 2:
            assert axis is not None and b is not None and c is None and alpha is not None and beta is None and gamma is None
            alpha= np.pi / 180 * alpha
            axis = np.asarray(axis, dtype=float)
            axis /= np.linalg.norm(axis)
            va = np.random.random(3)
            va -= axis*np.dot(axis, va)
            va *= a/np.linalg.norm(va)
            vb = np.dot(Rotation.from_rotvec(alpha*axis).as_matrix(), va)
            vb *= b/np.linalg.norm(vb)
            return Cell(np.vstack((va, vb)), pbc)
        elif dim == 1:
            assert axis is not None and b is None and c is None and alpha is None and beta is None and gamma is None
            axis = np.asarray(axis)
            va = a*axis/np.linalg.norm(axis)
            return Cell([va], pbc)
        else:
            raise RuntimeError(f"Wrong pbc {pbc}.")


    def getCellVectors(self):
        """
        :return: 3*3 array of cell vectors.
        """
        return copy(self._cellVectors)

    def getCellVectorsPBC(self):
        """
        :return: x*3 array of periodic cell vectors, where 0 <= x <= 3.
        """
        return copy(self._cellVectors)[np.nonzero(self._pbc)]

    def getPBC(self):
        """
        :return: periodic boundary conditions in each direction.
        """
        return self._pbc

    def getCellParameters(self):
        """
        :return:  tuple of cell parameters: a, b, c, alpha, beta, gamma
        """
        if self.dim == 3:
            a = np.linalg.norm(self._cellVectors[0, :])
            b = np.linalg.norm(self._cellVectors[1, :])
            c = np.linalg.norm(self._cellVectors[2, :])
            alpha = 180 / np.pi * np.arccos(np.dot(self._cellVectors[1, :], self._cellVectors[2, :]) / (b * c))
            beta = 180 / np.pi * np.arccos(np.dot(self._cellVectors[0, :], self._cellVectors[2, :]) / (a * c))
            gamma = 180 / np.pi * np.arccos(np.dot(self._cellVectors[0, :], self._cellVectors[1, :]) / (a * b))
            return a, b, c, alpha, beta, gamma
        elif self.dim == 2:
            cellVectors = self.getCellVectorsPBC()
            a = np.linalg.norm(cellVectors[0, :])
            b = np.linalg.norm(cellVectors[1, :])
            alpha = 180 / np.pi * np.arccos(np.dot(cellVectors[0, :], cellVectors[1, :]) / (a * b))
            return a, b, alpha
        elif self.dim == 1:
            a = np.linalg.norm(self.getCellVectorsPBC()[0])
            return a,
        else:
            raise RuntimeError(f"Wrong dim {self.dim}.")

    def getVolume(self):
        """
        :return: unit cell volume if cell is 3D periodic.
        """
        assert self.dim == 3 or self.dim == 0
        return np.abs(np.linalg.det(self._cellVectors))

    def getArea(self):
        """
        :return: unit cell area if cell is 2D periodic.
        """
        assert self.dim == 2 or self.dim == 1
        if self.dim == 2:
            nonzeroPBC = np.nonzero(self._pbc)[0]
            nonzeroCellVectors = self._cellVectors[nonzeroPBC]
        else:
            nonzeroPBC = np.nonzero(self._antipbc)[0]
            nonzeroCellVectors = self._cellVectors[nonzeroPBC]
        return np.linalg.norm(np.cross(nonzeroCellVectors[0], nonzeroCellVectors[1]))

    def getLength(self):
        """
        :return: unit cell length if cell is 1D periodic.
        """
        assert self.dim == 2 or self.dim == 1
        if self.dim == 1:
            nonzeroPBC = np.nonzero(self._pbc)[0]
            nonzeroCellVectors = self._cellVectors[nonzeroPBC]
        else:
            nonzeroPBC = np.nonzero(self._antipbc)[0]
            nonzeroCellVectors = self._cellVectors[nonzeroPBC]
        return np.linalg.norm(nonzeroCellVectors[0])

    def getMaxNumSlabs(self, axis, N):
        """
        Calculates maximal number of choices of slabs origins in give direction.
        :param axis: direction axis.
        :param N: number of atoms/molecules in cell.
        :return: maximal number of choices of slabs origins.
        """
        if self.dim == 3:
            volume = self.getVolume()
            if axis == 0:
                L = volume / np.linalg.norm(np.cross(self._cellVectors[1, :], self._cellVectors[2, :]))
            elif axis == 1:
                L = volume / np.linalg.norm(np.cross(self._cellVectors[0, :], self._cellVectors[2, :]))
            elif axis == 2:
                L = volume / np.linalg.norm(np.cross(self._cellVectors[0, :], self._cellVectors[1, :]))
            else:
                raise ValueError(f"Wrong axis {axis}.")
            Lchar = 0.5 * np.power((volume / N), (1 / 3)) # average 'radius' of a molecule in the cell
            Nmax =  L / Lchar
        elif self.dim == 2:
            if self._pbc[axis] == 0:
                L = self.getLength()
                Lchar = 0.5 * np.power((L * self.getArea() / N), (1 / 3))
                Nmax = L / Lchar
            else:
                volume = self.getArea() * self.getLength()
                if axis == 0:
                    L = volume / np.linalg.norm(np.cross(self._cellVectors[1, :], self._cellVectors[2, :]))
                elif axis == 1:
                    L = volume / np.linalg.norm(np.cross(self._cellVectors[0, :], self._cellVectors[2, :]))
                elif axis == 2:
                    L = volume / np.linalg.norm(np.cross(self._cellVectors[0, :], self._cellVectors[1, :]))
                else:
                    raise ValueError(f"Wrong axis {axis}.")
                Lchar = 0.5 * np.power((volume / N), (1 / 3))
                Nmax = L / Lchar
        elif self.dim == 1:
            if self._pbc[axis] == 1:
                L = self.getLength()
                Lchar = 0.5 * np.power((L * self.getArea() / N), (1 / 3))
                Nmax = L / Lchar
            else:
                area = self.getArea()
                L = 2 * np.sqrt(area / np.pi)
                Lchar = 0.5 * np.power((area * self.getLength() / N), (1 / 3))
                Nmax = L / Lchar
        elif self.dim == 0:
            Nmax =  0.5 * np.power((6 * np.pi ** 2 * N), (1 / 3))
        else:
            raise RuntimeError(f"Wrong dim {self.dim}.")
        return Nmax

    def getCornersCoordinates(self):
        """
        :return: 8*3 array of absolute coordinates of each corner of unit cell.
        """
        coordinates = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    coordinates.append(i*self._cellVectors[0] + j*self._cellVectors[1] + k*self._cellVectors[2])
        return np.asarray(coordinates, dtype = float)

    def cartesianToFractional(self, coordinates):
        """
        Converts cartesian coordinates to fractional.

        :param coordinates: N*3 array of cartesian coordinates

        :return: N*3 array of fractional coordinates.
        """
        return np.linalg.solve(self._cellVectors.T, coordinates.T).T

    def fractionalToCartesian(self, coordinates):
        """
        Converts fractional coordinates to cartesian.

        :param coordinates: N*3 array of fractional coordinates.

        :return: N*3 array of cartesian coordinates
        """
        return np.dot(self._cellVectors.T, coordinates.T).T

    def fractionalToCartesianOperator(self, operator):
        """
        TODO convert some operator matrix defined in fractional space to transformation object in cartesian space.

        :param operator:

        :return:
        """
        return operator

    def getWrapedCartesianCoordinates(self, coordinates):
        """
        Translates all coordinates to their image within unit cell via cell vectors in cartesian space.

        :param coordinates: cartesian coordinates

        :return: cartesian coordinates wrapped to unit cell.
        """
        return self.fractionalToCartesian(self.getWrapedFractionalCoordinates(self.cartesianToFractional(coordinates)))

    def getWrapedFractionalCoordinates(self, coordinates):
        """
        Translates all coordinates to their image within unit cell via cell vectors in fractional space.

        :param coordinates: fractional coordinates

        :return: fractional coordinates wrapped to unit cell.
        """
        return np.divmod(coordinates, 1/np.asarray(self._pbc, dtype=float))[1]

    def randomTransformation(self):
        """
        Creates transformation object which respects unit cell. I.e. origin is moved only in periodic directions.
        And structure axis if exists is not moved.
        Structure axis is periodic axis in 1D case and axis orthogonal to two periodic axes in 2D case,

        :return: transformation object.
        """
        pbcVectorsCart = self.getCellVectorsPBC()
        pbcSum = len(pbcVectorsCart)
        transVec = np.dot(np.random.rand(pbcSum), pbcVectorsCart)
        if pbcSum == 0:
            rotMatrix = Rotation.random().as_matrix()
        elif pbcSum == 1:
            axis = pbcVectorsCart[0]
            rotVec = np.pi * np.random.random() * axis / np.linalg.norm(axis)
            rotMatrix = Rotation.from_rotvec(rotVec).as_matrix()
        elif pbcSum == 2:
            axis = np.cross(pbcVectorsCart[0,:], pbcVectorsCart[1,:])
            rotVec = np.pi * np.random.random() * axis / np.linalg.norm(axis)
            rotMatrix = Rotation.from_rotvec(rotVec).as_matrix()
        else:
            rotMatrix = Rotation.random().as_matrix()
        centerCellVec = self.fractionalToCartesian(np.array([0.5, 0.5, 0.5]))
        transVec = transVec - centerCellVec + np.dot(rotMatrix.T, centerCellVec)
        return Transformation.fromMatrix(rotMatrix, np.dot(rotMatrix, transVec))

    def getFittedTransformations(self, initialCoordinates, cell):
        """
        For given set of coordinates and cell object, calculates transformations
        which translate each position into its image within this unit cell along periodic cell vectors of given unit cell.

        :param initialCoordinates: 3 vector of coordinates which need to be translated.
        :param cell: cell object containing periodic vectors along which translation should be done.

        :return: list of **Transformation** objects.
        """
        inds = np.nonzero(cell.getPBC())
        vectors = cell.getCellVectors()[inds]
        minAndMax = np.asarray([(np.min(coords), np.max(coords))
                                for coords in cell.cartesianToFractional(self.getCornersCoordinates()).T[inds]],
                               dtype = float)\
                    - cell.cartesianToFractional(initialCoordinates)[inds]
        minAndMax = np.asarray(np.ceil(minAndMax), dtype=int).reshape((-1,2))
        if np.all(minAndMax[:,1] > minAndMax[:,0]):
            closeShifts = []
            for minCoordinate, maxCoordinate in minAndMax:
                if closeShifts:
                    newCloseShifts = []
                    for shift in closeShifts:
                        newCloseShifts.extend([shift + (i,) for i in range(minCoordinate, maxCoordinate)])
                    closeShifts = newCloseShifts
                else:
                    closeShifts = [(i,) for i in range(minCoordinate, maxCoordinate)]
            closeShifts = np.asarray(closeShifts, dtype=int)
            closeCoordinates = initialCoordinates.reshape((1,3)) + np.dot(closeShifts, vectors)
        else:
            closeCoordinates = []

        fittedCoordinates = [coord for coord in closeCoordinates if (np.all(0. <= self.cartesianToFractional(coord)) and
                                                                     np.all(self.cartesianToFractional(coord) < 1.))]

        return [Transformation.fromRotVector([0.,0.,0.], finalCoordinates - initialCoordinates)
                for finalCoordinates in fittedCoordinates]

    def getEnvelopeCell(self, coordinates=None, vacuumSize=0):
        """
        :param coordinates: cartesian atomic coordinates
        :param vacuumSize: vacuum distance added along cell vector

        :return:  new cell object, corresponding to
        """
        newCellVectors = []
        for vector, isPeriodic in zip(self._cellVectors, self._pbc):
            if isPeriodic:
                newCellVectors.append(vector)
            else:
                vector = vector / np.linalg.norm(vector)
                if coordinates is not None:
                    proj = np.dot(coordinates, vector)
                    newCellVectors.append((np.max(proj) - np.min(proj) + vacuumSize) * vector)
                else:
                    newCellVectors.append(vacuumSize * vector)
        return Cell(np.asarray(newCellVectors, dtype=float), self._pbc)

    def center(self, coordinates, affectedDims=None):
        """
        Center atoms in unit cell.

        Centers the coordinates in the unit cell, so there is the same
        amount of vacuum along all cellvectors, specified in affectedDims.

        :param coordinates: list of coordinates of N atoms (Nx3 np.array)
        :param affectedDims: iterable of int/bool/float, specifying the dimensions to act on.
            Default behavior is to center along all vectors, for which pbc is 0

        :return:
        """
        if affectedDims is None:
            affectedDims = 1 - np.asarray(self._pbc, dtype=int)
        affectedDims = np.array(affectedDims).reshape((1, 3))
        fracCoords = self.cartesianToFractional(coordinates)
        shift = np.array([0.5, 0.5, 0.5]) - 0.5 * (np.min(fracCoords, axis=0) + np.max(fracCoords, axis=0))
        newFrac = fracCoords + shift * affectedDims
        return self.fractionalToCartesian(newFrac)

    def decomposeCell(self, other):
        """
        Decompose cell vectors of given unit cell as linear composition of cell vectors of this unit cell.

        :param other: unit cell to decompose.

        :return: 3*3 matrix of decomposition coefficients.
        """
        assert self._pbc == other.getPBC()
        matrix = np.eye(3)
        inds = np.nonzero(self._pbc)
        matrix[inds] = np.linalg.solve(self.getCellVectors().T,other.getCellVectors().T).T[inds]
        return matrix
