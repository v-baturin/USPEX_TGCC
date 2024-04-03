from typing import Counter

from .Generics import VectorN3, Vector3, SquareMatrixN, SquareMatrixN3, Tiling3
from .Cell import Cell
from .Element import Element

class AtomicStructure:
    """
    This class describe pure geometry of any atomic structure including crystals, molecules, nanoparticles etc.
    Such pure geometry consists of coordinates of atoms and their types. In case of periodic structures it naturally
    includes cell object which describes periodicity.
    """

    @staticmethod
    def initFromFractionalCoordinates(atomTypes: list[Element],
                                      coordinates: VectorN3,
                                      cell: Cell,
                                      edges: list[tuple[int, int]] = None,
                                      faces: list[tuple[int, int, int]] = None
                                      ) -> 'AtomicStructure':
        """
        Alternative constructor. Calculates cartesian coordinates from fractional coordinate and given **Cell** object.

        :param atomTypes: list of types of atoms in structure.
        :param coordinates: array of fractional coordinates of atoms. Its first dimension must coincide with size of atomTypes.
        :param cell: **Cell** object for periodic structures.

        """
        pass

    def getAligned(self, axis: Vector3) -> 'AtomicStructure':
        """
        Creates another **AtomicStructures** instance with the same cell parameters and atomic coordinates
         but aligned along given axis.

        :param axis: 3-vector along which the new structre will be aligned.
        :raises RuntimeError: if used on 0D or 3D structure.
        :return: new **AtomicStructures** instance.
        """
        pass

    def createAtNewCoordinates(self, coordinates: VectorN3) -> 'AtomicStructure':
        pass

    def __len__(self) -> int:
        pass

    def getAtomTypes(self) -> list[Element]:
        """
        :return: copy of atom types sequence.
        """
        pass

    def getCell(self) -> Cell:
        """
        :return: copy of associated **Cell** object.
        """
        pass

    def getCartesianCoordinates(self) -> VectorN3:
        """
        :return: copy of cartesian coordinates of atoms in structure.
        """
        pass

    def getFractionalCoordinates(self) -> VectorN3:
        """
        :return: calculates and returns fractional coordinates of atoms if structure has assosiated **Cell**  object.
        :raises: RuntimeError if structure does not have associated **Cell** object.
        """
        pass

    def getComposition(self) -> Counter:
        """
        :return: calculates and returns composition of the structure as **Counter** object.
        """
        pass

    def getFormula(self) -> str:
        """
        :return: calculates and returns formula of the structure.
        """
        pass

    def getAllDistances(self) -> SquareMatrixN:
        """
        :return: N*N matrix of pairwise distances between atoms, where N number of atoms in structure unit.
        """
        pass

    def getAllPairVectors(self) -> SquareMatrixN3:
        """
        :return: N*N matrix of pairwise vectors between atoms, where N number of atoms in structure unit.
        """
        pass

    def getCenterOfMassCartesianCoordinates(self) -> Vector3:
        """
        :return: calculates cartesian coordinates of geometrical center of atoms of structure unit.
        """
        pass

    def getCenterOfMassFractionalCoordinates(self) -> Vector3:
        """
        :return: calculates fractional coordinates of geometrical center of atoms of structure unit.
        """
        pass

    def getTrigonalizedCellStructure(self) -> 'AtomicStructure':
        pass

    def getRectifiedCell(self) -> Cell:
        """
        :return: **Cell** object depending on dimensionality.

            0d: Cell made of unit principal eigenvectors

            1d: Keep periodic vector from original cell, The other two are perpendicular to it,
            directed along principal directions of
            a structure, flatten along periodic vector

            2d: Keep periodic vectors from original cell. The third is a unity vector perpendicular to those two.

            3d: Returns original Cell
        """
        pass

    def makeSupercell(self, matrix: Tiling3) -> Cell:
        """
        For periodic structures constructs supercell representation of the same structure.
        I.e. it has associated **Cell** object multiple of initial **Cell** object
        and concatenated arrays of coordinates and atom types from blocks constituting the supercell.
        New cell vectros are given by formula

        >>> newCellVectors = matrix.dot(self.getCell().getCellVectors())

        :param matrix: 3*3 array of integers defining the supercell.

        :return: new **AtomicStructure** object.
        """
        pass

    def getPerturbatedStructure(self, fixedIndices: list[int]) -> 'AtomicStructure':
        pass
