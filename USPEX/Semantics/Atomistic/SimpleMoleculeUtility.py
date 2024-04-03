from abc import ABC, abstractmethod
from typing import Counter

class SimpleMoleculeUtility(ABC):
    """
    Utility providing methods for work with simple molecules.
    """

    @abstractmethod
    def populateStructure(self, cell, operations):
        """
        Creates list of molecules by placing corresponding molecule in place specified by map *coordinates*
        in orientation specified by map operations.

        :param cell: unit cell.
        :param operations: map {<molecule_symbols> : <orientation>}

        :return: list of molecules.
        """
        pass

    @abstractmethod
    def determineMoleculeType(self, molecule):
        """
        Assuming every type of molecules defined in calculation has unique formula,
        determines molecule symbol of given molecule.

        :param molecule: atomic structure representing molecule.

        :return: molecule symbol.
        """
        pass

    @abstractmethod
    def getElementalComposition(self, composition) -> Counter:
        """
        For given molecular composition {<molecule_symbol> : <amount>} calculates elemental composition {<element> : <amount>}.

        :param composition: molecular composition.

        :return: elemental composition.
        """
        pass

    @abstractmethod
    def checkMinDistances(self, entry, minDistMatrix) -> bool:
        """
        Calculates minimal distances between atoms excluding intramolecular distances.

        :param molecules: list of molecules.
        :param cell: unit cell.

        :return: matrix of distances between atoms.
        """
        pass

    @staticmethod
    @abstractmethod
    def rotationClearance(inertiaValues):
        """
        Calculates relative mobility of structure around axes with given inertia values.

        :param inertiaValues: inertia values of axes.

        :return: relative clearances around axes.
        """
        pass

    @classmethod
    @abstractmethod
    def rotateFlexDiherdal(cls, molecule, i: int, angle: float):
        """
        Rotate the *i* th flexible dihedral angle of molecule by the angle *angle* .

        :param i: index of flexible dihedral angle to rotate.
        :param angle: angle by which the dihedral should be rotated.
        """
        pass

    @classmethod
    @abstractmethod
    def molecule_CN(cls, molecule):
        """
        Method which roughly (very roughly!!!) estimates the coordination numbers of a molecule.

        :param molecule: atomic structure representing molecule.

        :return: array of coordination numbers.
        """
        pass

    @staticmethod
    @abstractmethod
    def detectBonds(molecule):
        pass

    @staticmethod
    @abstractmethod
    def zmatrixToCoord(zmatrix, fmt):
        """
        Function that transforms Z-matrix to XYZ coordinates. Remember that the Z-matrix of a molecule is defined
        in spherical coordinates, so we need a lot of transformations from (r, theta, phi) to (x, y, z).

        :type zmatrix: numpy array
        :param zmatrix:
            Z-matrix of a molecule.
        :type fmt: numpy array
        :param fmt:
            for each atom in the molecule are listed the indices of three other atoms,
            with respect to which the parameters of the Z-matrix are calculated.

        :rtype: numpy array
        :return:
            XYZ coordinates of atoms in the molecule.
        """
        pass

    @staticmethod
    @abstractmethod
    def coordToZmatrix(coords, fmt):
        """
        Function that transforms XYZ to Z-matrix coordinates. Remember that the Z-matrix of a molecule is defined
        in spherical coordinates, so we need a lot of transformations from (x, y, z) to (r, theta, phi).

        :type coords: numpy array
        :param coords:
            XYZ coordinates.
        :type fmt: numpy array
        :param fmt:
            for each atom in the molecule are listed the indices of three other atoms,
            with respect to which the parameters of the Z-matrix are calculated.

        :rtype: numpy array
        :return:
            Z-matrix of the molecule.
        """
        pass

    @staticmethod
    @abstractmethod
    def find_pair(coor, radii):
        """
        This function checks all the atom pairs and constructs the neighbor list.
        The bond length is estimated by the covalent radii of the atoms.

        :type coor: numpy array
        :param coor: Nx3 array of atomic coordinates.
        :type radii: numpy array
        :param radii: Nx1 array of atomic radii.

        :rtype: list of list of int
        :return: list with the indices of neighboring atoms for each atom.
        """
        pass
