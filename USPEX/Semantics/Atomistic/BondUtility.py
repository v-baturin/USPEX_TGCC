from abc import ABC, abstractmethod

class BondUtility(ABC):

    @abstractmethod
    def isConnected(self, structure, cutoff=None) -> bool:
        """
        checks if structure is connected, with bonds graph based on thresholds based on atom valence radii
        Rcutoff(type_i, type_j) = checkConnectivityCutoffFactor * (Rval(type_i) + Rval(type_j))

        @param structure: AtomicStructure instance
        @param cutoff: str, dict, float, int
        @return: bool
        """
        pass

    @abstractmethod
    def buildCutoffDict(self, structure, cutoff=None):
        """
        Builds dictionary of cutoffs compatible with ase.neighborlist.primitive_neighbor_list. Each dict item
        corresponds to a pair of elements with values of threshold bond distances
        Cutoffs are build as:
            1. 'strong' cutoff based on classic USPEX checkConnectivity
            2. 'vdw' cutoff as sum of van der Waals radii
        @param structure: AtomicStructure instance
        @param cutoffParameter: float or int, factor or increment depending on cutoffType
        @return: dict of cutoffs consistent with  cutoff dict parameter
        """
        pass

    @abstractmethod
    def getAllBondsInCutoff(self, structure, cutoff):
        """
        Gets all bonds in structure, whose lengths do not exceed cut-off
        Cut-off parameter is passed to ase.neighborlist.primitive_neighbor_list, hence its format
        @param structure: AtomicStructure instance
        @param cutoff: float or dict or list or array
                Cutoff for neighbor search. It can be:

                    * A single float: This is a global cutoff for all elements.
                    * A dictionary: This specifies cutoff values for element
                      pairs. Specification accepts element numbers of symbols.
                      Example: {(1, 6): 1.1, (1, 1): 1.0, ('C', 'C'): 1.85}
                    * A list/array with a per atom value: This specifies the radius of
                      an atomic sphere for each atom. If spheres overlap, atoms are
                      within each other's neighborhood. See :func:`~ase.neighborlist.natural_cutoffs`
                      for an example on how to get such a list.
        @return: (strongBonds: List, weakBonds: List) (separated according to goodBonds-based criteria)
        """
        pass

    @abstractmethod
    def getMinimalGraphBonds(self, structure) -> list:
        '''
        Calculates bond graph minimal for the structure to be 3D connected.

        :param structure: AtomicStructure instance
        :return: bond graph
        '''
        pass

    @abstractmethod
    def calcHardness(self, structure, bonds) -> float:
        '''
        Calculate hardness for a given structure from bond hardness model.
        See http://han.ess.sunysb.edu/hardness/ for details.

        :param system:
        :return H: hardness (GPa).
        '''
        pass

    @abstractmethod
    def calcSoftModes(self, system, bonds, kVector0):
        '''
        The function calculates vibrational modes based on the dynamic matrix (D) constructed from bond hardness model.

        K-vector should be in A^-1, very important!
            reciprocal_lat_x = 2pi*(lat_y X lat_z)/V
            k_abs = k*reciprocal_lattice

        :param system:
        :param kVector0: K-vector.
        :return freq: frequencies of all modes.
        :return eigvector: eigenvector of all modes.
        '''
        pass

    @staticmethod
    @abstractmethod
    def calcCoordinationNumbers(structure):
        pass

    @abstractmethod
    def getDistances(self, symbols, pressure):
        """
        For given array of symbols generates matrix of minimal distances.
        If minimal distance for pair of symbols is not predefined calculates it using volumeUtility.

        :param symbols: N array of symbols
        :param pressure: external pressure.

        :return: N*N array of minimal distances.
        """
        pass
