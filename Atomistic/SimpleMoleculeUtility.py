import numpy as np
from itertools import combinations_with_replacement

from .Element import Element
from .Transformation import Transformation


class SimpleMoleculeUtility(object):
    def __init__(self, atomicStructureFactory):
        self.atomicStructureFactory = atomicStructureFactory

    def getMinDistances(self, molecules, cell):
        """
        Check if the structure meets the minimal distance constraints provided with Minimal Distances Matrix.

        :rtype: bool
        :return: True if the structure meet the constraint, False otherwise.
        """

        # positions = self.atoms.get_positions()
        # i_init, j_init, vect = primitive_neighbor_list(quantities='ijD', pbc=self.atoms.pbc, cell=self.atoms.get_cell(complete=True),
        #                                          positions=positions, cutoff=minDistMatrix,
        #                                          numbers=self.atoms.numbers, use_scaled_positions=False)
        #
        # # whether system is molecular or not
        # # Check whether all pairs of atoms, which are closer than minDistMatrix and are related to the same molecule
        # for i, j, v in zip(i_init, j_init, vect):
        #     inMolecule = False
        #     for mol in self._molecules:
        #         if i in mol and j in mol and np.allclose(v, positions[j] - positions[i]):
        #             inMolecule = True
        #             break
        #     if not inMolecule: return False
        # return True

        structure, disassembler = self.atomicStructureFactory.assemble(molecules, cell)
        N = len(structure)
        if N < 2:
            return True
        actualDistances = structure.getAllDistances()
        constNeighbours = np.vstack([np.eye(3), -np.eye(3)])
        for inds, molecule in zip(disassembler.indices, molecules):
            distVectorsMatrix = molecule.getAllPairVectors()
            for i, distVectorsRow in enumerate(distVectorsMatrix):
                for j, vect in enumerate(distVectorsRow):
                    vect = cell.cartesianToFractional(vect)
                    if np.all(np.abs(vect) < 1.0):
                        dists = np.linalg.norm(vect + constNeighbours, axis=1)
                        distVectorsMatrix[i,j] = cell.fractionalToCartesian(vect + constNeighbours[np.argmin(dists)])
            actualDistances[tuple(np.meshgrid(inds, inds))] = np.linalg.norm(distVectorsMatrix, axis=2)

        return structure.getAtomTypes(), actualDistances


    def molecule_CN(self, molecule):
        """
        Method which roughly (very roughly!!!) estimates the coordination numbers of a molecule.

        :type i: int
        :param i: Molecule index.
        :rtype: numpy array
        :return: Array of coordination numbers.
        """
        radiu = np.array([Element(atom).covalent_radius for atom in molecule.get_chemical_symbols()])
        CN = np.fromiter((len(neighbours) for neighbours in find_pair(molecule.coordinates, radiu)), dtype=int)
        return CN

    def rotatePrinciple(self, molecule, axis: int, angle: float):
        """
        Rotate the *i* th molecule with respect to *axis* =[0,1,2] principle axis by the angle *angle* .

        :type i: int
        :param i: index of molecule to rotate.
        :type axis: int
        :param axis: index of axis [0,1,2] with respect to which the molecule should be rotated.
        :type angle: float
        :param angle: angle by which the molecule should be rotated.
        """
        assert axis < 3
        values, vectors = molecule.getPrincipleAxes()
        ref = np.max(values)
        value = values[axis]
        vector = vectors[axis]
        vector /= np.linalg.norm(vector)
        if value > 0.0001:
            angle *= ref/value
            vector *= angle
        else:
            vector *= 0
        return Transformation.fromRotVector(vector, [0.,0.,0.,]).transform(molecule)

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
    n_atom = len(radii)
    # maximum 6 coordination, 7 gives the coordination number
    # pair = np.zeros((n_atom, N_max), dtype=int)
    pair = [[] for x in radii]

    for i in range(n_atom):
        for j in range(i + 1, n_atom):
            if np.linalg.norm(coor[i] - coor[j]) < 1.2 * (radii[i] + radii[j]):
                pair[i].append(j)
                pair[j].append(i)

    # we assume there is no isolated atom
    if n_atom > 1:
        for i in range(n_atom):
            if len(pair[i]) == 0:
                print('atom_{} is not connected to any other atom'.format(i))
                print('Please check your MOL file again. Serious WARNING.... ')

    return pair
