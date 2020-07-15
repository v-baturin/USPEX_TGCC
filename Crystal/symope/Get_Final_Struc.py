import numpy as np

from .latConverter import latConverter


def Get_Final_Struc(newLattice, candidate, permutationBack):
    """
    The function returns back correct sequence of the axes.
    :param newLattice: input lattice.
    :param candidate: input coordinates.
    :param permutationBack: sequence of the axes.
    :return Lattice: output lattice.
    :return candidate: output coordinates.
    """

    newLattice1 = latConverter(newLattice)
    newLattice2 = np.copy(newLattice1)
    candidate1 = np.copy(candidate)

    for axis in range(3):
        newLattice2[0, axis] = newLattice1[0, permutationBack[axis]]
        candidate[:, axis] = candidate1[:, permutationBack[axis]]

    Lattice = latConverter(newLattice2)

    return Lattice, candidate
