"""
USPEX.Common.Atomistic.mol.find_pair
====================================

Function that finds the neighbours of each atom

.. codeauthor:: Artem Samtsevich <samtsevichartem@gmail.com>
"""

from numpy.linalg import norm


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
            if norm(coor[i] - coor[j]) < 1.2 * (radii[i] + radii[j]):
                pair[i].append(j)
                pair[j].append(i)

    # we assume there is no isolated atom
    if n_atom > 1:
        for i in range(n_atom):
            if len(pair[i]) == 0:
                print('atom_{} is not connected to any other atom'.format(i))
                print('Please check your MOL file again. Serious WARNING.... ')

    return pair
