__author__ = 'asamtsevich'

from numpy.linalg import norm

def find_pair(coor, radii):
    '''
    This function checks(?) all the atom pairs and constructs the neighbor list
    The bond length is estimated by the covalent radii.
    :param coor: (n*3 matrix)
    :param radii: (n array)
    :return Pair:
    '''
    n_atom = len(radii)
    # Maximum 6 coordination, 7 gives the coordination number
    # Pair = np.zeros((n_atom, N_max), dtype=int)
    Pair = [[] for x in radii]

    for i in range(n_atom):
        for j in range(i+1, n_atom):
            if norm(coor[i] - coor[j]) < 1.2*(radii[i]+radii[j]):
                Pair[i].append(j)
                Pair[j].append(i)

    # We assume there is no isolated atom
    if n_atom > 1:
        for i in range(n_atom):
            if len(Pair[i]) == 0:
                print('atom_{} is not connected to any other atom'.format(i))
                print('Please check your MOL file again. Serious WARNING.... ')

    return Pair
