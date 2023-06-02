import numpy as np


def GetPermutation(nsym):
    """
    For monoclinic and orthorhombic groups we do lattice vector exchange/swap
    so we can generate non conventional symmetries too (P2mm, Pm2m, etc.).
    They are changed back after Stokes' code generates the 'conventional' structure
    for monoclinic only a and c are swapped, for orthorombic - any combination.
    :param nsym: space group name.
    :return permutation: permutation.
    :return permutationBack: back permutation.
    """

    if 2 < nsym <= 15:  # monoclinic
        rnd = np.random.rand()
        if rnd > 0.5:
            permutation = np.asarray([2, 1, 0])  # only swap x and z coordinates
            permutationBack = np.copy(permutation)
        else:
            permutation = np.asarray([0, 1, 2])  # leave as is
            permutationBack = np.copy(permutation)

    elif 15 < nsym < 75:  # orthorhombic
        # Possible permutations:
        ListPerm = np.asarray([
            [0, 1, 2],
            [2, 1, 0],
            [0, 2, 1],
            [1, 0, 2],
            [2, 0, 1],
            [1, 2, 0],
        ])

        id = np.random.randint(6)
        permutation = np.copy(ListPerm[id, :])  # all swaps allowed
        if id == 4:
            permutationBack = np.copy(ListPerm[5, :])  # exchange is not reversible: (3 1 2) -> (2 3 1)
        elif id == 5:
            permutationBack = np.copy(ListPerm[4, :])  # exchange is not reversible: (2 3 1) -> (3 1 2)
        else:
            permutationBack = np.copy(permutation)

    else:
        permutation = [0, 1, 2]  # no swaps
        permutationBack = np.copy(permutation)

    return permutation, permutationBack


if __name__ == "__main__":
    nsym = 16
    perm, permBack = GetPermutation(nsym)

    print('')
