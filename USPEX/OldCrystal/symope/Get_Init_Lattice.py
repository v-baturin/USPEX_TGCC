import numpy as np

from .latConverter import latConverter


def Get_Init_Lattice(lat, permutation):
    """
    The function is to get initial lattice.
    :param lat: input lattice (can be a number (= volume), 3x3 or 1x6 arrays).
    :param permutation: permutation.
    :return lat1: output lattice (1x6, angles in degrees).
    """

    if type(lat) not in [list, np.ndarray]:  # random lattice, lat =  volume
        lat1 = np.zeros((1, 6))
        lat1[0, 0:3] = np.random.random(3) + 0.5

        x = 0.0
        while x < 0.3:
            lat1[0, 3:6] = (np.random.random(3) * 120. + 30.) * np.pi / 180.  # fortran works with degrees
            x = 1. - np.cos(lat1[0, 3]) ** 2 - np.cos(lat1[0, 4]) ** 2 - np.cos(lat1[0, 5]) ** 2 + 2. * np.cos(
                lat1[0, 3]) * np.cos(lat1[0, 4]) * np.cos(lat1[0, 5])

        ratio = lat / np.linalg.det(latConverter(lat1))
        lat1[0, 3:6] *= (180. / np.pi)  # fortran works with degrees, convert to degrees
        lat1[0, 0:3] *= (ratio ** (1. / 3.))

    else:  # fixed lattice, lat = lattice
        lat = np.asarray(lat)

        if lat.shape == (3, 3):  # INPUT is 3x3 -> convert it to 1x6
            lat1 = latConverter(lat)
        else:  # INPUT is 1*6
            lat1 = np.copy(lat)

        lat_tmp = np.copy(lat1)
        for axis in range(3):
            lat1[0, axis] = lat_tmp[0, permutation[axis]]

        lat1[0, 3:6] *= (180.0 / np.pi)  # fortran works with degrees, convert to degrees

    return lat1


if __name__ == "__main__":
    from pprint import pprint

    # ------------------------------------

    l = 123.320730852
    perm = np.asarray([0, 1, 2])
    l1 = Get_Init_Lattice(l, perm)
    pprint(l1)
    print('')

    # ------------------------------------

    l = np.asarray([[2.474, 8.121, 6.138, 90.0, 90.0, 90.0]])
    l[0, 3:6] *= (np.pi / 180.0)
    perm = np.asarray([2, 1, 0])
    l1 = Get_Init_Lattice(l, perm)
    pprint(l1)
    print('')

    # ------------------------------------

    l = np.asarray([
        [2.474, 0.0, 0.0],
        [0.0, 8.121, 0.0],
        [0.0, 0.0, 6.138],
    ])
    perm = np.asarray([2, 1, 0])
    l1 = Get_Init_Lattice(l, perm)
    pprint(l1)
    print('')

    # ------------------------------------

    print('')
