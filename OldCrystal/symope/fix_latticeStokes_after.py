import numpy as np

from .spaceGroups import SpaceGroups


def fix_latticeStokes_after(nsym, oldLat, oldCoord):
    """
    The function fixed some Stokes problems, for example non-conventional choice of primitive monoclinic lattice.
    :param nsym: symmetry group number.
    :param oldLat: lattice to fix (2D 1x6 NumPy array).
    :param oldCoord: coordinates to fix.
    :return lat: fixed lattice (2D 1x6 NumPy array).
    :return coord: fixed coordinates.
    """

    lat = np.copy(oldLat)
    coord = np.copy(oldCoord)

    sGroup = SpaceGroups().return_group(nsym)  # space group's standard symbol

    if 2 < nsym < 16 and sGroup[0] == 'P':  # monoclinic, primitive
        # 8 7 10 90 90 100 => 7 10 8 90 100 90
        lat[4] = oldLat[5]  # non-90 angle is gamma for Stokes and beta conventionally
        lat[5] = 90.0
        lat[3] = 90.0
        lat[0] = oldLat[1]  # have to swap axes as well
        lat[1] = oldLat[2]  # have to swap axes as well
        lat[2] = oldLat[0]  # have to swap axes as well
        coord[:, 0] = oldCoord[:, 1]  # have to swap coordinates as well
        coord[:, 1] = oldCoord[:, 2]
        coord[:, 2] = oldCoord[:, 0]

    return lat, coord
