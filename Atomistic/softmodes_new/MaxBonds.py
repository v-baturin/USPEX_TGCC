#from copy import copy

#import numpy as np

from ase.neighborlist import primitive_neighbor_list
#from scipy.spatial.distance import cdist

from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure
#from USPEX.Common.Atomistic.Element import Element
from ..Bonds import Bond




def MaxBonds_new(system: AtomicStructure, cutoff:float=Bond.MAX_BOND) -> list:

    bonds = []
    i_init, j_init, dists, vecs, dirs = primitive_neighbor_list(quantities='ijdDS', pbc=system.pbc,
                                                          cell=system.get_cell(complete=True),
                                                          positions=system.get_scaled_positions(),
                                                          cutoff=cutoff, numbers=system.numbers,
                                                          use_scaled_positions=True)

    for i, j, dist, vec, dir in zip(i_init, j_init, dists, vecs, dirs):
        # TODO Why we had this less 0.5A and not more than 5A (usually)
        # if np.abs(dist - tmp_Rval) > cutoff or dist < 0.5:
        if dist < 0.5 or j < i:
            continue
        bonds.append(Bond(atom1=system[i], atom2=system[j], direction=dir, distance=dist, vector=vec))

    tmp_bonds = sorted(bonds, key=lambda x: x.delta)

    bonds = []
    while tmp_bonds:
        bond = tmp_bonds.pop(0)
        bonds_one_type = [bond]
        bonds_remain = []
        # Obtain all bonds with the same type by distance:
        for b in tmp_bonds:
            if b == bond:
                bonds_one_type.append(b)
            else:
                bonds_remain.append(b)
        tmp_bonds = bonds_remain
        bonds.append(bonds_one_type)
    return bonds
