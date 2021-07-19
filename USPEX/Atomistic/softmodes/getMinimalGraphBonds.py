import numpy as np

from ase.atoms import Atoms
from ase.neighborlist import primitive_neighbor_list
from itertools import chain, combinations_with_replacement
from scipy.sparse.csgraph import connected_components
from typing import Dict, List, Union, Tuple

from ..Bonds import Bond

_BONDS_CUTOFF = 5.0  # Angstroms


def _connectedComponents(N, bonds, pbc):
    """
    Calculate number of connected components.
    :param N: number of atoms
    :param bonds: bond graph
    :param pbc: pbc
    :return: Number of connected components
    """

    supercell_size = 2

    supercell_dims = pbc * (supercell_size - 1) + 1
    supercell_ranges = np.array([[0, 1]] * 3) * supercell_dims.reshape(3, -1)
    all_cells_in_super = np.array(np.meshgrid(*[range(*x) for x in supercell_ranges])).T.reshape(-1, 3)
    total_cells = len(all_cells_in_super)

    graph = np.zeros((total_cells * N, total_cells * N))

    pwrs = np.zeros(3, dtype=int)
    pwrs[pbc] = np.array([2, 1, 0])[np.sort(pbc)]
    for bond in chain(*bonds):
        i, j = bond.indicies
        for klm in all_cells_in_super:
            i_super = i + np.sum(supercell_dims ** pwrs * klm * N)
            j_super = j + np.sum(supercell_dims ** pwrs * ((klm + bond.direction) % supercell_dims) * N)
            graph[i_super, j_super] = 1

    N_components, labels = connected_components(graph)
    return N_components

def getMinimalGraphBonds(SYSTEM, goodBonds=None) -> list:
    '''
    Calculates bond graph minimal for the structure to be 3D connected.

    :param SYSTEM:
    :return:
    '''

    N_atom = len(SYSTEM)
    goodBonds = {frozenset((s1.short_name, s2.short_name)): np.power(s1.good_bonds * s2.good_bonds, 0.5)
                 for s1, s2 in
                 combinations_with_replacement(SYSTEM.getAtomTypes(), 2)} if goodBonds is None else goodBonds
    structure = Atoms(symbols=[s.short_name for s in SYSTEM.getAtomTypes()],
                      positions=SYSTEM.getCartesianCoordinates(),
                      cell=SYSTEM.getCell().getCellVectors(),
                      pbc=SYSTEM.getCell().getPBC())

    # 1) Calculate bonds within upper bound to max_bond.
    # 2) Group bonds by using same_bond criterion.
    bonds = []
    i_init, j_init, dists, vecs, dirs = primitive_neighbor_list(quantities='ijdDS', pbc=structure.pbc,
                                                                cell=structure.get_cell(complete=True),
                                                                positions=structure.get_scaled_positions(),
                                                                cutoff=Bond.MAX_BOND, numbers=structure.numbers,
                                                                use_scaled_positions=True)

    for i, j, dist, vec, dir in zip(i_init, j_init, dists, vecs, dirs):
        # TODO Why we had this less 0.5A and not more than 5A (usually)
        # if np.abs(dist - tmp_Rval) > cutoff or dist < 0.5:
        if dist < 0.5 or j < i:
            continue
        bonds.append(Bond(atom1=structure[i], atom2=structure[j], dir2=dir))

    tmp_bonds = sorted(bonds, key=lambda x: x.delta)

    bond_total = []
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
        bond_total.append(bonds_one_type)

    # 3) Add bonds by group.
    bond_in = []
    bond_left = []

    # delete short bonds
    for bond_group in bond_total:
        a, b = bond_group[0].symbols
        small_bond = -0.37 * np.log(goodBonds[frozenset((a, b))])
        if min([bond.delta for bond in bond_group]) < small_bond:
            bond_in.append(bond_group)  # Add by group
        else:
            bond_left.append(bond_group)
    # del bond_group[0]

    # 5, check 3D connectivity, if not satisfied, add more bonds
    #   but we only include those bonds which could increase connectivity
    # ---Looks like we have to include all bonds before the connectivity changes
    #   otherwise, we won't add them

    N_components = _connectedComponents(N_atom, bond_in, pbc=structure.pbc)
    # List = connectList(chain(*bond_in))

    while N_components > 1:
        # disp('The stuture is not fully connected, adding more bonds');
        bond_tmp = bond_in + [bond_left.pop(0)]
        # List_new = connectList(chain(*bond_tmp))
        N_components_new = _connectedComponents(N_atom, bond_tmp, pbc=structure.pbc)
        # if len(List_new) > len(List) or len(List) == 1: # increase connectivity accept
        if N_components_new < N_components:
            # disp('The connectivity is increased, accept adding more bonds');
            # List = List_new
            N_components = N_components_new
            bond_in = bond_tmp
            # else
            # disp('The connectivity is not increased, reject adding more bonds');

    # 6, Remove double count of bond like [i,i] pair;
    for i, bonds_tmp in enumerate(bond_in):
        indicies = []
        for j, bond in enumerate(bonds_tmp):
            a, b = bond.indicies
            if a == b:
                indicies.append(j)
        for j in sorted(indicies[::2], reverse=True):
            del bond_in[i][j]

    return bond_in
