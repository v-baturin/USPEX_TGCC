import numpy as np

from ase.atoms import Atoms
from ase.neighborlist import primitive_neighbor_list
from itertools import chain, combinations_with_replacement
from scipy.sparse.csgraph import connected_components
from typing import Dict, List, Union, Tuple

from ..Bonds import Bond


_BONDS_CUTOFF = 5.0     # Angstroms

def _connectedComponents(N, bonds):
    """
    Calculate number of connected components.
    :param N: number of atoms.
    :param bonds: bond graph.
    :return:
    """
    if N < 100:
        supper_size = 4
        center_cell = [1,2]
    else:
        supper_size = 3
        center_cell = [1]
    graph = np.zeros((supper_size**3*N,supper_size**3*N))
    indices = []
    for bond in chain(*bonds):
        i,j = bond.indicies
        k0,l0,m0 = bond.direction
        for k in range(supper_size):
            for l in range(supper_size):
                for m in range(supper_size):
                    i_super = i + (supper_size**2*k+supper_size*l+m)*N
                    j_super = j + (supper_size**2*(k+k0)+supper_size*(l+l0)+(m+m0))*N
                    if 0 <= j_super < supper_size**3*N:
                        graph[i_super, j_super] = 1
                    if (k in center_cell) and (l in center_cell) and (m in center_cell):
                        indices.extend(range(supper_size**2*k+supper_size*l+m,supper_size**2*k+supper_size*l+m+N))
    N_components, labels = connected_components(graph)
    return len(np.unique(labels[np.asarray(indices)]))


def getMinimalGraphBonds(SYSTEM, goodBonds = None) -> list:
    '''
    Calculates bond graph minimal for the structure to be 3D connected.

    :param SYSTEM:
    :return:
    '''

    N_atom = len(SYSTEM)
    goodBonds = {frozenset((s1.short_name, s2.short_name)):np.power(s1.good_bonds*s2.good_bonds, 0.5)
                 for s1,s2 in combinations_with_replacement(SYSTEM.getAtomTypes(), 2)} if goodBonds is None else goodBonds
    structure = Atoms(symbols=[s.short_name for s in SYSTEM.getAtomTypes()],
                      positions = SYSTEM.getCartesianCoordinates(),
                      cell = SYSTEM.getCell().getCellVectors(),
                      pbc = SYSTEM.getCell().getPBC())


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
        a,b = bond_group[0].symbols
        small_bond = -0.37 * np.log(goodBonds[frozenset((a,b))])
        if min([bond.delta for bond in bond_group]) < small_bond:
            bond_in.append(bond_group)    # Add by group
        else:
            bond_left.append(bond_group)
    # del bond_group[0]

    # 5, check 3D connectivity, if not satisfied, add more bonds
    #   but we only include those bonds which could increase connectivity
    # ---Looks like we have to include all bonds before the connectivity changes
    #   otherwise, we won't add them

    N_components = _connectedComponents(N_atom, bond_in)
    # List = connectList(chain(*bond_in))

    while N_components > 1:
        # disp('The stuture is not fully connected, adding more bonds');
        bond_tmp = bond_in + [bond_left.pop(0)]
        # List_new = connectList(chain(*bond_tmp))
        N_components_new = _connectedComponents(N_atom, bond_tmp)
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
            a,b = bond.indicies
            if a == b:
                indicies.append(j)
        for j in sorted(indicies[::2], reverse=True):
            del bond_in[i][j]

    return bond_in

