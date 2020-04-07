import numpy as np
from itertools import chain


from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure
from ..Bonds import connectList
from .MaxBonds import MaxBonds_new


_BONDS_CUTOFF = 5.0     # Angstroms

# def Bonds(system, cutoff=_BONDS_CUTOFF) -> Bonds:



def BondHardness_new(SYSTEM : AtomicStructure, goodBonds) -> list:
    '''
    The function calculates bonds which make contribution to hardness.
    Used only for softmodemutation case and does not used for any other cases.

    NOTE (for future generations):
    this function should be used for softmodes calculation and for hardness calculation as well.
    But it is not right now. So, this method is not used for hardness calculation.
    Method that is above right now works proper for hardness calculation.

    :param SYSTEM:
    :param goodBonds:
    :return:
    '''

    N_atom = len(SYSTEM)


    # 1) Calculate bonds within upper bound to max_bond.
    # 2) Group bonds by using same_bond criterion.
    bond_total = MaxBonds_new(SYSTEM)

    # 3) Add bonds by group.
    bond_in = []
    bond_left = []

    # delete short bonds
    for bonds in bond_total:
        a,b = bonds[0].symbols
        small_bond = -0.37 * np.log(goodBonds[f'{a}-{b}'])
        if min([bond.delta for bond in bonds]) < small_bond:
            bond_in.append(bonds)    # Add by group
        else:
            bond_left.append(bonds)
    # del bond_group[0]

    # 5, check 3D connectivity, if not satisfied, add more bonds
    #   but we only include those bonds which could increase connectivity
    # ---Looks like we have to include all bonds before the connectivity changes
    #   otherwise, we won't add them

    List = connectList(chain(*bond_in))

    while len(List) < N_atom:
        # disp('The stuture is not fully connected, adding more bonds');
        bonds = bond_left[0]
        bond_tmp = bond_in + [bonds]
        List_new = connectList(chain(*bond_tmp))
        del bond_left[0]
        if len(List_new) > len(List) or len(List) == 1: # increase connectivity accept
            # disp('The connectivity is increased, accept adding more bonds');
            List = List_new
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

