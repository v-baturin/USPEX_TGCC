import numpy as np


from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure
from ..Bonds import Bonds
from .MaxBonds import MaxBonds
from .AtomTypeCounter import atomTypeCounter


def BondHardness_new(SYSTEM : AtomicStructure, goodBonds) -> Bonds:
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

    small_bond = -0.37 * np.log(np.array(goodBonds))  # maximum deviation to consider a bond

    N_atom = len(SYSTEM)
    atomTypes, atom_type_seq = atomTypeCounter(SYSTEM.chemicalSymbols)

    # 1) Calculate bonds within upper bound to max_bond.
    # 2) Group bonds by using same_bond criterion.
    bond_total = MaxBonds(SYSTEM)
    bond_group = np.unique([bond.type for bond in bond_total]).tolist()
    bond_group_ref = np.copy(bond_group)

    # 3) Add bonds by group.
    bond_in = Bonds()

    # delete short bonds
    id = 1
    ToDelete = []
    while id < len(bond_group_ref):
        bonds = bond_total.getType(id)
        a,b = bonds[0].atoms()
        a,b = atom_type_seq[a], atom_type_seq[b]
        if min([b.delta for b in bonds]) < small_bond[a,b]:
            bond_in.extend(bonds)    # Add by group
            ToDelete.append(id)
        id = id + 1
    # del bond_group[0]

    # 5, check 3D connectivity, if not satisfied, add more bonds
    #   but we only include those bonds which could increase connectivity
    # ---Looks like we have to include all bonds before the connectivity changes
    #   otherwise, we won't add them

    List = bond_in.connectList()

    while len(List) < N_atom:
        # disp('The stuture is not fully connected, adding more bonds');
        id = bond_group[0]
        if id not in ToDelete:
            bonds = bond_total.getType(id)
            bond_tmp = bond_in + bonds
            List_new = bond_tmp.connectList()
            del bond_group[0]
            if len(List_new) > len(List) or len(List) == 1: # increase connectivity accept
                # disp('The connectivity is increased, accept adding more bonds');
                List = List_new
                bond_in = bond_tmp
                # else
                # disp('The connectivity is not increased, reject adding more bonds');
        else:
            if len(bond_group):
                del bond_group[0]

    # 6, Remove double count of bond like [i,i] pair;
    indicies = []
    for i, bond in enumerate(bond_in):
        a,b = bond.atoms()
        if a == b:
            indicies.append(i)
    for i in sorted(indicies[::2], reverse=True):
        del bond_in[i]

    return bond_in

