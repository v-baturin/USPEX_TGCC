def atomTypeCounter(chemicalSymbols):
    atomTypes = []
    atom_type_seq = []
    count = -1
    for symbol in chemicalSymbols:
        if symbol not in atomTypes:
            atomTypes.append(symbol)
            count += 1
        atom_type_seq.append(count)
    return atomTypes, atom_type_seq
