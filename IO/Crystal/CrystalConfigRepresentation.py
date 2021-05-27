import numpy as np
from collections import Counter
from .formatters import createHeader_wrap


MOL_CRYSTALS_PAPERS = '''\
'Zhu Q., Oganov A.R., Glass C.W., Stokes H. (2012)',
'Constrained evolutionary algorithm for structure prediction of',
'molecular crystals: methodology and applications.',
'Acta Cryst. B, 68, 215-226', \
'''

VARCOMP_PAPERS = '''\
Lyakhov A.O., Oganov A.R., Valle M. (2010)
Crystal structure prediction using evolutionary approach.
In: Modern methods of crystal structure prediction (ed: A.R. Oganov)
Berlin: Wiley-VCH

Oganov A.R., Ma Y., Lyakhov A.O., Valle M., Gatti C. (2010)
Evolutionary crystal structure prediction as a method
for the discovery of minerals and materials.
Rev. Mineral. Geochem. 71, 271-298
'''


def getTargetConfigRepresentation(target) -> list:
    header = []
    ut = target.utilities
    isMolSystem = ut.simpleMoleculeUtility.isTrueMolecular
    isVarComp = not ut.compositionSpace.isFixedComposition

    if isMolSystem:
        header += createHeader_wrap(['Molecular Crystals suggested papers:'], 'center')
        header += createHeader_wrap(MOL_CRYSTALS_PAPERS, 'left')

    if isVarComp:
        header += createHeader_wrap(['Variable Composition suggested papers:'], 'center')
        header += createHeader_wrap(VARCOMP_PAPERS.split('\n'), 'left')

    # ---------------------------------------------------------------------------

    formatted_rows = createHeader_wrap(['Block for system description'], 'center')
    formatted_rows.append('')

    row = '    System type          :  Crystal\n'
    row += f'    Molecular            :  {"Yes" if isMolSystem else "No"}\n'
    row += f'    Variable composition :  { "Yes" if isVarComp else "No"}\n'

    formatted_rows.append(row)
    header += formatted_rows

    # ---------------------------------------------------------------------------

    compositionSpace = ut.compositionSpace
    rows = ['    The investigated system is (block -- range): ']
    symbols = compositionSpace.symbols
    for block, rng in zip(compositionSpace.blocks, compositionSpace.range):
        rows.append(f'        {"".join(f"<{symbols[i]}>{block[i]}" for i in np.flatnonzero(block))}  --  {rng}')
    rows.append('')
    header += rows


    # ---------------------------------------------------------------------------

    # TODO: write information about molecules

    # ---------------------------------------------------------------------------

    try:
        cell = ut.cellUtility.getCell()
    except:
        cell = None

    if cell is not None:
        lattice = cell.getCellVectors()
        rows = ['    This is a fixed lattice calculation ',
                f'        {lattice[0, 0]:.4}   {lattice[0, 1]:.4}    {lattice[0, 2]:.4}',
                f'        {lattice[1, 0]:.4}   {lattice[1, 1]:.4}    {lattice[1, 2]:.4}',
                f'        {lattice[2, 0]:.4}   {lattice[2, 1]:.4}    {lattice[2, 2]:.4}']
    else:
        rows = ['    Volume (estimated) for blocks :']
        for block in compositionSpace.blocks:
            comp = Counter()
            for s, b in zip(symbols, block):
                comp += ut.simpleMoleculeUtility.getElementalComposition({s:b})
            volume = ut.cellUtility.getCellVolume(comp, ut.conditions)
            rows.append(f'        {"".join(f"<{symbols[i]}>{block[i]}" for i in np.flatnonzero(block))}  --  {volume:.4}')

    rows.append('')
    header += rows

    # ---------------------------------------------------------------------------

    text = ['Block for atomic description']
    formatted_rows = createHeader_wrap(text, 'center')
    formatted_rows.append('')

    symbols = set()
    for symbol in ut.compositionSpace.symbols:
        symbols.update(ut.simpleMoleculeUtility.molecules[symbol].getAtomTypes())
    symbols = sorted(symbols)
    minDistMatrix = ut.ionDistances.getDistances(symbols, ut.conditions)

    row = '    There are %1d types of atoms in the system:' % len(symbols)
    for symbol in symbols:
        row += '%5s' % symbol
    row += '\n'

    for i, symbol in enumerate(symbols):
        row += '    Minimum distances:                 %5s: ' % symbol
        for j in range(len(symbols)):
            row += '%4.2f  ' % minDistMatrix[i, j]
        row += '\n'
    row += '\n'

    for symbol1 in symbols:
        row += '           Good Bonds:                 %5s: ' % symbol1
        for symbol2 in symbols:
            row += '%4.2f  ' % (symbol1.good_bonds*symbol2.good_bonds) ** 0.5
        row += '\n'
    row += '\n'

    row += '             Valences:                        '
    for symbol in symbols:
        row += '%4.2f  ' % symbol.valence
    row += '\n'

    formatted_rows.append(row)
    header += formatted_rows

    # ---------------------------------------------------------------------------

    formatted_rows = createHeader_wrap(['Conditions'], 'center')
    formatted_rows.append('')
    if ut.conditions.externalPressure > 0:
        formatted_rows.append('* External Pressure is: %6.4f GPa *' % ut.conditions.externalPressure)
    formatted_rows.append('')

    header += formatted_rows

    return header
