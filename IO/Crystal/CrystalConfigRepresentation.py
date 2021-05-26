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
    isMolSystem = target.utilities.simpleMoleculeUtility.isTrueMolecular
    isVarComp = not target.utilities.compositionSpace.isFixedComposition

    if isMolSystem:
        header += createHeader_wrap(['Molecular Crystals:'], 'center')
        header += createHeader_wrap(MOL_CRYSTALS_PAPERS, 'left')

    if isVarComp:
        header += createHeader_wrap(['Variable Composition:'], 'center')
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

    text = ['Block for atomic description']
    formatted_rows = createHeader_wrap(text, 'center')
    formatted_rows.append('')

    symbols = set()
    for symbol in target.utilities.compositionSpace.symbols:
        symbols.update(target.utilities.simpleMoleculeUtility.molecules[symbol].getAtomTypes())
    symbols = sorted(symbols)
    minDistMatrix = target.utilities.ionDistances.getDistances(symbols, target.utilities.conditions)

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

    # TODO: write information about molecules

    # ---------------------------------------------------------------------------

    row = '    The investigated system is: '
    # TODO write information about composition space

    row += '\n'
    header += [row]

    # ---------------------------------------------------------------------------

    row = ''
    # TODO write information about lattice parameters

    header += [row]

    # ---------------------------------------------------------------------------

    formatted_rows = createHeader_wrap(['Conditions'], 'center')
    formatted_rows.append('')
    if target.utilities.conditions.externalPressure > 0:
        formatted_rows.append('* External Pressure is: %6.4f GPa *' % target.utilities.conditions.externalPressure)

    header += formatted_rows

    return header
