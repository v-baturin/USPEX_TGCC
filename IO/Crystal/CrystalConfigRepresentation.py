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


def getTargetConfigRepresentation(config) -> list:
    header = []
    isMolSystem = any('MOL' in symbol for symbol in ['symbols'])
    isVarComp = len(config.blocks) > 1

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

    symbols = config.chemicalSymbols
    row = '    There are %1d types of atoms in the system:' % len(symbols)
    for symbol in symbols:
        row += '%5s' % symbol
    row += '\n'

    for i, symbol in enumerate(symbols):
        row += '    Minimum distances:                 %5s: ' % symbol
        for j in range(len(symbols)):
            row += '%4.2f  ' % config.minDistMatrice[i, j]
        row += '\n'
    row += '\n'

    for i, symbol in enumerate(symbols):
        row += '           Good Bonds:                 %5s: ' % symbol
        for j in range(len(symbols)):
            row += '%4.2f  ' % config.goodBonds[i, j]
        row += '\n'
    row += '\n'

    row += '             Valences:                        '
    for valence in config.valences:
        row += '%4.2f  ' % valence
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

    text = ['Ab initio calculations']
    formatted_rows = createHeader_wrap(text, 'center')
    formatted_rows.append('')
    if config.externalPressure > 0:
        row = '* External Pressure is: %6.4f GPa *' % config.externalPressure
        formatted_rows.append(row)

    header += formatted_rows

    return header
