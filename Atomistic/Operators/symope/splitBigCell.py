from __future__ import division

import numpy as np

from ...CellUtility import Cell
from .symope_crystal import symope_crystal


def splitBigCell(CenterminDistMatrice, constLattice, fixRndSeed, startLat, splitInto, numIons, nsym, sym_coef, debug=False):
    """
    The function splits the big cell into smaller pieces and fills them with atoms.
    Keep in mind that we are dealing with a periodical problem (periodicity : 1).
        Example (coordinate) : 2.41 = 1.41 = 0.41 = -0.59 = - 1.59...
    :param startLat: 1x6 input lattice.
    :param splitInto: number of pieces.
    :param numIons: number of atoms.
    :param nsym: space group number.
    :param debug: enable fixed values for instead of random ones.
    :return lat: resulted lattice.
    :return coordinates: resulted coordinates.
    """

    #numIons = numIons_standardize(numIons)[0]

    splitByCellNumber = True

    #TODO fix const lattice
    #if ORG_STRUC['constLattice']:
    #    startLat = latConverter([ORG_STRUC['lattice']])

    b = np.copy(numIons)

    # find nStrEnt, number of structural entities per big cell
    # GrCD = greatest Atomistic divisor
    # generalized Euclidean algorithm for finding GrCD

    L = len(b)
    nFull = np.sum(numIons)  # number of atoms in the cell

    IX = startLat[0:3].argsort()  # lattice(1)<=lattice(2)<=lattice(3); lattice = startLat(IX)
    lattice = sorted(startLat[0:3])
    opt = 10.0 * nFull
    opt1 = 10.0 * nFull
    x = 0.0
    y = 0.0
    z = 0.0

    # Search for optimal splitting:
    if splitByCellNumber:
        i1 = int(splitInto ** (1 / 3.0))
        i2 = int(splitInto ** 0.5)
        coor = np.zeros(3)  # TODO: rename by real meaning
        for i in range(i1 + 1):
            for j in range(i, i2 + 1):
                k = np.floor(splitInto / float((i + 1) * (j + 1)))
                if k >= j + 1:
                    candidate = abs(splitInto - (i + 1) * (j + 1) * k)
                    split_lattice_match = (i + 1) / float(lattice[0]) + lattice[0] / float(i + 1) + \
                                          (j + 1) / float(lattice[1]) + lattice[1] / float(j + 1) + \
                                          (k + 0) / float(lattice[2]) + lattice[2] / float(k + 0)
                    if candidate < opt or (candidate == opt and split_lattice_match < opt1):
                        coor[0] = i + 1
                        coor[1] = j + 1
                        coor[2] = k
                        opt = candidate
                        opt1 = split_lattice_match

        IX1 = IX.argsort()

        x = coor[IX1[0]]
        y = coor[IX1[1]]
        z = coor[IX1[2]]

    else:  # splitInto - approximate number of atoms per subCell
        # TODO: Not yet implemented and probably not needed at all:
        pass

    doAtoms = np.zeros(L, dtype=int)  # number of atoms of a given type that fill the cell completely
    addAtoms = np.zeros(L, dtype=int)  # number of atoms of a given type that are added to some subcells only
    genAtoms = np.zeros(L, dtype=int)  # number of atoms of a given type to generate

    for atomType in range(L):
        doAtoms[atomType] = np.floor(numIons[atomType] / float(splitInto))
        addAtoms[atomType] = numIons[atomType] - splitInto * doAtoms[atomType]
        if addAtoms[atomType] > 0:
            genAtoms[atomType] = doAtoms[atomType] + 1  # generate extra atom, which will be added to some subcells only
        else:
            genAtoms[atomType] = doAtoms[atomType]

    coord_splitter = np.random.rand(np.sum(genAtoms[:]), 3)

    if debug:
        coord_splitter = np.asarray([
            [0.85730629, 0.34968992, 0.38697974],
            [0.1842412, 0.3517269, 0.42075164],
            [0.3196636, 0.44109966, 0.48899095],
            [0.28728442, 0.88434003, 0.90420956],
            [0.40270179, 0.5210671, 0.62774483],
            [0.28960974, 0.89941119, 0.91817915],
            [0.28472294, 0.39995279, 0.97584671],
            [0.50019545, 0.56832531, 0.30302932],
            [0.26098409, 0.76545493, 0.01824839],
            [0.74328425, 0.19505128, 0.09817633],
            [0.37358812, 0.1261526, 0.77364214],
            [0.09553393, 0.12163574, 0.9313818],
            [0.89700552, 0.13912972, 0.94532954],
            [0.11510054, 0.59842141, 0.59112328]
        ])

    if True:#ORG_STRUC['nsymN'][0] == 0:  # H. Stokes code to create a crystal with given symmetry
        lattice1 = np.copy(startLat)
        lattice1[0] /= float(x)
        lattice1[1] /= float(y)
        lattice1[2] /= float(z)
        lat1 = np.copy(lattice1)

        for l1 in range(5):
            #os.chdir(ORG_STRUC['homePath'] + '/CalcFoldTemp')
            coord_splitter, lat1 = symope_crystal(CenterminDistMatrice, constLattice, fixRndSeed, nsym, genAtoms, lat1, sym_coef)
            #os.chdir(ORG_STRUC['homePath'])
            # Resulted lattice from symope_crystal() is 3x3, so need to convert it to 1x6 and make 1-D:
            lat1 = Cell(cellVectors=lat1, pbc=(1,1,1)).getCellParameters()
            break

        if constLattice == 0:
            startLat[0] = lat1[0] * x  # we change the initial lattice, according to subcell symmetry group
            startLat[1] = lat1[1] * y
            startLat[2] = lat1[2] * z
            startLat[3:6] = lat1[3:6]

    coordinates = np.random.rand(nFull, 3)

    if debug:
        coordinates = np.asarray([
            [0.96822066, 0.2517101, 0.57822385],
            [0.70432748, 0.04922431, 0.76214002],
            [0.62703267, 0.2170886, 0.02314245],
            [0.04643411, 0.14843616, 0.54412294],
            [0.409345, 0.50896938, 0.31605907],
            [0.74867689, 0.01844094, 0.28138511],
            [0.39480167, 0.87694176, 0.38037434],
            [0.99158885, 0.53136403, 0.01556775],
            [0.18539039, 0.11225039, 0.84240503],
            [0.52714257, 0.72231447, 0.4316436],
            [0.74586343, 0.52479114, 0.60941549],
            [0.86324156, 0.60377726, 0.17225499],
            [0.71828695, 0.00217155, 0.09174453],
            [0.28274825, 0.45149692, 0.92053737],
            [0.0316237, 0.10084848, 0.87517295],
            [0.58678181, 0.25344978, 0.24082898],
            [0.84864768, 0.09918821, 0.16596798],
            [0.27601907, 0.0699852, 0.53321613],
            [0.86658662, 0.52981977, 0.3159709],
            [0.08424556, 0.24133808, 0.21273874],
            [0.9124918, 0.61147247, 0.67693292],
            [0.48289803, 0.02086843, 0.75952414],
            [0.0226039, 0.46693708, 0.99806062],
            [0.65324137, 0.63828763, 0.98402287],
            [0.62260686, 0.53020633, 0.30989301],
            [0.29863948, 0.01750441, 0.50428419],
            [0.00706726, 0.46146841, 0.82312196],
            [0.58832428, 0.18840487, 0.91445965]
        ])

    counter = 0
    counter1 = 0

    # Main loop, atoms that are translated to all subcells:
    for atomType in range(L):
        for a1 in range(doAtoms[atomType]):
            for i in range(int(x)):
                for j in range(int(y)):
                    for k in range(int(z)):
                        coordinates[counter, 0] = (i + coord_splitter[counter1, 0]) / float(x)
                        coordinates[counter, 1] = (j + coord_splitter[counter1, 1]) / float(y)
                        coordinates[counter, 2] = (k + coord_splitter[counter1, 2]) / float(z)
                        counter += 1
            counter1 += 1

        # Rest of the atoms, translated to some subcells (choosen randomly):
        p = np.random.permutation(splitInto)

        if debug:
            p = [1, 0]

        if addAtoms[atomType] > 0:
            counter1 += 1

        for a1 in range(addAtoms[atomType]):
            div1 = np.floor(p[a1] / float(x * y))
            mod1 = p[a1] - div1 * x * y
            if mod1 == 0:
                k = div1
                p[a1] = x * y
            else:
                k = div1 + 1
                p[a1] = mod1

            div1 = np.floor(p[a1] / float(x))
            mod1 = p[a1] - div1 * x
            if mod1 == 0:
                j = div1
                i = x
            else:
                j = div1 + 1
                i = mod1

            coordinates[counter, 0] = (i - 1 + coord_splitter[counter1, 0]) / float(x)
            coordinates[counter, 1] = (j - 1 + coord_splitter[counter1, 1]) / float(y)
            coordinates[counter, 2] = (k - 1 + coord_splitter[counter1, 2]) / float(z)
            counter += 1

    lat = Cell.initFromCellParameters(*startLat, pbc = (1,1,1)).getCellVectors()
    print('split into: x = %d, y = %d, z = %d' % (x, y, z))

    return lat, coordinates

'''
if __name__ == "__main__":
    from lib.mat2dict import loadmat

    test_dir = 'test_splitBigCell'

    variables = loadmat(test_dir + '/var.mat')

    ORG_STRUC = variables['ORG_STRUC']
    POP_STRUC = variables['POP_STRUC']

    ORG_STRUC['homePath'] = os.path.abspath(os.path.join(os.getcwd(), test_dir))

    lat, errorS, coordinates = splitBigCell(ORG_STRUC, POP_STRUC, variables['startLat'], variables['splitInto'],
                                            variables['numIons'], variables['nsym'])

    print('')
'''