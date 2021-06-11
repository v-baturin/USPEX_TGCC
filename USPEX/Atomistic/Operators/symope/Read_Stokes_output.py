import numpy as np


def Read_Stokes_output(content):#filename='rc.out'
    """
    This code can be used to read 2D/3D crystals.
    :param filename: Stokes' output file to parse.
    :return candidate: coordinates of a candidate structure.
    :return lattice: lattice of a candidate structure.
    :return errorS: error flag.
    """

    lattice = np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    candidate = np.zeros((0, 3))
    errorS = 0

    nsym = None
    numIons = None

#    with open(filename, 'r') as f:
    content = content.split('\n')

    for i in range(len(content)):
        if content[i].find('Space-group symmetry') >= 0:
            nsym = int(content[i].split(':')[1].strip())

        elif content[i].find('atoms of each type: ') >= 0:
            numIons = content[i].split(':')[1].strip().split()
            numIons = np.asarray([[int(x) for x in numIons]])

        elif content[i].find('cell parameters') >= 0:
            lat = content[i].split(':')[1].strip().split()
            lattice = np.asarray([float(x) for x in lat])
        elif content[i].find('atomic parameters') >= 0:
            for j in range(i + 2, len(content)):
                if content[j].strip() != '':
                    tmp = content[j].split('(')[0].strip().split()
                    tmp = np.asarray([float(x) for x in tmp])
                    candidate = np.vstack((
                        candidate,
                        tmp[2:5],
                    ))
                else:
                    continue
        elif content[i].find('error') >= 0:
            errorS = 1
            break

    if not errorS and numIons is not None:
        if np.sum(numIons) != candidate.shape[0]:
            print('Stokes output error: atomic coordinates are insufficient')

    return candidate, lattice, errorS


if __name__ == "__main__":
    test_dir = 'test_symope_crystal'
    # filename = test_dir + '/' + 'rc.out'
    filename = test_dir + '/' + 'rc2.out'

    candidate, lattice, errorS = Read_Stokes_output(filename)

    from lib.symope.fix_latticeStokes_after import fix_latticeStokes_after

    lat, coord = fix_latticeStokes_after(34, lattice, candidate)

    print('')
