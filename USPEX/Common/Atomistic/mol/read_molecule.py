import os

import numpy as np
import logging

from ..AtomicStructure import AtomicStructure
from ..Element import Element


def read_molecule(filename):
    '''
    Method for reading molecules from the file MOL_*.
    :param filename: really, it's a path to the file with a molecule
    :return: molecule as AtomicStructure object.
    '''

    dct = {}
    if os.path.exists(filename):
        with open(filename) as handle:
            name = handle.readline()[:-1]
            start = name.find('[')
            ending = name.find(']')
            if start != ending:
                name = name[start + 1:ending]

            # how many columns
            N_col = 8
            ind_mag = 8
            if 'charge' in name:  # GULP has one more line to specify charge
                logging.info('This calculation uses charge model, currently only supported in GULP')
                N_col += 1
                ind_mag = 9
            if 'mag' in name:  # Magnetic moments are present on the last line
                logging.info('This calculation uses predefined magnetic moments on {}'.format(name))
                N_col += 1

            num = int(handle.readline().split()[-1])
            temper = np.zeros((num, N_col))

            #An example:  C_R ===> Symbol: C  Label: R
            label = []
            symbol = []
            for i in range(num):
                a = handle.readline()[:-1]              
                tmp = a.split()
                if '_' in a:  # For GULP and DMACRYS
                    mark = a.find('_')  # how many chars for the element
                    label.append(tmp[0][mark + 1])
                    symbol.append(tmp[0][:mark])
                else:
                    label.append('')
                    symbol.append(tmp[0])


                temper[i, 0] = Element(tmp[0].split('_')[0]).z  # to find the index for the element
                temper[i, 1:] = tmp[1:]

            dct = {}
            dct['symbols'] = symbol
            dct['positions'] = temper[:, 1:4].tolist()
            dct['molecules'] = [list(range(len(symbol)))]
            dct['molFormats'] = [temper[:,4:7].astype(int).tolist()]
            dct['molFlexDihedrals'] = [np.flatnonzero(temper[3:, 7]).astype(int).tolist()]
            dct['molSymbols'] = [os.path.split(filename)[1]]
            dct['pbc'] = [False, False, False]
            if 'charge' in name:  # GULP+CHARGE
                dct['charges'] = temper[:, 8].tolist()
            if 'mag' in name:
                dct['magmoms'] = temper[:, ind_mag].tolist()

            molecule = AtomicStructure.fromDICT(dct)

            molecule.translate(-molecule.coordinates.mean(axis=0))
            a, b = molecule.principleAxis()
            molecule.set_positions(np.dot(molecule.coordinates, -b))
            dct = molecule.toDICT()


    return dct
