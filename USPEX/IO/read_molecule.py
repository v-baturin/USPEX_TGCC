"""
USPEX.Common.Atomistic.mol.read_molecule
========================================

Function for reading molecule from a MOL file

.. codeauthor:: Pavel Bushlanov <paulbush@mail.ru>
"""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)


def read_molecule(filename: str):
    """
    Function for reading molecules from the file MOL_*.

    :type filename: str
    :param filename: a path to the file with a molecule.
    :rtype: dict
    :return: molecule as a dictionary.
    """
    assert os.path.exists(filename), f'File {filename} with molecule does not exist. Please, check the path.'
    
    dct = {}
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
            logger.info('This calculation uses charge model, currently only supported in GULP')
            N_col += 1
            ind_mag = 9
        if 'mag' in name:  # magnetic moments are present on the last line
            logger.info(f'This calculation uses predefined magnetic moments on {name}')
            N_col += 1

        num = int(handle.readline().split()[-1])
        temper = np.zeros((num, N_col))

        # An example:  C_R ===> Symbol: C  Label: R
        label = []
        symbol = []
        for i in range(num):
            a = handle.readline()[:-1]              
            symbol_label, *tmp = a.split()
            symbol_label = symbol_label.split('_', maxsplit=1)
            if len(symbol_label) == 2:  # For GULP and DMACRYS
                symbol.append(symbol_label[0])
                label.append(symbol_label[1])
            else:
                symbol.append(symbol_label[0])
                label.append('')

            # temper[i, 0] = Element(tmp[0].split('_')[0]).z  # to find the index for the element
            temper[i, 1:] = tmp

        dct = {}
        dct['symbols'] = symbol
        dct['labels'] = label
        dct['positions'] = temper[:, 1:4].tolist()
        dct['configZMatrix'] = temper[:, 4:7].astype(int).tolist()
        dct['flexDihedrals'] = np.flatnonzero(temper[3:, 7]).astype(int).tolist()
        if 'charge' in name:  # GULP+CHARGE
            dct['charges'] = temper[:, 8].tolist()
        if 'mag' in name:
            dct['magmoms'] = temper[:, ind_mag].tolist()

        # molecule = AtomicStructure.fromDICT(dct)
        #
        # molecule.translate(-molecule.coordinates.mean(axis=0))
        # a, b = molecule.principleAxis
        # molecule.set_positions(np.dot(molecule.coordinates, -b))
        # dct = molecule.toDICT()

    return dct
