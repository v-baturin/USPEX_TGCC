"""
USPEX.Common.Atomistic.Fingerprints.structure_order
===================================================

Calculate structure order

.. codeauthor:: Maxim Rakitin
"""

import numpy as np


def structure_order(f, volume, structure, weight, deltaFing=0.08):
    """
    Calculate structure order.

    :type f: numpy array
    :param f: fingerprint values.
    :type volume: float
    :param volume: volume of the cell.
    :type structure: :class:`~USPEX.Common.Atomistic.AtomicStructure.AtomicStructure` or descendant
    :param structure: system for which we want to calculate structure order.
    :type weight: numpy array
    :param weight: weight used in fingerprint calculations.
    :type deltaFing: float
    :param deltaFing: parameter delta.
    :rtype: float
    :return: structure order.
    """

    '''
    s_order = 0.0

    r, c = f.shape
    for j in range(r):
        order_ab = 0.0
        for k in range(c):
            order_ab += f[j, k] ** 2.0

        order_ab *= deltaFing / (volume / float(sum(numIons))) ** (1.0 / 3.0)
        s_order += order_ab * weight[j]
    '''
    numIons = np.unique(structure.get_chemical_symbols(), return_counts=True)[1]
    s_order = np.sum(np.dot(weight, f**2))
    s_order *= deltaFing / (volume / float(sum(numIons))) ** (1.0 / 3.0)
    s_order **= 0.5

    return s_order
