"""
USPEX.Common.Atomistic.Fingerprints.cosine_distance
===================================================

Calculation of cosine distances

.. codeauthor:: Maxim Rakitin
"""

import numpy as np
from typing import Dict, Tuple, Union

def cosine_distance(fingerprint1: Dict[Union[Tuple[str,str], str], np.ndarray],
                    fingerprint2: Dict[Union[Tuple[str,str], str], np.ndarray],
                    weights1: Dict[Union[Tuple[str,str], str], float],
                    weights2: Dict[Union[Tuple[str,str], str], float]):
    """
    Calculation of cosine distances using eq.(6b) from JCP-2009.

    :type fingerprint1: Dict[Union[Tuple[str,str], str], np.ndarray]
    :param fingerprint1: atomic fingerprint 1.
    :type fingerprint2: Dict[Union[Tuple[str,str], str], np.ndarray]
    :param fingerprint2: atomic fingerprint 2.
    :type weights1: Dict[Union[Tuple[str,str], str], float]
    :param weights1: weight for a particular atom type in the cell.
    :type weights2: Dict[Union[Tuple[str,str], str], float]
    :param weights2: weight for a particular atom type in the cell.
    :rtype: float
    :return: resulted cosine distance.
    """

    coef1 = 0
    coef2 = 0
    coef3 = 0

    for key in set(fingerprint1.keys()) | set(fingerprint2.keys()):
        fing1 = fingerprint1[key] if key in fingerprint1 else np.zeros((1), dtype=float)
        fing2 = fingerprint2[key] if key in fingerprint2 else np.zeros((1), dtype=float)
        weight1 = weights1[key] if key in weights1 else 0
        weight2 = weights2[key] if key in weights2 else 0
        coef1 += np.sqrt(weight1 * weight2) * np.sum(fing1 * fing2)
        coef2 += weight1 * np.sum(fing1 * fing1)
        coef3 += weight2 * np.sum(fing2 * fing2)

    dist = (1 - coef1 / (coef2 * coef3) ** 0.5) / 2

    return dist
