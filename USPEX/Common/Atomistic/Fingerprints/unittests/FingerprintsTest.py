'''
@file        FingerprintsTest.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        02 October 2017
@brief       Class for testing Fingerprints.
'''


import unittest
import os
from ase.io import read

from lib.Fingerprints.Fingerprints import Fingerprints
from lib.Fingerprints.cosine_distance import cosine_distance


HOMEPATH = os.path.dirname(os.path.abspath(__file__))


class FingerprintsTest(unittest.TestCase):

    def test_atomic(self):
        system1 = read(HOMEPATH + '/system1_POSCAR')
        system2 = read(HOMEPATH + '/system2_POSCAR')
        f1 = Fingerprints(system1)
        f2 = Fingerprints(system2)
        self.assertTrue(cosine_distance(f1.fingerprint,f2.fingerprint,1) < 1.0e-6)


#import numpy as np
#np.set_printoptions(threshold = 10000, precision = 4, suppress=True)
#print(f1.fingerprint[3])
#print(f2.fingerprint[0])
#dm1 = make_matrices(system1)
#dm2 = make_matrices(system2)
#btype1 = dm1[:,1] * 2 + dm1[:,2]
#btype2 = dm2[:,1] * 1 + dm2[:,2]
#print(len(dm1[np.where(btype1==3),3][0]),len(dm2[:,3]))

