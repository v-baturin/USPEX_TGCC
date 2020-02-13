import os
import time
import warnings

import numpy as np

from USPEX.Common.Config import Config
from USPEX.Common.Atomistic.Element import Element
from USPEX.Common.Atomistic.AtomicStructure import AtomicStructure
from .BondHardness_new import BondHardness_new
from .calcHardness import calcHardness
from .calcSoftModes import calcSoftModes
from .AtomTypeCounter import atomTypeCounter


warnings.filterwarnings('ignore')


class VibrationalMode(object):
    frequency = None
    eigenvector = None

    def __init__(self, FREQUENCY, EIGENVECTOR, SUPERCELL=np.ones(3)):
        self.frequency = float(FREQUENCY)
        self.eigenvector = np.array(EIGENVECTOR)
        self.supercell = np.array(SUPERCELL)


class Softmodes(list):
    '''
    The class is to process softmodes, hardness, etc. for a given structure instance.
    :param system:
    :param kvector: k-vector specified by user.
    '''

    modes = []

    duration_h = 0.0
    duration_sm = 0.0

    def __init__(self, config : Config, system : AtomicStructure, kvector=np.zeros(3)):
        '''

        :param config:
        :param system:
        :param kvector:
        '''

        super(Softmodes, self).__init__()
        # Inputs:
        self._config = config
        self.system = system

        # Calculate "smart" default values:

        atomTypes, atom_type_seq = atomTypeCounter(system.chemicalSymbols)

        if hasattr(config, 'valences'):
            self.val = config.valences
        else:
            self.val = [Element(x).valence for x in atomTypes]

        if hasattr(config, 'valenceElectrons'):
            self.N_val = config.valenceElectrons
        else:
            self.N_val = [Element(x).valence_electrons for x in atomTypes]
        self.R_val = np.array([Element(atomType).covalent_radius for atomType in atomTypes])

        system.bonds = BondHardness_new(system, config.goodBonds)
        self._calc_soft_modes(system, kvector)

    def hardness(self):
        '''
        :return: hardness of the material.
        '''
        start_time = time.time()
        hardness = calcHardness(self._config, self.system)
        self.duration_h = time.time() - start_time
        return hardness

    def _calc_soft_modes(self, system, kvector=np.zeros(3)):
        start_time = time.time()
        # each eigenvector has written as column. So, next we are using transpose matrix.
        frequencies, eigenvectors = calcSoftModes(system, self.R_val, self.N_val, self.val, kvector)
        self.extend([VibrationalMode(f, v) for f, v in zip(frequencies, eigenvectors.T)])
        self.duration_sm = time.time() - start_time


#
# if __name__ == '__main__':
#     from old.lib.fingerprints.Structure import Structure
#     test_dir = 'test_SoftModes2'
#     for i in range(0, 5):
#     # for i in range(4, 5):
#         filename = test_dir + '/POSCAR_%i' % (i + 1)
#         print('\nStructure: %s' % os.path.basename(filename))
#         print('-----------------------------------------------------------')
#
#         s = Structure(filename)
#
#         for j in range(2):  # makesupercell
#             s.make_supercell(dim=(j + 1))
#
#             sm = Softmodes(s)
#
#             H = sm.calc_hardness()
#             freq, eig = sm.calc_soft_modes()
#             # sm.calc_soft_modes_K()
#
#             print('Total N of Atoms: %i' % np.sum(s.num_atoms))
#             print('Hardness        : %.5f GPa' % H)
#             print('Time (hardness) : %10.4f seconds' % sm.duration_h)
#             print('Time (softmodes): %10.4f seconds' % sm.duration_sm)
#             print('Time (soft old) : %10.4f seconds' % sm.duration_sm_old)
#
#             diff = np.linalg.norm(sm.freq - sm.freq_old)
#
#             print('freq: %f <===> freq_old: % f   | diff: %.2e' % (
#                 np.linalg.norm(sm.freq), np.linalg.norm(sm.freq_old), diff))
#
#             if diff > 0.01:
#                 print('Warning, results are inconsistent, STOP')
#
#             print('')
#
#
