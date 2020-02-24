'''
@file        CrystalConfig.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        5 December 2016
@brief       Class for description of configuration of crystal target space.
'''

from .AtomisticConfig import AtomisticConfig
from ..XRay.SpectrumAnalyzer import SpectrumAnalyzer


class CrystalConfig(AtomisticConfig):
    '''

    '''

    def __init__(self, isConstLattice : bool=False, latticeValues=None, xraydata=None, **kwargs):
        '''

        :param isConstLattice:
        :param latticeValues:
        :param kwargs:
        '''
        #TODO
        # * latticeValues

        super(CrystalConfig, self).__init__(**kwargs)
        self.isConstLattice = isConstLattice
        self.latticeValues = latticeValues
        self.xraydata = xraydata

        if xraydata is not None:
            assert isinstance(xraydata, dict)
            self._spectrumAnalyzer = SpectrumAnalyzer(**xraydata)

    def isGoodSystem(self, SYSTEM) -> bool:
        '''
        Method which checks if the structure meet composition constraint.
        
        :param SYSTEM:
        :return:
        '''
        return super().isGoodSystem(SYSTEM) and self.isGoodLattice(SYSTEM)

    def xraydistance(self, system):
        if hasattr(self, '_spectrumAnalyzer'):
            return self._spectrumAnalyzer[system]
        else:
            raise RuntimeError('Cannot optimize the quantity xraydistance. No experimental X-ray data found.')
