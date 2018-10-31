'''
@file        CrystalConfig.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        5 December 2016
@brief       Class for description of configuration of crystal target space.
'''

from .AtomisticConfig import AtomisticConfig


class CrystalConfig(AtomisticConfig):
    '''

    '''

    def __init__(self, isConstLattice : bool =False, latticeValues=None, **kwargs):
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

    def isGoodSystem(self, SYSTEM) -> bool:
        '''
        :param SYSTEM:
        :return:
        '''
        return super().isGoodSystem(SYSTEM) and self.isGoodLattice(SYSTEM)

