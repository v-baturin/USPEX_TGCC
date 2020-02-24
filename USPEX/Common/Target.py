import logging
logger = logging.getLogger(__name__)


'''
@file        Target.py
@author:     Pavel Bushlanov
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    paulbush@mail.ru
@date        21 July 2016
@brief       Abstract class for different target configuration spaces
'''


from types import SimpleNamespace

from .VariationOperators import VariationOperators


class Target(object):

    knownTargetTypes = {}

    def __init__(self, type : str, **kwargs):
        targetDef = SimpleNamespace(**self.knownTargetTypes[type])
        self.config = targetDef.configType(**kwargs)
        self.pool = targetDef.poolType(self.config)
        
        self.hybridizations = []
        for hybridizationType in targetDef.variationOperators.hybridizationTypes:
            if hybridizationType.__name__ in kwargs:
                self.hybridizations.append(hybridizationType(self.config, self.pool, **kwargs[hybridizationType.__name__]))

        self.mutations = []
        for mutationType in targetDef.variationOperators.mutationTypes:
            if mutationType.__name__ in kwargs:
                self.mutations.append(mutationType(self.config, self.pool, **kwargs[mutationType.__name__]))

        self.creations = []
        for creationType in targetDef.variationOperators.creationTypes:
            if creationType.__name__ in kwargs:
                self.creations.append(creationType(self.config, self.pool, **kwargs[creationType.__name__]))

        self.variationOperators = self.hybridizations + self.mutations + self.creations


    @classmethod
    def registerTarget(cls, name : str, configType : type, poolType : type, variationOperators : VariationOperators):
        assert name not in cls.knownTargetTypes
        cls.knownTargetTypes[name] = {'configType' : configType, 'poolType' : poolType,
                                      'variationOperators' : variationOperators}
