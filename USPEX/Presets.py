from os.path import join, expanduser, exists, dirname
from os import makedirs

from .IO.InputParser import read


def udateSystemWithPrefix(system, data, property, prefix):
    if prefix is None:
        if property == 'system':
            system.update(data)
        else:
            system[property] = data
    else:
        system[f'{prefix}.{property}'] = data


presetFitness = {
    ('aging', 'values'): ('plus', 'values', ('multiply', ('minus', ('mean', 'values'), ('min', 'values')),
                                             'antiseeds.corrections')),
    ('getRelCCHSpace', 'values'): ('getRelativeCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                          'simpleMoleculeUtility.composition'), 'values'),
    ('getAbsCCHSpace', 'values'): ('getAbsoluteCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                          'simpleMoleculeUtility.composition'), 'values'),
    ('heightCCH', 'values'): ('convexHullHeight', ('getRelCCHSpace', 'values')),
    ('heightCS', 'values'): ('simpleHeight', ('getRelCCHSpace', 'values')),
    'energyCCH': ('heightCCH', 'energy'),
    'enthalpyCCH': ('heightCCH', 'enthalpy'),
    'energyCS': ('heightCS', 'energy'),
    'enthalpyCS': ('heightCS', 'enthalpy'),
    'refinedEnergy': ('divide', ('minus', 'energy', 'onlyEnvironment.energy'), 'supercellFactor'),
    'refinedEnthalpy': ('divide', ('minus', 'enthalpy', 'onlyEnvironment.enthalpy'), 'supercellFactor'),
    'refinedEnergyCCH': ('heightCCH', 'refinedEnergy'),
    'refinedEnthalpyCCH': ('heightCCH', 'refinedEnthalpy'),
    'normRefinedEnthalpy': ('divide', 'refinedEnthalpy', 'cellUtility.area'),
    'normRefinedAbsCompCH': ('convexHullHeight', ('getAbsCCHSpace', 'normRefinedEnthalpy')),
    'refinedInterfaceEnergy': ('minus', ('minus', 'energy', 'onlyLowerEnvironment.energy'),
                               'onlyUpperEnvironment.energy'),
    'refinedInterfaceEnergyCCH': ('heightCCH', 'refinedInterfaceEnergy'),
}
