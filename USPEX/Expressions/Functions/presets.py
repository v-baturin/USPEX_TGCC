presetFitness = {
    ('aging', 'values'): ('plus', 'values', ('multiply', ('minus', ('mean', 'values'), ('min', 'values')),
                                             'antiseeds.corrections.origin')),
    ('getRelCCHSpace', 'values'): ('getRelativeCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                          'simpleMoleculeUtility.composition.origin'), 'values'),
    ('getAbsCCHSpace', 'values'): ('getAbsoluteCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                          'simpleMoleculeUtility.composition.origin'), 'values'),
    ('heightCCH', 'values'): ('convexHullHeight', ('getRelCCHSpace', 'values')),
    ('heightCS', 'values'): ('simpleHeight', ('getRelCCHSpace', 'values')),
    # 'energyCCH': ('heightCCH', 'energy'),
    # 'enthalpyCCH': ('heightCCH', 'enthalpy'),
    # 'energyCS': ('heightCS', 'energy'),
    # 'enthalpyCS': ('heightCS', 'enthalpy'),
    # 'refinedEnergy': ('divide', ('minus', 'energy', 'onlyEnvironment.energy'), 'supercellFactor'),
    # 'refinedEnthalpy': ('divide', ('minus', 'enthalpy', 'onlyEnvironment.enthalpy'), 'supercellFactor'),
    # 'refinedEnergyCCH': ('heightCCH', 'refinedEnergy'),
    # 'refinedEnthalpyCCH': ('heightCCH', 'refinedEnthalpy'),
    # 'normRefinedEnthalpy': ('divide', 'refinedEnthalpy', 'cellUtility.area'),
    # 'normRefinedAbsCompCH': ('convexHullHeight', ('getAbsCCHSpace', 'normRefinedEnthalpy')),
    # 'refinedInterfaceEnergy': ('minus', ('minus', 'energy', 'onlyLowerEnvironment.energy'),
    #                            'onlyUpperEnvironment.energy'),
    # 'refinedInterfaceEnergyCCH': ('heightCCH', 'refinedInterfaceEnergy'),
}

def applyPresets(optType):
    optType_ref = optType
    if optType in presetFitness:
        optType = presetFitness[optType]
    elif isinstance(optType, tuple):
        funcName, *funcParams = optType
        for probeFitness in presetFitness.keys():
            if isinstance(probeFitness, tuple) and probeFitness[0] == funcName and len(probeFitness) == len(optType):
                funcName, *templateParams = probeFitness
                optType = presetFitness[probeFitness]
                for param, templateParam in zip(funcParams, templateParams):
                    optType = _substituteParams(optType, templateParam, param)
                break
    if optType != optType_ref:
        optType = applyPresets(optType)
    return optType

def applyPresetsRecursive(optType):
    if isinstance(optType, list):
        optType = tuple(optType)
    optType = applyPresets(optType)
    if isinstance(optType, tuple):
        optType = (optType[0], *(applyPresetsRecursive(param) for param in optType[1:]))
    return optType


def _substituteParams(optType, templateParam, param):
    if optType == templateParam:
        optType = param
    elif isinstance(optType, tuple):
        funcName, *funcParams = optType
        optType = (funcName,)
        for funcParam in funcParams:
            optType += (_substituteParams(funcParam, templateParam, param),)
    return optType

