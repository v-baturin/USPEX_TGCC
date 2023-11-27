from ..components import AtomicStructureRepresentation, PowderSpectrumAnalyzer, SingleCrystalSpectrumAnalyzer,\
    EnvironmentUtility, JunctionUtility, SimpleMoleculeUtility


def compileParams(main: dict) -> dict:
    stages = main['stages']
    for i, stage in enumerate(stages):
        if 'tag' not in stage:
            stages[i]['tag'] = str(i+1)
        if 'stageType' not in stage:
            stage['stageType'] = 'atomistic'
        if 'source' not in stage:
            stage['source'] = str(i) if i>0 else 'origin'

    if 'optimizer' in main and 'target' in main['optimizer']:
        optimizer = main['optimizer']
        target = optimizer['target']
        if 'defaultSuffix' not in target:
            target['defaultSuffix'] = stages[-1]['tag'] if stages else 'origin'
        symbols = target['compositionSpace']['symbols']
        defaultVolumeType = 0
        cutoffVDW = False
        molecules = {}
        molSitesMapping = {}
        elementalSymbols = set()
        for i, symbol in enumerate(symbols):
            if not isinstance(symbol, dict):
                elementalSymbols.add(symbol)
            else:
                molecule = AtomicStructureRepresentation.readXYZ(**symbol)
                if not len(molecule.edges):
                    molecule = SimpleMoleculeUtility.detectBonds(molecule)
                molecules[symbol['name']] = molecule
                symbols[i] = symbol['name']
                if 'sites' in symbol:
                    for site in symbol['sites']:
                        site['junctionTypes'] =\
                            JunctionUtility.calculateJunctionTypes(molecule,
                                                                   junctionsDescription=site['junctionTypes'])
                    molSitesMapping[symbol['name']] = symbol['sites']
                else:
                    defaultVolumeType = 0.5
                    cutoffVDW = True
                elementalSymbols |= set([x.short_name for x in molecule.getAtomTypes()])
        if 'junctionUtility' in target:
            target['junctionUtility']['molSitesMapping'] = molSitesMapping
        else:
            target['junctionUtility'] = {'molSitesMapping': molSitesMapping}
        if molecules:
            if 'simpleMoleculeUtility' in target:
                target['simpleMoleculeUtility']['molecules'] = molecules
            else:
                target['simpleMoleculeUtility'] = {'molecules': molecules}
        if 'selection' in optimizer:
            selection = optimizer['selection']
            if len(target['compositionSpace']['blocks']) > 1:
                selection['globalParentsPool'] = True
            if 'optType' not in selection:
                selection['optType'] = optimizer['optType']
        if 'bondUtility' not in target:
            target['bondUtility'] = {}
        if 'volumeType' not in target['bondUtility']:
            target['bondUtility']['volumeType'] = defaultVolumeType
        if cutoffVDW:
            target['bondUtility']['cutoff'] = 'vdw'
        if 'fingerprintUtility' not in optimizer:
            optimizer['fingerprintUtility'] = 'radialDistributionUtility'
        if 'powderSpectrumAnalyzer' in target:
            target['powderSpectrumAnalyzer'] = PowderSpectrumAnalyzer.parse(target['powderSpectrumAnalyzer'])
        if 'singleCrystalSpectrumAnalyzer' in target:
            sCS = target['singleCrystalSpectrumAnalyzer']
            sCS['expReflections'] = SingleCrystalSpectrumAnalyzer.parse(sCS.pop('hklFile'))
        if 'radialDistributionUtility' not in target:
            target['radialDistributionUtility'] = {}
        if 'symbols' not in target['radialDistributionUtility']:
            target['radialDistributionUtility']['symbols'] = sorted(elementalSymbols)
        if 'suffix' not in target['radialDistributionUtility']:
            target['radialDistributionUtility']['suffix'] = target['defaultSuffix']
        if 'environmentUtility' in target:
            for environmentDesciption in target['environmentUtility']['environments']:
                environmentDesciption.update(EnvironmentUtility.build(**environmentDesciption))

    return main
