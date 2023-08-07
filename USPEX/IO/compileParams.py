from ..components import AtomisticRepresentation, PowderSpectrumAnalyzer, SingleCrystalSpectrumAnalyzer,\
    EnvironmentUtility, JunctionUtility


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
        symbols = target['compositionSpace']['symbols']
        defaultVolumeType = 0
        cutoffVDW = False
        molecules = {}
        molSitesMapping = {}
        elementalSymbols = set()
        for i, symbol in enumerate(symbols):
            if not isinstance(symbol, dict):
                elementalSymbols.add(symbol)
            elif 'type' in symbol and symbol.pop('type') == 'adsorbant':
                structure = AtomisticRepresentation.readXYZ(symbol['filename'])
                molecules[symbol['name']] = structure
                symbols[i] = symbol['name']
                for site in symbol['sites']:
                    site['junctionTypes'] =\
                        JunctionUtility.calculateJunctionTypes(structure,
                                                               junctionsDescription=site['junctionTypes'])
                molSitesMapping[symbol['name']] = symbol['sites']
                elementalSymbols |= set([x.short_name for x in structure.getAtomTypes()])
            else:
                defaultVolumeType = 0.5
                cutoffVDW = True
                structure = AtomisticRepresentation.readMol(symbol['filename'])
                molecules[symbol['name']] = structure
                symbols[i] = symbol['name']
                elementalSymbols |= set([x.short_name for x in structure.getAtomTypes()])
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
        if 'environmentUtility' in target:
            for environmentDesciption in target['environmentUtility']['environments']:
                environmentDesciption.update(EnvironmentUtility.build(**environmentDesciption))

    return main
