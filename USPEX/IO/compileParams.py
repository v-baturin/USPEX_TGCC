from ..XRay.PowderSpectrumAnalyzer import PowderSpectrumAnalyzer
from ..XRay.SingleCrystalSpectrumAnalyzer import SingleCrystalSpectrumAnalyzer
from ..Atomistic.EnvironmentUtility import EnvironmentUtility
from .read_molecule import read_molecule


def compileParams(main: dict) -> dict:
    stages = main['stages']
    for i, stage in enumerate(stages):
        if 'tag' not in stage:
            stages[i]['tag'] = str(i+1)

    if 'optimizer' in main and 'target' in main['optimizer']:
        optimizer = main['optimizer']
        target = optimizer['target']
        symbols = target['compositionSpace']['symbols']
        molecules = {}
        elementalSymbols = set()
        for i, symbol in enumerate(symbols):
            if isinstance(symbol, dict):
                molDct = read_molecule(symbol['filename'])
                molecules[symbol['name']] = molDct
                symbols[i] = symbol['name']
                elementalSymbols |= set(molDct['symbols'])
            else:
                elementalSymbols.add(symbol)
            if molecules:
                target['simpleMoleculeUtility'] = {'molecules': molecules}
        selection = optimizer['selection']
        if len(target['compositionSpace']['blocks']) > 1:
            selection['globalParentsPool'] = True
        if 'bondUtility' not in target:
            target['bondUtility'] = {}
        if 'volumeType' not in target['bondUtility']:
            if molecules:
                target['bondUtility']['volumeType'] = 0.5
            else:
                target['bondUtility']['volumeType'] = 0
        if 'fingerprintUtility' not in optimizer:
            optimizer['fingerprintUtility'] = 'radialDistributionUtility'
        if 'optType' not in selection:
            selection['optType'] = optimizer['optType']
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
