import logging
import sys
from copy import copy

from ..XRay.PowderSpectrumAnalyzer import PowderSpectrumAnalyzer
from ..XRay.SingleCrystalSpectrumAnalyzer import SingleCrystalSpectrumAnalyzer
from .read_molecule import read_molecule

logger = logging.getLogger(__name__)


def compileParams(main: dict, **definitions) -> dict:
    stages = main['stages']
    for i, stage in enumerate(stages):
        if stage in definitions:
            stage = copy(definitions[stage])
            assert isinstance(stage, dict)
            stages[i] = stage
        if 'taskManager' in stage and isinstance(stage['taskManager'], str) and stage['taskManager'] in definitions:
            taskManager = definitions[stage['taskManager']]
            assert isinstance(taskManager, dict)
            stage['taskManager'] = taskManager
        if 'tag' not in stage:
            stages[i]['tag'] = str(i+1)

    if 'optimizer' in main and 'target' in main['optimizer']:
        optimizer = main['optimizer']
        target = optimizer['target']
        assert 'compositionSpace' in target
        assert 'symbols' in target['compositionSpace']
        symbols = target['compositionSpace']['symbols']
        molecules = {}
        for i, symbol in enumerate(symbols):
            if symbol in definitions:
                try:
                    molDct = read_molecule(definitions[symbol]['filename'])
                    molecules[symbol] = molDct
                except Exception as ex:
                    logger.exception(ex)
                    exc_info = sys.exc_info()
                    raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
            if molecules:
                target['simpleMoleculeUtility'] = {'molecules': molecules}
        if 'ionDistances' not in target:
            target['ionDistances'] = {}
        if 'volumeType' not in target['ionDistances']:
            if molecules:
                target['ionDistances']['volumeType'] = 0.5
            else:
                target['ionDistances']['volumeType'] = 0
        if 'environmentUtility' in target:
            assert 'environments' in target['environmentUtility']
            for i, environment in enumerate(target['environmentUtility']['environments']):
                assert environment in definitions
                target['environmentUtility']['environments'][i] = definitions[environment]
        if 'fingerprintUtility' not in optimizer:
            optimizer['fingerprintUtility'] = 'radialDistributionUtility'
        selection = optimizer['selection']
        assert 'optType' in optimizer
        if 'optType' not in selection:
            selection['optType'] = optimizer['optType']
        if 'powderSpectrumAnalyzer' in target:
            filename = target['powderSpectrumAnalyzer']
            try:
                target['powderSpectrumAnalyzer'] = PowderSpectrumAnalyzer.parse(filename)
            except Exception as ex:
                logger.exception(ex)
                exc_info = sys.exc_info()
                raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
        if 'singleCrystalSpectrumAnalyzer' in target:
            dct = {}
            if 'cellParameters' in target['singleCrystalSpectrumAnalyzer']:
                dct['cellParameters'] = target['singleCrystalSpectrumAnalyzer']['cellParameters']
            try:
                hklFile = target['singleCrystalSpectrumAnalyzer']['hklFile']
                dct['expReflections'] = SingleCrystalSpectrumAnalyzer.parse(hklFile)
                target['singleCrystalSpectrumAnalyzer'] = dct
            except Exception as ex:
                logger.exception(ex)
                exc_info = sys.exc_info()
                raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    return main
