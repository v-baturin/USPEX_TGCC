import logging
logger = logging.getLogger(__name__)

import sys
import numpy as np
from copy import copy

from ..XRay.PowderSpectrumAnalyzer import PowderSpectrumAnalyzer
from ..XRay.SingleCrystalSpectrumAnalyzer import SingleCrystalSpectrumAnalyzer
from ..Presets import presetOutput
from .read_molecule import read_molecule


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
        target = main['optimizer']['target']
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
        if 'conditions' not in target:
            target['conditions'] = {}
        if 'volumeType' not in target['conditions']:
            if molecules:
                target['conditions']['volumeType'] = 0.5
            else:
                target['conditions']['volumeType'] = 0

        if 'powderSpectrumAnalyzer' in target:
            filename = target['powderSpectrumAnalyzer']
            try:
                target['powderSpectrumAnalyzer'] = PowderSpectrumAnalyzer.parse(filename)
            except Exception as ex:
                logger.exception(ex)
                exc_info = sys.exc_info()
                raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
        if 'singleCrystalSpectrumAnalyzer' in target:
            filename = target['singleCrystalSpectrumAnalyzer']
            try:
                target['singleCrystalSpectrumAnalyzer'] = SingleCrystalSpectrumAnalyzer.parse(filename)
            except Exception as ex:
                logger.exception(ex)
                exc_info = sys.exc_info()
                raise exc_info[0].with_traceback(exc_info[1], exc_info[2])

    return main
