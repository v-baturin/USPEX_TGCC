import logging
logger = logging.getLogger(__name__)

import sys
import numpy as np
from copy import copy

from ..XRay.PowderSpectrumAnalyzer import PowderSpectrumAnalyzer
from ..XRay.SingleCrystalSpectrumAnalyzer import SingleCrystalSpectrumAnalyzer
from ..Atomistic.mol.read_molecule import read_molecule
from ..Presets import presetOutput


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
        if 'compositionSpace' in target:
            assert 'symbols' in target['compositionSpace']
            symbols = target['compositionSpace']['symbols']
            for i, symbol in enumerate(symbols):
                if symbol in definitions:
                    try: 
                        mol = read_molecule(definitions[symbol]['filename'])
                        symbols[i] = mol
                    except Exception as ex:
                        logger.exception(ex)
                        exc_info = sys.exc_info()
                        raise exc_info[0].with_traceback(exc_info[1], exc_info[2])
    if 'output' not in main:
        if np.fromiter((minBlock == maxBlock for minBlock, maxBlock
                        in main['optimizer']['target']['compositionSpace']['range']), dtype=bool).all():
            main['output'] = presetOutput['CrystalFixComp']
        else:
            main['output'] = presetOutput['CrystalVarComp']
    return main
