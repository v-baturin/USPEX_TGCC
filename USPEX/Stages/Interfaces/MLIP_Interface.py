"""
USPEX.Stages.MLIP_Interface
===========================

.. codeauthor:: Michele Galasso <m.galasso@yandex.com>

"""
import logging
import shutil
import numpy as np

from pathlib import Path


logger = logging.getLogger(__name__)
EV_PER_CUBIC_ANGSTREM_PER_GPA = 1 / 160.21766208


class MLIP_Interface:
    '''
    Calculator for MLIP.
    Local running
    '''


    inputFile, outputFile, errorFile = 'input', 'output', 'error'
    argsFile = 'args'

    # working input files
    in_cfg_file = 'input.cfg'

    # working output files
    out_cfg_file = 'output.cfg'
    out_sampled_file = 'sampled.cfg_0'

    DEFAULT_SLEEP_TIME = 10
    atomisticRepresentation = None
    atomicDisassemblerType = None

    @classmethod
    def registerTypes(cls, atomisticRepresentation, atomicDisassemblerType):
        cls.atomisticRepresentation = atomisticRepresentation
        cls.atomicDisassemblerType = atomicDisassemblerType

    def __init__(self, tag: str, mode: str, potential: str, specorder, args: str = None, trainingSet: str = None,
                 targetProperties: list = None, **kwargs):

        self.tag = tag
        self.tmp = f'tmp_{tag}'
        self.mode = mode
        self.potential = potential
        self.specorder = specorder
        self.trainingSet = trainingSet
        if self.mode == 'select_add':
            assert self.trainingSet is not None
        argsFile = f'Specific/mlip_args_{tag}' if args is None else args
        with open(pj(os.getcwd(), argsFile)) as f:
            self.args = f.read()
        if targetProperties is not None:
            self.targetProperties = targetProperties
        elif self.mode == 'train':
            self.targetProperties = ['potential', 'trainingSet']
        elif self.mode == 'select_add':
            self.targetProperties = ['sample']
        else:
            self.targetProperties = []

    def prepareLocalCalculation(self, system, calcFolder: str):

        # create empty input file
        with open(calcFolder/self.inputFile, 'wt') as f:
            pass

        if 'trajectory' in system:
            sample = system['trajectory']
        elif 'population' in system:
            sample = []
            for individual in system['population']:
                sample.extend(individual['trajectory'])
        else:
            raise RuntimeError('No mlip sample in system.')
        self.atomisticRepresentation.saveMLIPsample(pj(calcFolder, self.in_cfg_file), self.specorder, sample)

        shutil.copy2(self.potential, calcFolder)

        if self.mode == 'train':
            args = f'train {bn(self.potential)} {self.in_cfg_file} {self.args}'
        elif self.mode == 'select_add':
            args = f'select_add {bn(self.potential)} {bn(self.trainingSet)}' \
                   f' {self.in_cfg_file} {self.out_cfg_file} {self.args}'
            shutil.copy2(self.trainingSet, calcFolder)
        else:
            raise RuntimeError(f'Mode {self.mode} unsupported.')

        return args

    def isConverged(self, calcFolder: str):
        if os.path.isfile(pj(calcFolder, self.out_cfg_file)):
            with open(pj(calcFolder, self.out_cfg_file), 'r') as f:
                content = f.read()
            if content:
                return True
            # # if the structure ended up unrelaxed because of extrapolation
            # elif os.path.isfile(pj(calcFolder, self.out_sampled_file)):
            #     with open(pj(calcFolder, self.errorFile)) as stderr:
            #         content = stderr.read()
            #     if not content:
            #         return True
        return self.mode == 'train'

    def readOutput(self, system, calcFolder: str):
        results = {}
        if 'sample' in self.targetProperties:
            sample = self.atomisticRepresentation.readMLIPsample(pj(calcFolder, self.out_cfg_file), self.specorder)
            system['sample'] = sample
        if 'potential' in self.targetProperties:
            shutil.copy2(pj(calcFolder, bn(self.potential)), self.potential)
        with open(pj(calcFolder, self.in_cfg_file), 'r') as f:
            content = f.read()
        results['isStable'] = len(content) == 0
        if 'trainingSet' in self.targetProperties:
            with open(self.trainingSet, 'a') as f:
                f.write(content)

        # if 'stressTensor' in self.targetProperties:
        #     stress_tensor = np.zeros((3, 3))
        #     stress_tensor[0, 0] = data['stresses'][0]
        #     stress_tensor[1, 1] = data['stresses'][1]
        #     stress_tensor[2, 2] = data['stresses'][2]
        #     stress_tensor[1, 2] = data['stresses'][3]
        #     stress_tensor[2, 1] = data['stresses'][3]
        #     stress_tensor[0, 2] = data['stresses'][4]
        #     stress_tensor[2, 0] = data['stresses'][4]
        #     stress_tensor[0, 1] = data['stresses'][5]
        #     stress_tensor[1, 0] = data['stresses'][5]
        #
        # else:
        #     ID = system['ID']
        #     logger.info(f'structure {ID} led to extrapolation and will be discarded.')
        #     # system['structure'].set_cell(np.identity(3) * system['structure'].minVectorLength * 0.9)
        #     system['enthalpy'] = 1000
        return results
