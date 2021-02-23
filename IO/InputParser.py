'''
@file        InputParser.py
@author:     Artem Samtsevich
@copyright:  2017 Oganov's Lab. All rights reserved.
@contact:    samtsevichartem@gmail.com
@date        21 July 2016
@brief       Class for parsing input parameters
'''
CODE_NAMES = {
    1: 'vasp',
    3: 'gulp',
    4: 'lammps'
}

CODE_INPUT_NAMES_KEYS = {
    1: 'incar',
    3: 'ginput',
    4: 'lammps_in'
}

CODE_INPUT_NAMES_VALUES = {
    1: 'Specific/INCAR',
    3: 'Specific/ginput',
    4: 'Specific/lammps.in'
}

CODE_INPUT_OPTIONS_KEYS = {
    3: 'goptions'
}

CODE_INPUT_OPTIONS_VALUES = {
    3: 'Specific/goptions'
}

SPECIFIC_COPY_EXCEPTIONS = {
    1: ['INCAR','POTCAR'],
    3: ['ginput', 'goptions'],
    4: ['lammps.in']

}
import os


class InputParser(object):

    def __init__(self, filePath=None, wd = './'):
        assert isinstance(filePath, str) or filePath is None
        self.filePath = filePath
        self.wd = wd

    def parse(self):
        try:
            with open(self.wd + self.filePath, 'r') as f:
                content = f.readlines()
        except:
            return None

        input_variables = {}
        newBlock = False
        for line in content:
            if line.strip()[:2] in ['#', '*']:
                continue
            for comment_char in ['#', '*']:
                if comment_char in line:
                    line = line.split(comment_char)[0]
            if not line:
                continue
            if ':' in line and not newBlock:
                row = line.split(':')
                key, value = row[1].split()[0].strip(), row[0].strip()
                input_variables[key] = value
                continue
            if '%' in line and not 'end' in line.lower():
                blockName = line.split('%')[1][:-1].strip(' ')
                blockContent = ''
                newBlock = True
                continue
            if '%end' in line.lower().replace(' ', ''):
                newBlock = False
                input_variables[blockName] = blockContent
            if newBlock:
                blockContent += line

        params = {
            'system': {},
            'stages': []
        }

        systemType = None
        varcomp = False
        systemTypeDict = {}
        systemTypeDictHeredity = {}
        systemTypeDictTwinning = {}
        systemTypeDictRandtop = {}

        SimpleKeys = ['numParallelCalcs', 'numGenerations', 'stopCrit', 'stopFitness', 'repeatForStatistics']
        SystemTypeSimpleKeys = ['minAngle', 'maxAngle', 'maxAt', 'minAt', 'minVectorLength']

        if 'KresolStart' in input_variables.keys():
            KPOINTS = []
            for i, kres in enumerate(input_variables['KresolStart'].split()):
                KPOINTS.append(float(kres))

        for key in input_variables.keys():

            if key in SimpleKeys:
                if not '.' in input_variables[key]:
                    params[key] = int(input_variables[key])
                else:
                    params[key] = float(input_variables[key])
                continue

            if key in SystemTypeSimpleKeys:
                if not '.' in input_variables[key]:
                    systemTypeDict[key] = int(input_variables[key])
                else:
                    systemTypeDict[key] = float(input_variables[key])
                continue

            if key == 'calculationMethod':
                params['engine'] = {'type': input_variables[key], 'popSize': int(input_variables.get('populationSize', None)),
                                                    'initialPopSize': int(input_variables.get('initialPopSize', None))}
                continue

            if key == 'abinitioCode':
                for i, code in enumerate(input_variables[key].split()):
                    code = int(code)
                    assert code in CODE_NAMES.keys()
                    codeDict = {
                        'type': CODE_NAMES[code],
                        'params': {
                            'commandExecutable': input_variables['commandExecutable'].strip('\n'),
                            'workingDirectory' : './'
                        }
                    }

                    if 'whichTaskManager' in input_variables.keys():
                        with open(self.wd + 'HEADER') as f:
                            tmHeader = f.read()
                        tm = input_variables['whichTaskManager']
                        codeDict['params'].update({'taskManager': {'type': tm, 'header': tmHeader}})

                    if 'remote' in input_variables.keys():
                        print("Remote is not implemented")
                        exit(1)

                    if code == 1:
                        codeDict['params']['potcars'] = []
                        specificFiles = os.listdir(self.wd + 'Specific')
                        for file in specificFiles:
                            if 'POTCAR' in file:
                                codeDict['params']['potcars'].append('Specific/' + file)

                    if code in [3,4]:
                        codeDict['params']['libs'] = []
                        specificFiles = os.listdir(self.wd + 'Specific')
                        for file in specificFiles:
                            if not any([item in file for item in SPECIFIC_COPY_EXCEPTIONS[code]]):
                                codeDict['params']['libs'].append('Specific/' + file)

                    if 'KresolStart' in input_variables.keys():
                        codeDict['params'].update({'kresol' : KPOINTS[i]})
                    if code in CODE_INPUT_NAMES_KEYS.keys():
                        codeDict['params'].update({CODE_INPUT_NAMES_KEYS[code] : CODE_INPUT_NAMES_VALUES[code] + '_{:d}'.format(i + 1)})
                    if code in CODE_INPUT_OPTIONS_KEYS.keys():
                        codeDict['params'].update({CODE_INPUT_OPTIONS_KEYS[code] : CODE_INPUT_OPTIONS_VALUES[code] + '_{:d}'.format(i + 1)})

                    params['stages'].append(codeDict)

                continue

            if key == 'calculationType':
                if input_variables[key][0] == '3':
                    systemTypeDict['type'] = 'Crystal'
                if input_variables[key][2] == '1':
                    varcomp = True
                continue

            if key == 'atomType':
                systemTypeDict['symbols'] = input_variables[key].split()
                continue

            if key == 'numSpecies':
                systemTypeDict['blocks'] = [[int(x) for x in line.split()] for line in input_variables['numSpecies'].split('\n')[:-1]]
                continue

            if key == 'IonDistances':
                systemTypeDict['ionDistances'] = [
                    [float(x) for x in line.split()] for line in input_variables[key].split('\n')[:-1]
                ]
                continue

            if key == 'ExternalPressure':
                systemTypeDict['externalPressure'] = float(input_variables[key])
                continue

            for key, operator in zip(['fracGene', 'fracRand', 'fracTopRand', 'fracTwin', 'fracPerm', 'fracAtomsMut'],
                                     ['heredity', 'random', 'randtop', 'twinning', 'permutation', 'softmodemutation']):
                if key in input_variables.keys():
                    systemTypeDict[operator] = {'initFrac': float(input_variables[key])}

        if varcomp:
            maxAt = int(systemTypeDict['maxAt'])
            fixed = [[0, maxAt]] * len(systemTypeDict['blocks'])
        else:
            fixed = [[1, 1]] * len(systemTypeDict['blocks'])

        systemTypeDict['fixed'] = fixed
        params['system'] = systemTypeDict
        return params
