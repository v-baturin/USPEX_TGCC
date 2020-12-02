import json
from ast import literal_eval
from pprint import pformat
from os.path import join, expanduser, exists

FILENAME = join(expanduser('~'), '.uspex-again.py')

if not exists(FILENAME):
    presetFitness = {
        'enthalpyCCH': ('convexHullHeight', ('getRelativeCHSpace', 'compositionSpace.numBlocks',  'enthalpy'))
    }

    presetOutput = {
        'CrystalFixComp': {
            'columns': [
                ('enthalpy', 'Enthalpy (eV)'),
                ('volume', 'Volume (A^3)'),
                ('symmetry', 'SYMMETRY (N)')
            ]
        },
        'CrystalVarComp': {
            'columns': [
                ('enthalpy', 'Enthalpy (eV)'),
                ('volume', 'Volume (A^3)'),
                ('symmetry', 'SYMMETRY (N)'),
                ('enthalpyCCH', 'Enthalpy per Block above CCH (eV)')
            ]
        }
    }

    content = {
        'presetFitness' : pformat(presetFitness, indent=4),
        'presetOutput' : pformat(presetOutput, indent=4)
    }
    with open(FILENAME, 'wt') as f:
        json.dump(content, f, indent=4)
else:
    with open(FILENAME, 'rt') as f:
        content = json.load(f)
    presetFitness = literal_eval(content['presetFitness'])
    presetOutput = literal_eval(content['presetOutput'])