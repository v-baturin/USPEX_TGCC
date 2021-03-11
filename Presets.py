from os.path import join, expanduser, exists, dirname
from os import makedirs

from .InputParser import read, write


FILENAME = join(expanduser('~'), '.config/uspex-again/presets.uspex')
makedirs(dirname(FILENAME), exist_ok=True)


if not exists(FILENAME):
    definitions = {
        'presetFitness': {
            'enthalpyCCH': ('convexHullHeight', ('getRelativeCHSpace', 'compositionSpace.numBlocks',  'enthalpy'))
        },

        'presetOutput': {
            'CrystalFixComp': {
                'columns': [
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('cellUtility.volume', 'Volume (A^3)'),
                    ('cellUtility.symmetry', 'SYMMETRY (N)')
                ]
            },
            'CrystalVarComp': {
                'columns': [
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('cellUtility.volume', 'Volume (A^3)'),
                    ('cellUtility.symmetry', 'SYMMETRY (N)'),
                    ('enthalpyCCH', 'Enthalpy per Block above CCH (eV)')
                ]
            }
        }
    }

    write(FILENAME, definitions)
else:
    definitions = read(FILENAME)

presetFitness = definitions['presetFitness']
presetOutput = definitions['presetOutput']