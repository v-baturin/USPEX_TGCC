from os.path import join, expanduser, exists, dirname
from os import makedirs

from .IO.InputParser import read, write


FILENAME = join(expanduser('~'), '.config/uspex-again/presets.uspex')
makedirs(dirname(FILENAME), exist_ok=True)


if not exists(FILENAME):
    definitions = {
        'presetFitness': {
            'enthalpyCCH': ('convexHullHeight', ('getRelativeCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                                        'simpleMoleculeUtility.composition'), 'enthalpy'))
        },

        'presetOutput': {
            'CrystalFixComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('cellUtility.volume', 'Volume (A^3)'),
                    ('cellUtility.symmetry', 'SYMMETRY (N)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ]
            },
            'CrystalVarComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('enthalpyCCH', 'Enthalpy per Block above CCH (eV)'),
                    ('cellUtility.volume', 'Volume (A^3)'),
                    ('cellUtility.symmetry', 'SYMMETRY (N)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ]
            }
        }
    }

    write(FILENAME, definitions)
else:
    definitions = read(FILENAME)

presetFitness = definitions['presetFitness']
presetOutput = definitions['presetOutput']