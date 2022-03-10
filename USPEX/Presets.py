from os.path import join, expanduser, exists, dirname
from os import makedirs

from .IO.InputParser import read, write


FILENAME = join(expanduser('~'), '.config/uspex/presets.uspex')
makedirs(dirname(FILENAME), exist_ok=True)


if not exists(FILENAME):
    definitions = {
        'presetFitness': {
            'enthalpyCCH': ('convexHullHeight', ('getRelativeCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                                        'simpleMoleculeUtility.composition'), 'enthalpy')),
            'refinedEnthalpy': ('minus', 'enthalpy', 'environmentEnthalpy'),
            'normRefinedEnthalpy': ('divide', 'refinedEnthalpy', 'cellUtility.area'),
            'normRefinedAbsCompCH': ('convexHullHeight', ('getAbsoluteCHSpace', ('compositionSpace.numBlocksFromCompositions',
                                                                        'simpleMoleculeUtility.composition'), 'normRefinedEnthalpy')),
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
            },
            'Nano2DFixComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('cellUtility.area', 'Area (A^2)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ]
            },
            'Nano2DVarComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('enthalpyCCH', 'Enthalpy per Block above CCH (eV)'),
                    ('cellUtility.area', 'Volume (A^2)'),
                    ('normRefinedEnthalpy', 'Formation energy per unit area (eV / A^2)'),
                    ('normRefinedAbsCompCH', 'Formation energy per unit area above CCH (eV / A^2)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ]
            },
            'Nano1DFixComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('cellUtility.length', 'Period (A)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ]
            },
            'Nano1DVarComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('enthalpyCCH', 'Enthalpy per Block above CCH (eV)'),
                    ('cellUtility.length', 'Period (A)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ]
            },
            'Nano0DFixComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ]
            },
            'Nano0DVarComp': {
                'columns': [
                    ('simpleMoleculeUtility.composition', 'Composition'),
                    ('enthalpy', 'Enthalpy (eV)'),
                    ('enthalpyCCH', 'Enthalpy per Block above CCH (eV)'),
                    ('radialDistributionUtility.structureOrder', 'Structure order'),
                    ('radialDistributionUtility.averageOrder', 'Average order'),
                    ('radialDistributionUtility.quasientropy', 'Quasientropy')
                ]
            }

        }
    }

else:
    definitions = read(FILENAME)

presetFitness = definitions['presetFitness']
presetOutput = definitions['presetOutput']