import unittest
import json
import filecmp
import shutil

from pathlib import Path

from ...Optimizers.PoolEntry import PoolEntry, EntryFlavour
from ...components import GlobalOptimizer, Atomistic
from ..OutputRepresentation import OutputRepresentation

TESTPATH = Path(__file__).parent


class Output_Test(unittest.TestCase):
    def test_1(self):
        folder_name = 'output_results'
        folder_name_ref = 'output_reference'

        optimizerConfig = {'type': 'GlobalOptimizer',
                           'target': {'type': 'Atomistic',
                                      'conditions': {'externalPressure': 100},
                                      'compositionSpace': {'symbols': ['Mg', 'Al', 'O'], 'blocks': [[4, 8, 16]],
                                                           'range': [[1, 1]]},
                                      'cellUtility': {'pbc': (1,1,1)},
                                      'radialDistributionUtility': {'symbols': ['Mg', 'Al', 'O'], 'suffix': 5}
                                      },
                           'fingerprintUtility': 'radialDistributionUtility',
                           'optType': '.enthalpy.5',
                           'stopFitness': -655.062,
                           'selection': {'type': 'USPEXClassic', 'popSize': 10,
                                         'optType': ('aging', '.enthalpy.5'),
                                         'fractions': {'heredity': [0.0, 1.0, 0.5],
                                                       'twinning': [0.0, 1.0, 0.1],
                                                       'softmodemutation': [0.0, 1.0, 0.1],
                                                       'randSym': [0.0, 1.0, 0.1],
                                                       'randTop': [0.0, 1.0, 0.1],
                                                       'permutation': [0.0, 1.0, 0.1]
                                                       }
                                         }
                           }
        numStages = 5
        stages = [{'tag': f'{i+1}', 'type': 'gulp'} for i in range(numStages)]
        popSize = 10
        numGenerations = 3
        stopCrit = 3
        numParallelCalcs = 2
        output = {
            'stages': ['1', '2', '3', '4', '5'],
            'columns': [
                ('.enthalpy.5', 'Enthalpy (eV)'),
                ('cellUtility.volume.5', 'Volume (A^3)'),
                ('cellUtility.symmetry.5', 'SYMMETRY (N)'),
            ],
            'presentConvexHull': False,
            'presentPareto': (),
            'toDraw': [
                ('dep', '.enthalpy.5', 'per_atom', 'ID', 'raw'),
                ('dep', '.enthalpy.5', 'raw', 'ID', 'raw'),
                ('dep', '.enthalpy.5', 'per_atom', 'cellUtility.volume.5', 'per_atom'),
                ('stat', '.enthalpy.5', 'per_atom', '', ''),
            ]
        }

        infos = []
        optimizer = GlobalOptimizer(**optimizerConfig)
        extensions = optimizer.target.propertyExtensions

        for gen in range(numGenerations):
            for i in range(popSize):
                with open(TESTPATH / f"output_data/system{gen * popSize + i}s0", "r") as f:
                    structure = json.load(f)
                structure.update(Atomistic.readAtomicStructure(
                    TESTPATH / f"output_data/system{gen * popSize + i}s0.vasp"))
                system = PoolEntry(structure['ID'], EntryFlavour(extensions=extensions, **structure))
                optimizer.pool.allSystems[system.ID] = system
                for j in range(numStages):
                    try:
                        with open(TESTPATH/f"output_data/system{gen * popSize + i}s{j+1}", "r") as f:
                            structure = json.load(f)
                        structure.update(Atomistic.readAtomicStructure(TESTPATH/f"output_data/system{gen*popSize+i}s{j+1}.vasp"))
                        for key, value in structure.items():
                            if key == 'ID':
                                assert value == system.ID
                                continue
                            prefix, prop = key.split('.')
                            system.setProperty(prop, value, extension=prefix, suffix=str(j+1))
                    except FileNotFoundError:
                        break
                system.getProperty('structure', extension='atomistic', suffix='5')
                system.getProperty('structure', extension='atomistic', suffix='origin')
            with open(TESTPATH/f"output_data/analisis{gen}", "r") as f:
                infos.append(json.load(f))
            with open(TESTPATH/f"output_data/targetState{gen}", "r") as f:
                targetState = json.load(f)
                for ID in targetState[1]:
                    optimizer.pool.allSystems[ID].setProperty('isBad', False, suffix='5')
                    if ID not in optimizer.pool.goodSystemIDs:
                        optimizer.pool.goodSystemIDs.append(ID)
                optimizer.best = set(targetState[0])
            with open(TESTPATH/f"output_data/population{gen}", "r") as f:
                population = [optimizer.pool.allSystems[ID] for ID in json.load(f)]
            optimizer.pool.generations.append(dict(
                bestSystems=optimizer.best,
                allSystems=population
            ))

        representation = OutputRepresentation(optimizer, optimizer=optimizerConfig,
                                              stages=stages, numParallelCalcs=numParallelCalcs,
                                              numGenerations=numGenerations, stopCrit=stopCrit,
                                              path=TESTPATH/folder_name,
                                              output=output)

        representation.presentSystems(optimizer)
        representation.presentOutput(optimizer, printDate=False)

        dcmp = filecmp.dircmp(TESTPATH/folder_name_ref, TESTPATH/folder_name)
        self.assertEqual(len(dcmp.diff_files), 0)
        self.assertEqual(len(dcmp.common_dirs), 1)
        for diff_file in dcmp.subdirs[dcmp.common_dirs[0]].diff_files:
            self.assertTrue(".svg" in diff_file or "POSCARS" in diff_file)

        shutil.rmtree(TESTPATH/folder_name)
