import unittest
import os
import json
import filecmp
import shutil

from ...components import GlobalOptimizer, CrystalSystemRepresentation
from ..OutputRepresentation import OutputRepresentation

TESTPATH = os.path.dirname(os.path.abspath(__file__))


class Output_Test(unittest.TestCase):
    def test_1(self):
        folder_name = 'output_results'
        folder_name_ref = 'output_reference'

        optimizerConfig = {'type': 'GlobalOptimizer',
                           'target': {'type': 'Crystal',
                                      'conditions': {'externalPressure': 100, 'volumeType': 0},
                                      'compositionSpace': {'symbols': ['Mg', 'Al', 'O'], 'blocks': [[4, 8, 16]],
                                                           'range': [[1, 1]]},
                                      'cellUtility': {'pbc': (1,1,1)}
                                      },
                           'fingerprintUtility': 'radialDistributionUtility',
                           'fitness': 'enthalpy',
                           'stopFitness': -655.062,
                           'selection': {'type': 'USPEXClassic', 'popSize': 10,
                                         'fitness': ('getAntiseedsCorrections', 'enthalpy'),
                                         'fractions': {'heredity': [0.0, 1.0, 0.5],
                                                       'twinning': [0.0, 1.0, 0.1],
                                                       'softmodemutation': [0.0, 1.0, 0.1],
                                                       'randSym': [0.0, 1.0, 0.1],
                                                       'randTop': [0.0, 1.0, 0.1],
                                                       'permutation': [0.0, 1.0, 0.1]
                                                       }
                                         }
                           }
        popSize = 10
        numStages = 5
        numGenerations = 3
        numParallelCalcs = 2
        output = {
            'columns': [
                ('enthalpy', 'Enthalpy (eV)'),
                ('volume', 'Volume (A^3)'),
                ('symmetry', 'SYMMETRY (N)')
            ]
        }

        infos = []
        optimizers = []
        populations = []
        systems = {}
        for gen in range(numGenerations):
            for i in range(popSize):
                system = []
                for j in range(numStages + 1):
                    try:
                        with open(os.path.join(TESTPATH, f"output_data/system{gen * popSize + i}s{j}"), "r") as f:
                            structure = json.load(f)
                        with open(os.path.join(TESTPATH, f"output_data/system{gen*popSize+i}s{j}.vasp"), "r") as f:
                            structure.update(CrystalSystemRepresentation.readAtomicStructure(f))
                        system.append(structure)
                    except FileNotFoundError:
                        break
                systems[system[0]['ID']] = system
            with open(os.path.join(TESTPATH, f"output_data/analisis{gen}"), "r") as f:
                infos.append(json.load(f))
            with open(os.path.join(TESTPATH, f"output_data/targetState{gen}"), "r") as f:
                targetState = json.load(f)
                optimizer = GlobalOptimizer(**optimizerConfig)
                for ID in targetState[1]:
                    system = systems[ID][-1]
                    optimizer.target.pool.uniqueSystems += (system,)
                    optimizer.target.pool.allSystems[ID] = system
                optimizer.best = set(targetState[0])
                optimizers.append(optimizer)
            with open(os.path.join(TESTPATH, f"output_data/population{gen}"), "r") as f:
                populations.append([systems[ID][-1] for ID in json.load(f)])

        representation = OutputRepresentation(name=folder_name, optimizer=optimizerConfig,
                                              stages=[None]*numStages, numParallelCalcs=numParallelCalcs,
                                              path=os.path.join(TESTPATH, folder_name),
                                              output=output)

        representation.presentInfo(infos)
        representation.presentSystems(systems, optimizers[-1])
        representation.presentOutput(populations, optimizers[-1], printDate=False)
        representation.presentOptimizer(optimizers, optimizers[-1])

        dcmp = filecmp.dircmp(os.path.join(TESTPATH, folder_name_ref), os.path.join(TESTPATH, folder_name))
        self.assertEqual(len(dcmp.diff_files), 0)
        self.assertEqual(len(dcmp.common_dirs), 1)
        for diff_file in dcmp.subdirs[dcmp.common_dirs[0]].diff_files:
            self.assertTrue(".svg" in diff_file or "POSCARS" in diff_file)

        shutil.rmtree(os.path.join(TESTPATH, folder_name))
