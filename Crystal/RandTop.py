import logging
logger = logging.getLogger(__name__)

import itertools
import json
import numpy as np
import os
import pandas
import time

from sympy.combinatorics.partitions import Partition, RGS_rank

from ..SpaceGroups.TopologicalNet import TopologicalNet
from ..SpaceGroups.SpaceGroups3D import Group

from .Random import Random, randomPermutation, VOFailed


HOMEPATH = os.path.dirname(os.path.abspath(__file__))
with open(f'{HOMEPATH}/idealnets.json', 'rt') as f:
    TOPOLOGICAL_NETS = pandas.DataFrame.from_dict(json.load(f)).T


def randomPartitionSampler(itemsNumber : int, partitionsNumber : int, samplesNumber : int):
    if itemsNumber < partitionsNumber:
        raise ValueError("Number of items should be greater than or equal to number of partitions.")
    usedRGS = []
    attempts = 1000
    while (samplesNumber > len(usedRGS)) and attempts > 0:
        RGS = []
        n = itemsNumber - partitionsNumber
        for i in list(range(partitionsNumber)):
            if i == partitionsNumber - 1:
                k = n
            else:
                k = np.random.randint(n) if n else 0
            n -= k
            RGS.append(i)
            RGS.extend(np.random.randint(i + 1,size=k))
        if RGS_rank(RGS) not in usedRGS:
            usedRGS.append(RGS_rank(RGS))
            yield Partition.from_rgs(RGS, list(range(itemsNumber))).partition
        else:
            attempts -= 1

def decompose3(m123):
    """
    Lists all decompositions of a number into 3 multiplicands.
    :param m123: A number to be decomposed.
    :return: List of 3-tuples.
    """
    decompositions = [(m123, 1, 1)]
    for m1 in list(range(1, m123)):
        if not m123 % m1:
            m23 = m123 // m1
            decompositions.append((m1,m23,1))
            for m2 in list(range(1, m23)):
                if not m23%m2:
                    m3 = m23//m2
                    decompositions.append((m1,m2,m3))
    return decompositions


def generateStructureWithRandomTopology(numberOfAtoms3, supercells = None, coordinationNumbers3 = None):
    if coordinationNumbers3 is None:
        coordinationNumbers3 = np.zeros(len(numberOfAtoms3))
    totalAtomNubmber3 = sum(numberOfAtoms3)
    appropriateNets = TOPOLOGICAL_NETS[totalAtomNubmber3%TOPOLOGICAL_NETS['totalAtomNumber'] == 0]
    compstart  = time.perf_counter()
    for name, params in appropriateNets.sample(min(appropriateNets.shape[0], 100)).iterrows():
        compend = time.perf_counter()
        if compend - compstart > 60:
            break
        if totalAtomNubmber3//params['totalAtomNumber'] > 4:
            continue
        net = TopologicalNet(name, Group.getGroupFromSymbol(params['groupName']), params['nods'], params['bonds'])
        logger.debug(f'Trying {name} topology with {params["groupName"]} symmetry')
        if supercells is None:
            supercells = decompose3(totalAtomNubmber3 // params['totalAtomNumber'])
        else:
            supercells = [supercell for supercell in supercells if np.prod(supercell) == totalAtomNubmber3 // params['totalAtomNumber']]
        for supercell in randomPermutation(supercells):
            logger.debug(f'Trying {supercell} supercell')
            topstart = time.perf_counter()
            for flavour in randomPermutation(net.flavours(tuple(supercell))):
                topend = time.perf_counter()
                if topend - topstart > 15:
                    break
                if len(flavour.nodes) >= len(numberOfAtoms3):
                    atomPermutations = list(itertools.permutations(zip(list(range(len(numberOfAtoms3))), numberOfAtoms3, coordinationNumbers3)))
                    for nodePartition in randomPartitionSampler(len(flavour.multiplicities), len(numberOfAtoms3), 50):
                        logger.debug(f'Trying {nodePartition} partition')
                        for atoms3 in randomPermutation(atomPermutations):
                            permutationAtoms, numberOfAtoms, coordinationNumbers = zip(*atoms3)
                            numberOfNodes = [np.sum(flavour.multiplicities[np.asarray(nodeTypeGroup, dtype=int)])
                                              for nodeTypeGroup in nodePartition]
                            if np.all(np.asarray(numberOfNodes) == np.asarray(numberOfAtoms)):
                                coordinates = []
                                operations = []
                                for atomNumber in np.argsort(permutationAtoms):
                                    nodeIndices = np.asarray(list(nodePartition)[atomNumber], dtype=np.int)
                                    coordinates.append(flavour.group(flavour.nodes[nodeIndices]))
                                    operations.append([flavour.operations[ind] for ind in nodeIndices])
                                cell = np.asarray(params['cell']) * np.asarray(supercell)
                                return name, cell, coordinates, operations
    raise VOFailed



class RandTop(Random):

    name = 'TopRandom'

    logger = logger

    def __init__(self, *args, supercells = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.supercells = supercells

    def generateStructure(self, composition, latVol):
        symbols = list(composition.keys())
        numIons = list(composition.values())
        inds = np.flatnonzero(numIons)
        zeroInds = np.sort(list(set(range(len(numIons))) - set(inds)))
        for i in list(range(10)):
            try:
                # generateStructureWithRandomTopology properly works if all elements in numIons are nonzero.
                # inds are indices of nonzero elements of numIons.
                name, cell, coordinates, operations = generateStructureWithRandomTopology([numIons[i] for i in inds],
                                                                                          supercells=self.supercells)
            except VOFailed:
                raise VOFailed
            except Exception:
                self.logger.exception('Exception in generateStructureWithRandomTopology')
                continue
            # zeroInds are indices of zero elements of numIons. Here we reconstruct numIons consistent arrays
            # by inserting empty arrays at zeroInds sites of arrays created with generateStructureWithRandomTopology.
            for ind in zeroInds:
                coordinates.insert(ind, [])
                operations.insert(ind, [])
            all_coordinates = np.vstack([*itertools.chain(*coordinates)])
            if name in self.arxiv:
                for arxivCoordinates in self.arxiv[name]:
                    if (all_coordinates.shape == arxivCoordinates.shape) and np.allclose(all_coordinates, arxivCoordinates):
                        continue
            if name in self.arxiv:
                self.arxiv[name].append(all_coordinates)
            else:
                self.arxiv[name] = [all_coordinates]
            cell = cell * np.power(latVol / np.abs(np.linalg.det(cell)), 1.0 / 3.0)
            return name, cell, dict(zip(symbols, coordinates)), dict(zip(symbols, operations))
        raise VOFailed
