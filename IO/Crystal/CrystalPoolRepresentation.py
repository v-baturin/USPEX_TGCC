import os
import io
import shutil
import matplotlib
import numpy as np

from ase.io import write
from os.path import join as pj


matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .CrystalSystemRepresentation import SystemsTable, CrystalSystemRepresentation

EXTENDED_CONVEX_HULL_ENERGY_RANGE = 0.5

class CrystalPoolRepresentation(object):

    def __init__(self, RES_FOLDER: str, columns, toDraw: list = None, rangeECH = EXTENDED_CONVEX_HULL_ENERGY_RANGE, **kwargs):
        self.RES_FOLDER = RES_FOLDER
        self.columns = columns
        if toDraw is None:
            self.toDraw = [('dep', 'enthalpy', 'raw', 'ID', 'raw'),
                           ('dep', 'enthalpy', 'per_atom', 'ID', 'raw'),
                           ('dep', 'enthalpy', 'per_atom', 'cellUtility.volume', 'per_atom'),
                           ('stat', 'enthalpy', 'per_atom', '', '')]
        else:
            self.toDraw = toDraw
        self.rangeECH = rangeECH

    def __call__(self, optimizers, optimizer):
        content_BESTIndividuals = ''
        content_convexHull = ''
        table_goodStructures = SystemsTable(self.columns, isRank=True)
        table_extendedConvexHull = SystemsTable(self.columns, isRank=True)
        io_BESTgatheredPOSCARS = io.StringIO('')
        io_goodStructuresPOSCARS = io.StringIO('')
        io_extendedConvexHullPOSCARS = io.StringIO('')

        os.makedirs(self.RES_FOLDER, exist_ok=True)

        fitness = optimizer.optType

        for generation, opt in enumerate(optimizers):
            content_BESTIndividuals += f'Generation {generation}\n'
            pool = opt.target.pool
            table = SystemsTable(self.columns)
            for ID in opt.best:
                table.update(ID, pool.allSystems[ID], opt.fitness)
            content_BESTIndividuals += table.table.get_string() + '\n'
        with open(pj(self.RES_FOLDER, 'BESTIndividuals'), 'w') as fp:
            fp.write(content_BESTIndividuals)

        for opt in optimizers:
            pool = opt.target.pool
            for ID in opt.best:
                CrystalSystemRepresentation.writeAtomicStructure(io_BESTgatheredPOSCARS, pool.allSystems[ID])
        with open(pj(self.RES_FOLDER, 'BESTgatheredPOSCARS'), 'w') as fp:
            io_BESTgatheredPOSCARS.seek(0)
            shutil.copyfileobj(io_BESTgatheredPOSCARS, fp)

        if len(optimizer.target.utilities.compositionSpace.blocks) == 1:
            allFitnesses = optimizer.fitness.getAllFitnesses(fitness)
            fronts = optimizer.fitness.sort(list(optimizer.target.pool.uniqueSystems), allFitnesses)
            for rank, front in enumerate(fronts):
                for system in front:
                    table_goodStructures.update(system['ID'], system, optimizer.fitness, rank=rank)
                    CrystalSystemRepresentation.writeAtomicStructure(io_goodStructuresPOSCARS, system)

            with open(pj(self.RES_FOLDER, 'goodStructures'), 'w') as fp:
                fp.write(table_goodStructures.table.get_string() + '\n')

            with open(pj(self.RES_FOLDER, 'goodStructures_POSCARS'), 'w') as fp:
                io_goodStructuresPOSCARS.seek(0)
                shutil.copyfileobj(io_goodStructuresPOSCARS, fp)


        else:
            for generation, opt in enumerate(optimizers):
                convexHull = [system for system, value
                              in zip(opt.target.pool.uniqueSystems, opt.fitness.calcFitness('enthalpyCCH'))
                              if value == 0]
                content_convexHull += f'Generation {generation}\n'
                table = SystemsTable(self.columns)
                for system in convexHull:
                    table.update(system['ID'], system, opt.fitness)
                content_convexHull += table.table.get_string() + '\n'

            with open(pj(self.RES_FOLDER, 'convex_hull'), 'w') as fp:
                fp.write(content_convexHull)

            extendedConvexHull = [system for system, value
                                  in zip(optimizer.target.pool.uniqueSystems, optimizer.fitness.calcFitness('enthalpyCCH'))
                                  if value < self.rangeECH]

            allFitnesses = optimizer.fitness.getAllFitnesses(fitness)
            fronts = optimizer.fitness.sort(extendedConvexHull, allFitnesses)

            for rank, front in enumerate(fronts):
                for system in front:
                    table_extendedConvexHull.update(system['ID'], system, optimizer.fitness, rank=rank)
            with open(pj(self.RES_FOLDER, 'extended_convex_hull'), 'w') as fp:
                fp.write(table_extendedConvexHull.table.get_string())

            for front in fronts:
                for system in front:
                    CrystalSystemRepresentation.writeAtomicStructure(io_extendedConvexHullPOSCARS, system)
            with open(pj(self.RES_FOLDER, 'extended_convex_hull_POSCARS'), 'w') as fp:
                io_extendedConvexHullPOSCARS.seek(0)
                shutil.copyfileobj(io_extendedConvexHullPOSCARS, fp)

            compositionSpace = optimizer.target.utilities.compositionSpace
            if len(compositionSpace.blocks) == 2 and convexHull:
                self._drawExtendedConvexHull2(compositionSpace, convexHull, extendedConvexHull)
            elif len(compositionSpace.blocks) == 3 and convexHull:
                self._drawExtendedConvexHull3(compositionSpace, convexHull, extendedConvexHull)

        self._drawProperties(optimizer.target.pool.uniqueSystems, optimizer.fitness)


    def _drawProperties(self, uniqueSystems, fitness):
        for type, propertyY, typeY, propertyX, typeX in self.toDraw:
            if type == 'dep':
                Y = []
                X = []
                for system in uniqueSystems:
                    valueX = fitness.getFitnessByID(propertyX, system['ID'])
                    valueY = fitness.getFitnessByID(propertyY, system['ID'])
                    if typeY == 'raw':
                        Y.append(valueY)
                    elif typeY == 'per_atom':
                        Y.append(valueY/len(system['molecules']))
                    if typeX == 'raw':
                        X.append(valueX)
                    elif typeX == 'per_atom':
                        X.append(valueX/len(system['molecules']))
                plt.clf()
                plt.plot(X,Y,'go')
                plt.ylabel(f'{propertyY}({typeY})')
                plt.xlabel(f'{propertyX}({typeX})')
                plt.savefig(pj(self.RES_FOLDER, f'{propertyY}({typeY})_vs_{propertyX}({typeX}).svg'))
            elif type == 'stat':
                Y = []
                for system in uniqueSystems:
                    value = fitness.getFitnessByID(propertyY, system['ID'])
                    if not np.isinf(value):
                        if typeY == 'raw':
                            Y.append(value)
                        elif typeY == 'per_atom':
                            Y.append(value/len(system['molecules']))
                plt.clf()
                plt.hist(Y, len(Y)//10+1, facecolor='g', alpha=0.75)
                plt.savefig(pj(self.RES_FOLDER, f'{propertyY}({typeY})_statistics.svg'))

    def _drawExtendedConvexHull2(self, compositionSpace, convexHull, extendedConvexHull):
        leftNumBlocks = np.asarray(compositionSpace.numBlocks(convexHull[0]['simpleMoleculeUtility.composition']), dtype = float)
        leftNumBlocksTotal = np.sum(leftNumBlocks)
        leftNumBlocks /= leftNumBlocksTotal
        leftEnthalpy = convexHull[0]['enthalpy']/leftNumBlocksTotal
        rightNumBlocks = leftNumBlocks
        rightEnthalpy = leftEnthalpy
        for system in convexHull:
            numBlocks = np.asarray(compositionSpace.numBlocks(system['simpleMoleculeUtility.composition']), dtype = float)
            numBlocksTotal = np.sum(numBlocks)
            numBlocks /= numBlocksTotal
            if numBlocks[1] < leftNumBlocks[1]:
                leftNumBlocks = numBlocks
                leftEnthalpy = system['enthalpy'] / numBlocksTotal
            elif numBlocks[1] > rightNumBlocks[1]:
                rightNumBlocks = numBlocks
                rightEnthalpy = system['enthalpy'] / numBlocksTotal
        Xch = []
        Ych = []
        for system in convexHull:
            numBlocks = np.asarray(compositionSpace.numBlocks(system['simpleMoleculeUtility.composition']), dtype = float)
            numBlocksTotal = np.sum(numBlocks)
            numBlocks /= numBlocksTotal
            C = np.array([leftNumBlocks, rightNumBlocks])
            E = np.array([leftEnthalpy, rightEnthalpy])
            Enthalpy = system['enthalpy']/numBlocksTotal - np.dot(np.linalg.lstsq(C.T, numBlocks)[0], E)
            Xch.append(numBlocks[1])
            Ych.append(Enthalpy)
        inds = np.argsort(Xch)
        Xch = np.asarray(Xch)[inds]
        Ych = np.asarray(Ych)[inds]
        X = []
        Y = []
        for system in extendedConvexHull:
            numBlocks = np.asarray(compositionSpace.numBlocks(system['simpleMoleculeUtility.composition']), dtype = float)
            numBlocksTotal = np.sum(numBlocks)
            numBlocks /= numBlocksTotal
            C = np.array([leftNumBlocks, rightNumBlocks])
            E = np.array([leftEnthalpy, rightEnthalpy])
            Enthalpy = system['enthalpy']/numBlocksTotal - np.dot(np.linalg.lstsq(C.T, numBlocks)[0], E)
            X.append(numBlocks[1])
            Y.append(Enthalpy)
        np.savetxt(pj(self.RES_FOLDER, 'ExtendedConvexHull.csv'), np.stack((X,Y), axis=-1), fmt='%6.3f', delimiter=',')
        plt.clf()
        plt.plot(X,Y,'go')
        plt.plot(Xch,Ych,'b-o')
        plt.ylabel('Enthalpy of formation (eV/atom)')
        components = []
        for block in compositionSpace.blocks:
            components.append(''.join(f'{symbol}{mult}' for symbol,mult in zip(compositionSpace.symbols, block)))
        plt.xlabel(f'Composition ratio: {components[1]}/({components[0]}+{components[1]})')
        plt.savefig(pj(self.RES_FOLDER, 'ExtendedConvexHull.svg'))


    def _drawExtendedConvexHull3(self, compositionSpace, convexHull, extendedConvexHull):
        pass
