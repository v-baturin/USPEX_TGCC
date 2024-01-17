import os.path
import random
import copy
import networkx as np
import networkx as nx
import numpy as np
from .tobacco_3.tobacco import __file__ as tobacco_path
TOBACCO_DIR = os.path.dirname(tobacco_path)

class MOF_Heredity:

    def __init__(self,utilities, suffix, nslabs=2, attempts=100):
        self.cellUtility = utilities.cellUtility
        self.atomistic = utilities.atomistic
        self.mofutility = utilities.mofUtility
        self.topology_heredity = Topology_heredity(utilities=utilities,suffix=suffix,nslabs=nslabs, attempts = attempts)
        self.tobacco_path = TOBACCO_DIR
        self.suffix = suffix
        self.conditions = utilities.conditions

    def __call__(self, mof1, mof2,flavourFactory = None):
        graph1, graph2 = mof1.getProperty('topology','mofutility'),mof2.getProperty('topology','mofutility')
        cell1, cell2 = mof1.getProperty('cell','atomistic'),mof2.getProperty('cell','atomistic')
        offspringGraph, cell = self.topology_heredity(graph1,cell1,graph2,cell2)
        offspringTemplatePath = self.write_graph2tobacco_template(offspringGraph,cell)
        MOF = self.mofutility.get_MOF_from_tobacco(offspringTemplatePath,self.cellUtility,self.atomistic,graph=offspringGraph)
        MOF = flavourFactory(**MOF)
        self.conditions.putConditions(MOF)
        return MOF

    def write_graph2tobacco_template(self,graph,cell):
        new_template_path = os.path.join(self.tobacco_path,'templates','offspring_template'+str(int(self.suffix)))
        with open(new_template_path,'w') as template:
            template.write('loop_\n')
            template.write('_symmetry_equiv_pos_as_xyz\n')
            template.write('x, y, z\n')
            template.write(f'_cell_length_a                    {cell.getCellParameters()[0]}\n')
            template.write(f'_cell_length_b                    {cell.getCellParameters()[1]}\n')
            template.write(f'_cell_length_c                    {cell.getCellParameters()[2]}\n')
            template.write(f'_cell_angle_alpha                   {cell.getCellParameters()[3]}\n')
            template.write(f'_cell_angle_beta                    {cell.getCellParameters()[4]}\n')
            template.write(f'_cell_angle_gamma                    {cell.getCellParameters()[5]}\n')
            template.write('loop_\n')
            template.write('_atom_site_label\n')
            template.write('_atom_site_type_symbol\n')
            template.write('_atom_site_fract_x\n')
            template.write('_atom_site_fract_y\n')
            template.write('_atom_site_fract_z\n')
            node_numbers = []
            for n in graph.nodes.data():
                node_number = int(''.join([l for l in n.name if l.isdigit()]))
                node_numbers.append(node_number)
            nodes = list(graph.nodes.data())
            nodes = [nodes[i] for i in np.argsort(node_numbers)]
            for n,n_data in nodes:
                node_name = str(n.name)
                node_symbol = ''.join([l for l in node_name if l.isalpha()])
                node_fcoords = graph.nodes[n]['fcoords']
                template.write(f'{node_name} {node_symbol} {str(node_fcoords)[1:-1]}\n')
            edges = list(graph.edges.data())
            neighbours_names = [(e[0].name,e[1].name) for e in edges]
            neighbours_numbers = []
            for name1,name2 in neighbours_names:
                number1 = ''.join([l for l in name1 if l.isdigit()])
                number2 = ''.join([l for l in name2 if l.isdigit()])
                neighbours_numbers.append((number1,number2))
            _neighbours_numbers = copy.copy(neighbours_numbers)
            _neighbours_numbers.sort()
            numbers_argsort = [neighbours_numbers.index(i) for i in _neighbours_numbers]
            edges = [edges[i] for i in numbers_argsort]
            for node1,node2,edge_data in edges:
                distance = edge_data['length']
                if np.all(edge_data['label'] == np.array([0, 0, 0])):
                    label = '.'
                else:
                    label = '1_'+''.join(str(edge_data+np.array([5,5,5]))[1:-1].split())
                template.write(f'{node1.name} {node2.name} {str(distance)} {label} S\n')
        return new_template_path




class Topology_heredity:
    def __init__(self, utilities, suffix, nslabs, attempts):
        self.nslabs = nslabs
        self.attempts = attempts
        self.cellUtility = utilities.cellUtility
        self.suffix = suffix
        self.mofutility = utilities.mofUtility


    def __call__(self, graph1, cell1, graph2, cell2):
        for n in graph1.nodes:
            graph1.nodes[n]['parent_graph'] = graph1
        composition1 = len(graph2.nodes)
        for n in graph2.nodes:
            graph2.nodes[n]['parent_graph'] = graph2
        composition2 = len(graph2.nodes)
        outputCell = random.choice((cell1,cell2))
        for i in range(self.attempts):
            if self.cellUtility.isGoodCell(outputCell):
                axis = np.random.randint(3)
                nslabs = self.nslabs
                gaugesOfSlabs = tuple(np.random.randint(3, 9, size=nslabs).tolist())
                slabs1 = Topology_slab.getRandomSlabs(graph1, inputCell=cell1, outputCell=outputCell,
                                             axis=axis, gaugesOfSlabs=gaugesOfSlabs)

                slabs2 = Topology_slab.getRandomSlabs(graph2, inputCell=cell2, outputCell=outputCell,
                                             axis=axis, gaugesOfSlabs=gaugesOfSlabs)

                goodCandidateNodes = []
                goodCandidateDepths = []
                badCandidateNodes = []
                badCandidateDepths = []

                parity = 0
                for slab1, slab2 in zip(slabs1, slabs2):
                    if parity == 0:
                        goodCandidateNodes.extend(slab1.nodes)
                        goodCandidateDepths.extend(slab1.depths)
                        badCandidateNodes.extend(slab2.nodes)
                        badCandidateDepths.extend(slab2.depths)
                        parity = 1
                    else:
                        badCandidateNodes.extend(slab1.nodes)
                        badCandidateDepths.extend(slab1.depths)
                        goodCandidateNodes.extend(slab2.nodes)
                        goodCandidateDepths.extend(slab2.depths)
                        parity = 0

                goodCandidateNodes = [goodCandidateNodes[i] for i in reversed(np.argsort(goodCandidateDepths))]
                badCandidateNodes = [badCandidateNodes[i] for i in np.argsort(badCandidateDepths)]

                desiredComposition = round((composition1+composition2)/2)
                all_nodes = goodCandidateNodes+badCandidateNodes
                outputNodes = [all_nodes[i] for i in range(desiredComposition)]
                offspringGraph_with_loose_edges = self.mofutility.get_offspringGraph_with_loose_edges(outputNodes)
                offspringGraph = self.mofutility.restore_loose_edges(offspringGraph_with_loose_edges,
                                                                     algo = self.mofutility.createEdgesAlgo,cell=outputCell)
                if len(outputNodes) == desiredComposition:
                    return (offspringGraph, outputCell)
        raise RuntimeError("MOF_Heredity failed.")




class Topology_slab:
    def __init__(self,indices, depths, nodes):
        self.indices = np.asarray(indices, dtype=int)
        self.depths = np.asarray(depths, dtype=float)
        self.nodes = nodes

    @staticmethod
    def getSlabs(graph, inputCell, outputCell, axis, gaugesOfSlabs, transformation):
        nodes = graph.nodes.data()
        assert inputCell.getPBC() == outputCell.getPBC()
        if inputCell.dim == 1:
            inputAxis = inputCell.getCellVectorsPBC()
            inputAxis /= np.linalg.norm(inputAxis)
            outputAxis = outputCell.getCellVectorsPBC()
            outputAxis /= np.linalg.norm(outputAxis)
            assert np.allclose(inputAxis, outputAxis)
        elif inputCell.dim == 2:
            inputAxis = inputCell.getCellVectorsAntiPBC()
            inputAxis /= np.linalg.norm(inputAxis)
            outputAxis = outputCell.getCellVectorsAntiPBC()
            outputAxis /= np.linalg.norm(outputAxis)
            assert np.allclose(inputAxis, outputAxis)
        inputCell = transformation.transformCell(inputCell)
        slabs = tuple(([], [], []) for i in gaugesOfSlabs)
        coordinateBounds = np.cumsum(gaugesOfSlabs) / np.sum(gaugesOfSlabs)
        for i, node in enumerate(nodes):
            centerOfMassCoordinatesInitial = nodes[node]['ccoords']
            centerOfMassCoordinates = transformation.transformCoordinates(centerOfMassCoordinatesInitial)
            pbc = outputCell.getPBC()
            dimensionality = np.sum(pbc)
            if dimensionality == 1 and pbc[axis]:
                for fittedTransformation in outputCell.getFittedTransformations(centerOfMassCoordinates, inputCell):
                    coordinates = outputCell.cartesianToFractional(
                        fittedTransformation.transformCoordinates(centerOfMassCoordinates))
                    coordinate = coordinates[axis]
                    for j, upperBoundCoordinate in enumerate(coordinateBounds):
                        if coordinate <= upperBoundCoordinate:
                            lowerBoundCoordinate = 0 if j < 1 else coordinateBounds[j - 1]
                            indices, depths, mols = slabs[j]
                            indices.append(i)
                            depths.append(
                                np.min((upperBoundCoordinate - coordinate, coordinate - lowerBoundCoordinate)))
                            mols.append((fittedTransformation * transformation).transform(node))
                            break
            else:
                coordinates = inputCell.cartesianToFractional(centerOfMassCoordinates)
                coordinates = inputCell.getWrapedFractionalCoordinates(coordinates)
                inds = np.nonzero(outputCell.getAntiPBC())
                assert len(inds[0]) == 0
                coordinates[inds] = outputCell.cartesianToFractional(centerOfMassCoordinates)[inds]
                coordinate = coordinates[axis]
                for j, upperBoundCoordinate in enumerate(coordinateBounds):
                    if coordinate <= upperBoundCoordinate:
                        lowerBoundCoordinate = 0 if j < 1 else coordinateBounds[j - 1]
                        indices, depths, slab_nodes = slabs[j]
                        indices.append(i)
                        depths.append(np.min((upperBoundCoordinate - coordinate, coordinate - lowerBoundCoordinate)))
                        transVector = outputCell.fractionalToCartesian(coordinates) - \
                                      np.dot(transformation.rotMatrix, centerOfMassCoordinatesInitial)
                        finalTransformation = type(transformation).fromMatrix(transformation.rotMatrix, transVector)
                        move_node(node,graph,finalTransformation,outputCell)
                        slab_nodes.append(node)
                        break
        return (Topology_slab(indices, depths, nodes) for indices, depths, nodes in slabs)

    @staticmethod
    def getRandomSlabs(graph, inputCell, outputCell, axis, gaugesOfSlabs):
        slabs = Topology_slab.getSlabs(graph, inputCell, outputCell, axis, gaugesOfSlabs,
                                         transformation = inputCell.randomTransformation())
        return slabs


def move_node(node,graph,transformation,cell):
    node_ccoords = graph[node]['ccoords']
    node_ccoords = transformation.transformCoordinates(node_ccoords)
    graph[node]['fcoords'] = cell.cartesianToFractional(node_ccoords)