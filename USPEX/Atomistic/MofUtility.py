import networkx as nx
import copy
import numpy as np
import math
import os
import random
from .Operators.tobacco_3.tobacco import run_tobacco_serial, __file__ as tobacco_path
TOBACCO_DIR = os.path.dirname(tobacco_path)
class MofUtility:
    def __init__(self, nodes, linkers, createEdgesAlgo):
        self.nodes = nodes
        self.linkers = linkers
        self.createEdgesAlgo = createEdgesAlgo
        self.tobacco_path = TOBACCO_DIR

    @classmethod
    def get_random_template_path(cls):
        """puts in random cif from /template_database/"""
        templates_database_path = os.path.join(TOBACCO_DIR,'template_database')
        template = random.choice(os.listdir(templates_database_path))
        new_template_path = os.path.join(templates_database_path, template)
        return new_template_path
    @classmethod
    def get_random_templates(cls,N=100):
        templates_database_path = os.path.join(TOBACCO_DIR, 'template_database')
        templates = os.listdir(templates_database_path)
        random.shuffle(templates)
        templates_paths = [os.path.join(templates_database_path,t) for t in templates[0:N]]
        return templates_paths

    @classmethod
    def get_MOF_from_tobacco(cls,graph_template_paths,cellUtility,atomisticUtility,graph_for_mapping=None):
        if graph_for_mapping:
            node_buildingBlock_assignment = {node:graph_for_mapping.nodes[node] for node in graph_for_mapping.nodes}
            with open(os.path.join(TOBACCO_DIR, 'vertex_assignment.txt', 'w')) as settings:
                for node in node_buildingBlock_assignment:
                    node_symbol = ''.join([l for l in node.name if l.isalpha()])
                    settings.write(f'{node_symbol} {node_buildingBlock_assignment[node]}\n')
        else:
            settings = open(os.path.join(TOBACCO_DIR, 'vertex_assignment.txt'),'w')
            settings.close()
        if not isinstance(graph_template_paths,list):
            graph_template_paths = [graph_template_paths]
        sbu_atoms_coordinates, edges_coordinates, graph, scaled_cell_params = run_tobacco_serial(graph_template_paths)
        a, b, c, alpha, beta, gamma = scaled_cell_params
        _randomCell = cellUtility.getRandomCell(1, 1)
        cell = type(_randomCell).initFromCellParameters(pbc=(1, 1, 1), a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma)
        sbu_list = atomisticUtility.AtomicStructureRepresentation.fromTobacco(sbu_atoms_coordinates, cell)
        edges_list = atomisticUtility.AtomicStructureRepresentation.fromTobacco(edges_coordinates, cell)
        graph = cls.relabel_nodes(graph)
        MOF = {'atomistic.molecules': sbu_list + edges_list, 'atomistic.cell': cell, 'mofutility.topology': graph}
        return MOF

    @classmethod
    def move_node(cls,node,graph,transformation,cell):
        node_ccoords = graph[node]['ccoords']
        node_ccoords = transformation.transformCoordinates(node_ccoords)
        graph[node]['fcoords'] = cell.cartesianToFractional(node_ccoords)

    @classmethod
    def relabel_nodes(cls,graph):
        """Turns str-type nodes to Node class instance nodes. str-type nodes come from ToBaCCo"""
        mapping = {}
        for n in graph.nodes:
            mapping[n] = Node(n)
        nx.relabel_nodes(graph,mapping)
        return graph

    @classmethod
    def get_offspringGraph_with_loose_edges(cls, offspringNodes):
        """Adds edges to offspring nodes based on their previous connections in parent systems"""
        outputGraph = nx.MultiGraph()
        [outputGraph.add_node(n) for n in offspringNodes]
        for node, node_data in offspringNodes:
            parent_graph = node_data['parent_graph']
            edges_with_this_node = list(parent_graph.edges(node, data=True))
            for edge in edges_with_this_node:
                connected_node = edge[1]
                edge_data_dict = copy.copy(edge[2])
                if connected_node in offspringNodes and edge not in outputGraph.edges.data():
                    edge_data_dict['loose'] = False
                    outputGraph.add_edge(edge[0], edge[1], **edge_data_dict)
                elif connected_node not in offspringNodes:
                    edge_data_dict['loose'] = True
                    connected_node.is_loose_edge_end = True
                    outputGraph.add_edge(edge[0], edge[1], **edge_data_dict)
                else:
                    raise RuntimeError
        return outputGraph

    @classmethod
    def restore_loose_edges(cls, graph_with_loose_edges,algo,cell):
        edges = list(graph_with_loose_edges.edges.data())
        loose_edges = [edge for edge in edges if edge[2]['loose'] is True]
        for loose_edge in loose_edges:
            restored_edge = cls.get_edge_from_loose_edge(graph_with_loose_edges,loose_edge,algo,cell)
            graph_with_loose_edges.add_edge(restored_edge[0],restored_edge[1],**restored_edge[2])
        [graph_with_loose_edges.remove_edge(le) for le in loose_edges]
        graph = graph_with_loose_edges
        return graph

    @classmethod
    def get_edge_from_loose_edge(cls,graph,loose_edge,algo,cell):
        assert loose_edge[2]['loose'] is True
        assert loose_edge[0].is_loose_edge_end is True or loose_edge[1].is_loose_edge_end is True
        node = loose_edge[1] if loose_edge[0].is_loose_edge_end else loose_edge[0]
        new_neighbour,new_neighbour_label,new_neighbour_data = cls.get_new_neighbour(graph,loose_edge,cell,algo)
        new_edge_data = copy.copy(loose_edge[2])
        new_edge_data['loose'] = False
        new_edge_data['label'] = new_neighbour_label
        new_edge_data['length'] = np.linalg.norm(graph.nodes[node]['ccoords']-new_neighbour_data['ccoords'])
        new_edge_data['fcoords'] = (new_neighbour_data[['fcoords']]+graph.nodes[node]['fcoords'])/2
        new_edge_data['ccoords'] = (new_neighbour_data[['ccoords']]+graph.nodes[node]['ccoords'])/2
        new_edge_data['type'] = (node.symbol,new_neighbour.symbol)
        new_edge = (node,new_neighbour,new_edge_data)
        return new_edge

    @classmethod
    def get_new_neighbour(cls,graph,loose_edge,cell,algo='cone'):
        if algo == 'cone':
            node = loose_edge[1] if loose_edge[0].is_loose_edge_end else loose_edge[0]
            loose_edge_end = loose_edge[0] if loose_edge[0].is_loose_edge_end else loose_edge[1]
            loose_edge_vector = graph.nodes[loose_edge_end]['ccoords']-graph.nodes[node]['ccoords']
            cone_angle = cls.get_cone_angle(node, loose_edge_end, graph)
            nodes_supercell = cls.get_nodes_supercell(graph,cell=cell,node_to_exclude = node)
            angles = []
            distances = []
            for n_dict in nodes_supercell:
                n, n_label, n_data = n_dict.values()
                node_vector = n_data['ccoords'] - graph.nodes[node]['ccoords']
                distances.append(np.linalg.norm(node_vector))
                angle = cls.get_angle(loose_edge_vector,node_vector)
                angles.append(angle)
            nodes_in_cone = [nodes_supercell[i] for i in range(len(nodes_supercell)) if angles[i]<=cone_angle]
            nodes_in_cone_distances = [distances[i] for i in [nodes_supercell.index(n) for n in nodes_in_cone]]
            closest_node = nodes_in_cone[nodes_in_cone_distances.index(min(nodes_in_cone_distances))]
            new_neighbour = closest_node['node']
            new_neighbour_label = closest_node['label']
            new_neighbour_data = closest_node['node_data']
        else:
            new_neighbour, new_neighbour_label,new_neighbour_data = None, None, None
        return new_neighbour,new_neighbour_label, new_neighbour_data

    @classmethod
    def get_nodes_supercell(cls,graph,cell,node_to_exclude=None):
        # nodes_supercell = [{node:node,label,node_data_dict),...]
        # node_data_dict = {'ccoords':ccoords,}
        nodes_fcoords = [graph[n]['fcoords'] for n in graph.nodes]
        nodes_supercell = []
        trans_vectors = []
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                for k in [-1,0,1]:
                    trans_vectors.append(np.array([i,j,k]))
        for n in graph.nodes:
            for v in trans_vectors:
                image_node_data_dict = copy.copy(graph.nodes[n])
                image_fcoords = graph.nodes[n]['fcoords']+v
                image_node_data_dict['fcoords']=image_fcoords
                image_node_data_dict['ccoords']=cell.fractionalToCartesian(image_fcoords)
                image_node_label = v
                nodes_supercell.append({'node':n,'label':image_node_label,'node_data':image_node_data_dict})
        if node_to_exclude:
            for i, node_dict in enumerate(nodes_supercell):
                if node_dict['node']==node_to_exclude and np.isclose(node_dict['label'],np.array([0,0,0])):
                    node_to_exclude_index = i
                    break
            nodes_supercell.pop(node_to_exclude_index)
        return nodes_supercell

    @classmethod
    def get_cone_angle(cls,node,loose_edge_end,graph):
        connected_nodes = [n for n in graph.neighbours(node) if n.is_loose_edge_end is False]
        node_ccoords = graph[node]['ccoords']
        loose_edge_vector = graph[loose_edge_end]['ccoords'] - node_ccoords
        connected_edges_vectors = [graph[n]['ccoords']-node_ccoords for n in connected_nodes]
        return min([cls.get_angle(loose_edge_vector,v) for v in connected_edges_vectors])

    @classmethod
    def get_angle(cls,v1,v2):
        return math.acos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))

class Node:
    def __init__(self, name: str, label=[0,0,0], is_loose_edge_end=False):
        self.name = name
        self.symbol = ''.join(l for l in name.split() if l.isalpha())
        self.label = label
        self.is_loose_edge_end = is_loose_edge_end


