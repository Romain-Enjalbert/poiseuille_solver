from itertools import combinations
import numpy as np
import networkx as nx
import time
from decimal import Decimal


def process_network(edges_in_network):
    """
    Function that runs all of the network preprocessing required to solve the poiseuille matrix and the haematocrit split
    :param edges_in_network: (edge_#, node_0, node_1) in every row, standard edges notation
    :return network_bifurcations: (edge_1_#, edge_2_#, edge_3_#, central_node) at every row returns each bifurcation in the network, with all three edges in the bifurcation and the central node to the bifurcation
    :return network_bifurcation_nodes: List containing all nodes that are central to a bifurcation
    :return network_bifurcation_neighbour_nodes: (central node, neigh_node1, neigh_node2, neigh_node3) in every row for every bifurcation in network
    :return network_straights: (edge_1, edge_2, central_node) in every row, represents a straight
    :return network_straights_neighbour_nodes: (central_node, neighbour_node_1, neighbour_node_2) in every row
    """
    edges_combinations_pairs, edges_combinations_triplets = combinations_2_3(edges_in_network)
    network_bifurcations = identify_bifurcations(edges_combinations_triplets)
    network_straights = identify_straights(edges_combinations_pairs, network_bifurcations)
    network_bifurcation_neighbour_nodes = neighbour_nodes(edges_in_network)
    network_straights_neighbour_nodes = straights_neigh_nodes(network_straights, edges_in_network)

    return network_bifurcations, network_bifurcation_neighbour_nodes, network_straights, network_straights_neighbour_nodes


def new_process_network(edges_in_network, nodes_in_network):
    """
    Function that runs all of the network preprocessing required to solve the poiseuille matrix and the haematocrit split, speedup compared to "process_network" above
    :param edges_in_network: (edge_#, node_0, node_1) in every row, standard edges notation
    :param nodes_in_network: (node_#, x, y) in every row, standard nodes notation
    :return network_bifurcations: (edge_1_#, edge_2_#, edge_3_#, central_node) at every row returns each bifurcation in the network, with all three edges in the bifurcation and the central node to the bifurcation
    :return network_bifurcation_nodes: List containing all nodes that are central to a bifurcation
    :return network_bifurcation_neighbour_nodes: (central node, neigh_node1, neigh_node2, neigh_node3) in every row for every bifurcation in network
    :return network_straights: (edge_1, edge_2, central_node) in every row, represents a straight
    :return network_straights_neighbour_nodes: (central_node, neighbour_node_1, neighbour_node_2) in every row
    """
    network_bifurcations, network_bifurcation_neighbour_nodes, network_straights, network_straights_neighbour_nodes = [], [], [], []

    G = nx.Graph()
    for node in nodes_in_network:
        G.add_node(node[0])
    for count, edge in enumerate(edges_in_network):
        G.add_edge(int(edge[1]), int(edge[2]), index=count)

    for node in G.nodes():
        if G.degree(node) == 2:
            central_node = int(node)
            neighbour_node_1 = int(list(G.edges(node))[0][1])
            neighbour_node_2 = int(list(G.edges(node))[1][1])
            edge_1 = G.get_edge_data(central_node, neighbour_node_1)["index"]
            edge_2 = G.get_edge_data(central_node, neighbour_node_2)["index"]
            if edge_1 < edge_2:
                straight = [edge_1, edge_2, central_node]
                straight_nodes = np.array([central_node, neighbour_node_1, neighbour_node_2])
            else:
                straight = [edge_2, edge_1, central_node]
                straight_nodes = np.array([central_node, neighbour_node_2, neighbour_node_1])

            # if neighbour_node_1 < neighbour_node_2:
            #     straight_nodes = np.array([central_node, neighbour_node_1, neighbour_node_2])
            # else:
            #     straight_nodes = np.array([central_node, neighbour_node_2, neighbour_node_1])

            network_straights.append(straight)
            network_straights_neighbour_nodes.append(straight_nodes)

        if G.degree(node) == 3:
            central_node = int(node)
            neighbour_node_1 = int(list(G.edges(node))[0][1])
            neighbour_node_2 = int(list(G.edges(node))[1][1])
            neighbour_node_3 = int(list(G.edges(node))[2][1])

            edge_1 = G.get_edge_data(central_node, neighbour_node_1)["index"]
            edge_2 = G.get_edge_data(central_node, neighbour_node_2)["index"]
            edge_3 = G.get_edge_data(central_node, neighbour_node_3)["index"]

            if edge_1 < edge_2 < edge_3:
                bifurcation = [edge_1, edge_2, edge_3, central_node]
                bifurcation_nodes = np.array([central_node, neighbour_node_1, neighbour_node_2, neighbour_node_3])
            elif edge_1 < edge_3 < edge_2:
                bifurcation = [edge_1, edge_3, edge_2, central_node]
                bifurcation_nodes = np.array([central_node, neighbour_node_1, neighbour_node_3, neighbour_node_2])
            elif edge_2 < edge_1 < edge_3:
                bifurcation = [edge_2, edge_1, edge_3, central_node]
                bifurcation_nodes = np.array([central_node, neighbour_node_2, neighbour_node_1, neighbour_node_3])
            elif edge_2 < edge_3 < edge_1:
                bifurcation = [edge_2, edge_3, edge_1, central_node]
                bifurcation_nodes = np.array([central_node, neighbour_node_2, neighbour_node_3, neighbour_node_1])
            elif edge_3 < edge_2 < edge_1:
                bifurcation = [edge_3, edge_2, edge_1, central_node]
                bifurcation_nodes = np.array([central_node, neighbour_node_3, neighbour_node_2, neighbour_node_1])
            elif edge_3 < edge_1 < edge_2:
                bifurcation = [edge_3, edge_1, edge_2, central_node]
                bifurcation_nodes = np.array([central_node, neighbour_node_3, neighbour_node_1, neighbour_node_2])

            # if neighbour_node_1 < neighbour_node_2 < neighbour_node_3:
            #     bifurcation_nodes = np.array([central_node, neighbour_node_1, neighbour_node_2, neighbour_node_3])
            # elif neighbour_node_1 < neighbour_node_3 < neighbour_node_2:
            #     bifurcation_nodes = np.array([central_node, neighbour_node_1, neighbour_node_3, neighbour_node_2])
            # elif neighbour_node_2 < neighbour_node_1 < neighbour_node_3:
            #     bifurcation_nodes = np.array([central_node, neighbour_node_2, neighbour_node_1, neighbour_node_3])
            # elif neighbour_node_2 < neighbour_node_3 < neighbour_node_1:
            #     bifurcation_nodes = np.array([central_node, neighbour_node_2, neighbour_node_3, neighbour_node_1])
            # elif neighbour_node_3 < neighbour_node_2 < neighbour_node_1:
            #     bifurcation_nodes = np.array([central_node, neighbour_node_3, neighbour_node_2, neighbour_node_1])
            # elif neighbour_node_3 < neighbour_node_1 < neighbour_node_2:
            #     bifurcation_nodes = np.array([central_node, neighbour_node_3, neighbour_node_1, neighbour_node_2])

            network_bifurcations.append(bifurcation)
            network_bifurcation_neighbour_nodes.append(bifurcation_nodes)

    return network_bifurcations, network_bifurcation_neighbour_nodes, network_straights, network_straights_neighbour_nodes


def combinations_2_3(edges):
    """
    Takes as input the edges from the network and returns a combination of every possible pair and triplet of edges
    :param edges: Takes as input the edges from the network and returns a combination of every possible pair and triplet of edges
    :returns: (edge_1 in pair, edge_2 in pair) in every row. Each edge has same format as above
    :returns: (edge_1 in triplet, edge_2 in triplet, edge_3 in triplet) in every row. Each edge has same format as above
    """
    edges_pairs = combinations(edges, 2)
    edges_triplets = combinations(edges, 3)
    return edges_pairs, edges_triplets


def quicker_identify_bifurcation(edges_combinations_3):
    bifurcations = []
    for triplet_of_edges in edges_combinations_3:
        edges_nonet = []
        for edges in triplet_of_edges:
            edges_nonet.append(edges[0])
            edges_nonet.append(edges[1])
            edges_nonet.append(edges[2])  # below line checks in each group of 3 edges if they have a node in common
        if edges_nonet[1] == edges_nonet[4] == edges_nonet[7]:  # this identifies the common node between all 3 edges the three edges and the central node
            bifurcations.append([edges_nonet[0], edges_nonet[3], edges_nonet[6], edges_nonet[1]])
        elif edges_nonet[1] == edges_nonet[5] == edges_nonet[7]:
            bifurcations.append([edges_nonet[0], edges_nonet[3], edges_nonet[6], edges_nonet[1]])
        elif edges_nonet[1] == edges_nonet[4] == edges_nonet[8]:
            bifurcations.append([edges_nonet[0], edges_nonet[3], edges_nonet[6], edges_nonet[1]])
        elif edges_nonet[1] == edges_nonet[5] == edges_nonet[8]:
            bifurcations.append([edges_nonet[0], edges_nonet[3], edges_nonet[6], edges_nonet[1]])
        elif edges_nonet[2] == edges_nonet[4] == edges_nonet[7]:
            bifurcations.append([edges_nonet[0], edges_nonet[3], edges_nonet[6], edges_nonet[2]])
        elif edges_nonet[2] == edges_nonet[4] == edges_nonet[8]:
            bifurcations.append([edges_nonet[0], edges_nonet[3], edges_nonet[6], edges_nonet[2]])
        elif edges_nonet[2] == edges_nonet[5] == edges_nonet[7]:
            bifurcations.append([edges_nonet[0], edges_nonet[3], edges_nonet[6], edges_nonet[2]])
        elif edges_nonet[2] == edges_nonet[5] == edges_nonet[8]:
            bifurcations.append([edges_nonet[0], edges_nonet[3], edges_nonet[6], edges_nonet[2]])
    return bifurcations


def identify_bifurcations(edges_combinations_3):
    """
    Function to generate all bifurcations in the network
    :param edges_combinations_3: (edge_1, edge_2, edge_3) in every row. Edges have standard edge notation (edge_#, node_0, node_1). All possible combinations of triplets in network are in the list
    :return: (edge_1_#, edge_2_#, edge_3_#, central_node) at every row returns each bifurcation in the network, with all three edges in the bifurcation and the central node to the bifurcation. No order to edges
    """
    bifurcations = []
    for triplet_of_edges in edges_combinations_3:
        bifurcation = [] # edge 0, edge 1 and egde 2 of a bifurcation, last entry is common node to all three
        edges_nonet = []
        for edges in triplet_of_edges:
            edges_nonet.append(edges[0])
            edges_nonet.append(edges[1])
            edges_nonet.append(edges[2]) #below line checks in each group of 3 edges if they have a node in common
        if edges_nonet[1] == edges_nonet[4] == edges_nonet[7] or edges_nonet[1] == edges_nonet[5] == edges_nonet[7] or edges_nonet[1] == edges_nonet[4] == edges_nonet[8] or edges_nonet[1] == edges_nonet[5] == edges_nonet[8] or edges_nonet[2] == edges_nonet[4] == edges_nonet[7] or edges_nonet[2] == edges_nonet[4] == edges_nonet[8] or edges_nonet[2] == edges_nonet[5] == edges_nonet[7] or edges_nonet[2] == edges_nonet[5] == edges_nonet[8]:
            bifurcation.append(edges_nonet[0]) #here if there is a node in common between all three edges the number of the edges are added to a list
            bifurcation.append(edges_nonet[3])
            bifurcation.append(edges_nonet[6])
            if edges_nonet[1] == edges_nonet[4] == edges_nonet[7]: #this identifies the common node between all 3 edges and appends it
                bifurcation.append(edges_nonet[1])
            elif edges_nonet[1] == edges_nonet[5] == edges_nonet[7]:
                bifurcation.append(edges_nonet[1])
            elif edges_nonet[1] == edges_nonet[4] == edges_nonet[8]:
                bifurcation.append(edges_nonet[1])
            elif edges_nonet[1] == edges_nonet[5] == edges_nonet[8]:
                bifurcation.append(edges_nonet[1])
            elif edges_nonet[2] == edges_nonet[4] == edges_nonet[7]:
                bifurcation.append(edges_nonet[2])
            elif edges_nonet[2] == edges_nonet[4] == edges_nonet[8]:
                bifurcation.append(edges_nonet[2])
            elif edges_nonet[2] == edges_nonet[5] == edges_nonet[7]:
                bifurcation.append(edges_nonet[2])
            elif edges_nonet[2] == edges_nonet[5] == edges_nonet[8]:
                bifurcation.append(edges_nonet[2])
            else:
                break
            bifurcations.append(bifurcation)
    return bifurcations


def quicker_identify_straights(edges_combinations_2, bifurcations):
    central_bifurcation_nodes = []
    for bifurcation in bifurcations:
        central_bifurcation_nodes.append(bifurcation[3])
    straights_return = []
    for edges_pair in edges_combinations_2:
        edges_sextet = []
        for edge in edges_pair:
            edges_sextet.append(edge[0])
            edges_sextet.append(edge[1])
            edges_sextet.append(edge[2])
        if edges_sextet[1] == edges_sextet[4] and edges_sextet[1] not in central_bifurcation_nodes:
            straights_return.append([edges_sextet[0], edges_sextet[3], edges_sextet[1]])
        elif edges_sextet[1] == edges_sextet[5] and edges_sextet[1] not in central_bifurcation_nodes:
            straights_return.append([edges_sextet[0], edges_sextet[3], edges_sextet[1]])
        elif edges_sextet[2] == edges_sextet[4] and edges_sextet[2] not in central_bifurcation_nodes:
            straights_return.append([edges_sextet[0], edges_sextet[3], edges_sextet[2]])
        elif edges_sextet[2] == edges_sextet[5] and edges_sextet[2] not in central_bifurcation_nodes:
            straights_return.append([edges_sextet[0], edges_sextet[3], edges_sextet[2]])
    return straights_return


def identify_straights(edges_combinations_2, bifurcations):
    """
    Function to generate all straights in the network
    :param edges_combinations_2: (edge_1, edge_2) in every row. Edges have standard edge notation (edge_#, node_0, node_1). All possible combinations of edge pairs in network are in the list
    :param bifurcations: (edge_1_#, edge_2_#, edge_3_#, central_node) at every row returns each bifurcation in the network, with all three edges in the bifurcation and the central node to the bifurcation. No order to edges
    :return: (edge_1, edge_2, central_node) in every row, represents a straight
    """
    central_bifurcation_nodes = []
    for bifurcation in bifurcations:
        central_bifurcation_nodes.append(bifurcation[3])
    straights_return = []
    for edges_pair in edges_combinations_2:
        edges_sextet = []
        straight = []
        for edge in edges_pair:
            edges_sextet.append(edge[0])
            edges_sextet.append(edge[1])
            edges_sextet.append(edge[2])
        if edges_sextet[1] == edges_sextet[4] or edges_sextet[1] == edges_sextet[5] or edges_sextet[2] == edges_sextet[4] or edges_sextet[2] == edges_sextet[5]:
            straight.append(edges_sextet[0])
            straight.append(edges_sextet[3])
            if edges_sextet[1] == edges_sextet[4]:
                straight.append(edges_sextet[1])
            elif edges_sextet[1] == edges_sextet[5]:
                straight.append(edges_sextet[1])
            elif edges_sextet[2] == edges_sextet[4]:
                straight.append(edges_sextet[2])
            elif edges_sextet[2] == edges_sextet[5]:
                straight.append(edges_sextet[2])
            else:
                break
            if straight[2] not in central_bifurcation_nodes:
                straights_return.append(straight)
    return straights_return


def neighbour_nodes(edges):
    """
    Function to identify central node of a bifurcation and all three neighbouring nodes
    :param edges: (edge_#, node_0, node_1) in every row, standard edges notation
    :return: (central node, neigh_node1, neigh_node2, neigh_node3) in every row for every bifurcation in network
    """
    node_pairs = []    
    for edge in edges:
            edge_nodes = (edge[1], edge[2])
            node_pairs.append(edge_nodes)
    nodes_combin = combinations (node_pairs, 3)
    neighbours = []
    for nodes_triplet in nodes_combin:
        node_triplets = []
        for node in nodes_triplet: #puts these three pairs in a list of 6
            node_triplets.append(node[0])
            node_triplets.append(node[1])
        if node_triplets[0] == node_triplets[2] == node_triplets[4]: #this and below checks for 3 common nodes, if true, appends that node and other 3
            neighbours.extend((node_triplets[0],node_triplets[1],node_triplets[3],node_triplets[5]))
        elif node_triplets[0]==node_triplets[2]==node_triplets[5]:
            neighbours.extend((node_triplets[0],node_triplets[1],node_triplets[3],node_triplets[4]))
        elif node_triplets[0]==node_triplets[3]==node_triplets[4]:
            neighbours.extend((node_triplets[0],node_triplets[1],node_triplets[2],node_triplets[5]))
        elif node_triplets[0]==node_triplets[3]==node_triplets[5]:
            neighbours.extend((node_triplets[0],node_triplets[1],node_triplets[2],node_triplets[4]))      
        elif node_triplets[1]==node_triplets[2]==node_triplets[4]:
            neighbours.extend((node_triplets[1],node_triplets[0],node_triplets[3],node_triplets[5]))
        elif node_triplets[1]==node_triplets[2]==node_triplets[5]:
            neighbours.extend((node_triplets[1],node_triplets[0],node_triplets[3],node_triplets[4]))
        elif node_triplets[1]==node_triplets[3]==node_triplets[4]:
            neighbours.extend((node_triplets[1],node_triplets[0],node_triplets[2],node_triplets[5]))
        elif node_triplets[1]==node_triplets[3]==node_triplets[5]:
            neighbours.extend((node_triplets[1],node_triplets[0],node_triplets[2],node_triplets[4]))
    neighbours = np.array(neighbours)
    neighbours = np.reshape(neighbours, (int((len(neighbours)/4)), 4))
    return neighbours


def straights_neigh_nodes(straights, edges):
    """
    Function to get nodes that are in straights
    :param straights: (edge_1, ege_2, central_node_in_straight) at every row
    :param edges: (edge_#, node_1, node_2) at every row
    :return: (central_node, neighbour_node_1, neighbour_node_2) in every row
    """
    straight_nodes = []
    for straight in straights:
        central_node = straight[2]
        straight_nodes.append(central_node) # appends the central node of a straight
        edge_1, edge_2 = edges[straight[0]], edges[straight[1]]
        edge_1_node_1, edge_1_node_2 = edge_1[1], edge_1[2]
        edge_2_node_1, edge_2_node_2 = edge_2[1], edge_2[2]
        if edge_1_node_1 == central_node: #calls on the edges in the straights list, then appends the nodes that are not the central one (in straights list)
            straight_nodes.append(edge_1_node_2)
        elif edge_1_node_2 == central_node:
            straight_nodes.append(edge_1_node_1)
        else:
            assert 0
        if edge_2_node_1 == central_node:
            straight_nodes.append(edge_2_node_2)
        elif edge_2_node_2 == central_node:
            straight_nodes.append(edge_2_node_1)
        else:
            assert 0
    straight_nodes = np.array(straight_nodes)
    straight_nodes = np.reshape(straight_nodes, (int((len(straight_nodes)/3)), 3))
    return straight_nodes


def new_check_mass_balance(flowrates, haematocrits, straights, divergings, convergings, upper_bound=1.01, lower_bound=0.99):
    """

    :param flowrates:
    :param haematocrits:
    :param straights:
    :param divergings:
    :param convergings:
    :param upper_bound:
    :param lower_bound:
    :return:
    """

    print("start")
    # Ceck straights mass balance
    for straight in straights:
        edge_0, edge_1 = straight[0], straight[1]
        q0, q1 = abs(flowrates[edge_0]), abs(flowrates[edge_1])
        h0, h1 = haematocrits[edge_0], haematocrits[edge_1]

        # if h0 < 1e-4 and h1 < 1e-5:
        #     pass
        # else:
        #     assert 0

        if upper_bound >= round(Decimal(q0)/Decimal(q1), 2) >= lower_bound:
            pass
        else:
            print(("flowrate balance fail in straight ", straight, round(Decimal(q0)/Decimal(q1), 2), q0, q1))
            # assert 0

    print("Pre converging")
    for converging in convergings:
        edge_0, edge_1, edge_2 = converging[0], converging[1], converging[2]
        q0, q1, q2 = abs(flowrates[edge_0]), abs(flowrates[edge_1]), abs(flowrates[edge_2])
        h0, h1, h2 = haematocrits[edge_0], haematocrits[edge_1], haematocrits[edge_2]
        print("loops")
        if upper_bound >= round( Decimal(q0)/ Decimal(q1 + q2) , 2) >= lower_bound:
            pass
        else:
            print("flowrate balance fail in converging ", converging, round(Decimal(q0) / Decimal(q1 + q2), 2), q0, q1, q2)

        if upper_bound >= round(Decimal(h0 * q0) / Decimal(h1 * q1 + h2 * q2), 2) >= lower_bound:
            pass
        else:
            print("RBC balance fail in converging ", converging, round(Decimal(h0 * q0) / Decimal(h1 * q1 + h2 * q2), 2))

    print("Pre diverging")
    for diverging in divergings:
        edge_0, edge_1, edge_2 = diverging[0], diverging[1], diverging[2]
        q0, q1, q2 = abs(flowrates[edge_0]), abs(flowrates[edge_1]), abs(flowrates[edge_2])
        h0, h1, h2 = haematocrits[edge_0], haematocrits[edge_1], haematocrits[edge_2]

        if upper_bound >= round(q0/(q1 + q2)) >= lower_bound:
            pass
        else:
            print("flowrate balance fail in diverging ", diverging, round(q0 + q1 / (q2)))

        if upper_bound >= round(Decimal(h0 * q0) / Decimal(h1 * q1 + h2 * q2), 2) >= lower_bound:
            pass
        else:
            print("RBC balance fail in diverging ", diverging, round(Decimal(h0 * q0) / Decimal(h1 * q1 + h2 * q2), 2) )

    # check converging bifurcations mass balance
    # Check
    print("what anout here ?")
    return

def check_mass_balance(flowrates, haematocrits, straights, divergings, convergings, upper_bound=1.01, lower_bound=0.99):
    """
    Performs a mass balance at every degree 2 and 3 node on both blood flowrate and RBC flowrate to ensure mass is conserved
    :param flowrates: List of flowrates at all edges, indexed by edge number
    :param haematocrits: List of haematocrits at all edges, indexed by edge number
    :param straights: Degree 2 nodes with connected edges [edge1, edge2, central node]
    :param divergings: Diverging degree 3 nodes with connecting edges [Diverging edge, edge2, edge3, central node]
    :param convergings: Converging degree 3 nodes with connecting edges [Converging edge, edge2, edge3, central node]
    :param upper_bound: Error marging upper bound as a fraction
    :param lower_bound: Error margin lower bound as a fraction
    :return: Return true, but test will fail if one of the assertions fail
    """
    for straight in straights:
        edge_0, edge_1 = straight[0], straight[1]
        q0, q1 = abs(flowrates[edge_0]), abs(flowrates[edge_1])
        h0, h1 = haematocrits[edge_0], haematocrits[edge_1]

        assert upper_bound >= round(Decimal(q0)/Decimal(q1), 2) >= lower_bound, ("flowrate balance fail in straight ", straight, round(Decimal(q0)/Decimal(q1), 2), q0, q1)

        if h0 < 1e-5:
            assert h1 < 1e-5, (h0, h1)
        else:
            assert upper_bound >= round(Decimal(q0 * h0) / Decimal(q1 * h1), 2) >= lower_bound, ("RBC balance fail in straight ", straight, round(Decimal(q0 * h0) / Decimal(q1 * h1), 2) )

    for converging in convergings:
        edge_0, edge_1, edge_2 = converging[0], converging[1], converging[2]
        q0, q1, q2 = abs(flowrates[edge_0]), abs(flowrates[edge_1]), abs(flowrates[edge_2])
        h0, h1, h2 = haematocrits[edge_0], haematocrits[edge_1], haematocrits[edge_2]

        assert upper_bound >= round( Decimal(q0)/ Decimal(q1 + q2) , 2) >= lower_bound, ("flowrate balance fail in converging ", converging, round( Decimal(q0) / Decimal(q1 + q2), 2 ), q0, q1, q2 )
        if h1 + h2 == 0.:
            assert h0 < 1e-5, (h0)
        else:
            assert upper_bound >= round(Decimal(h0 * q0) / Decimal( h1 * q1 + h2 * q2), 2) >= lower_bound, ("RBC balance fail in converging ", converging, round(Decimal(h0 * q0) / Decimal( h1 * q1 + h2 * q2), 2))

    for diverging in divergings:
        edge_0, edge_1, edge_2 = diverging[0], diverging[1], diverging[2]
        q0, q1, q2 = abs(flowrates[edge_0]), abs(flowrates[edge_1]), abs(flowrates[edge_2])
        h0, h1, h2 = haematocrits[edge_0], haematocrits[edge_1], haematocrits[edge_2]

        assert upper_bound >= round(q0/(q1 + q2)) >= lower_bound, ("flowrate balance fail in diverging ", diverging, round(q0 + q1/(q2)))

        if h0 == 0.:
            assert h1 == 0.
            assert h2 == 0.
        else:
            assert upper_bound >= round(Decimal(h0 * q0) / Decimal(h1 * q1 + h2 * q2), 2) >= lower_bound, ("RBC balance fail in diverging ", diverging, round(Decimal(h0 * q0) / Decimal(h1 * q1 + h2 * q2), 2) )
    print("Mass balance passed")
    return True


def bifurcation_types(Q, bifurcations, edges, p):
    """
    Function to identify if bifurcations converge (simply mass balance) or diverge (haematocrit split rule to be applied)
    :param Q: Flowrate matrix with same index as edge number
    :param bifurcations: list of bifurcation in network (edge_1, edge_2, edge_3, central_node) at every row
    :param edges: (edge_#, node_1, node_2) at every row, Q[edge_#] gives flowrate for that edge
    :param p: pressure matrix, p[node_number] gives pressure for that node
    :return convergeance_return: (edge_1, edge_2, edge_3, central_node) where edge_2 and edge_3 converge into edge_1
    :return divergeance_return: (edge_1, edge_2, edge_3, central_node) where edge_1 bifurcates into edge_2 and edge_3
    """
    convergent_return, divergent_return = [], []

    for bifurcation in bifurcations: # (edge_1, edge_2, edge_3, central node)
        edge_0, edge_1, edge_2, central_node = bifurcation[0], bifurcation[1], bifurcation[2], bifurcation[3] # Test bifurcation[:] to unpack
        q_0, q_1, q_2 = abs(Q[edge_0]), abs(Q[edge_1]), abs(Q[edge_2])

        if q_0 == 0. and q_1 == 0. and q_2 == 0.:
            continue
        else:
            pass

        if q_0 > q_1 and q_0 > q_2:
            pass
        elif q_1 > q_0 and q_1 > q_2:
            temp = edge_0; edge_0 = edge_1; edge_1 = temp
        elif q_2 > q_0 and q_2 > q_1:
            temp = edge_0; edge_0 = edge_2; edge_2 = temp
        else:
            assert 0, (q_0, q_1, q_2, bifurcation)

        assert abs(Q[edge_0]) > abs(Q[edge_1])
        assert abs(Q[edge_0]) > abs(Q[edge_2])
        node_1, node_2 = edges[edge_0][1], edges[edge_0][2]
        if node_1 == central_node:
            different_node = node_2
        elif node_2 == central_node:
            different_node = node_1
        else:
            assert 0
        if p[different_node] > p[central_node]:
            divergent_return.append([edge_0, edge_1, edge_2, central_node])
        elif p[different_node] < p[central_node]:
            convergent_return.append([edge_0, edge_1, edge_2, central_node])
        elif p[different_node] == p[central_node]:
            continue # This is to protect in case the q == 0 checks at the begining don't work as the numerics don't quite give 0, it does happen
        else:
            assert 0

    return convergent_return, divergent_return


def bifurcation_check(divergent_check, convergent_check, Q_check, p_check, edges_check):
    """
    A load of assert statements to check if my bifurcations are rightly classified
    :param divergeance_check: list divergeant bifurcations (edge_1, edge_2, edge_3, central_node) in every row
    :param convergeance_check:list convergeant bifurcations (edge_1, edge_2, edge_3, central_node) in every row
    :param Q_check: list of edge flowrates with index edge_#
    :param p_check: list of node flowrates with index node_#
    :param edges_check: (edges_#, node_1, node_2) in every row, standard edges notation
    :return: no return, however it asserts an error if there is a problem
    """
    for bifurcation in divergent_check:
        edge_1, edge_2, edge_3, central_node = bifurcation[0], bifurcation[1], bifurcation[2], bifurcation[3]
        q_1, q_2, q_3 = Q_check[edge_1], Q_check[edge_2], Q_check[edge_3]
        assert abs(q_1) > abs(q_2)
        assert abs(q_1) > abs(q_3)
        node_1, node_2 = edges_check[edge_1][1], edges_check[edge_1][2]
        if node_1 == central_node:
            max_node = node_2
        elif node_2 == central_node:
            max_node = node_1
        else:
            assert 0
        node_3, node_4, node_5, node_6 = edges_check[edge_2][1], edges_check[edge_2][2], edges_check[edge_3][1], edges_check[edge_3][2]
        assert p_check[max_node] >= p_check[node_1]
        assert p_check[max_node] >= p_check[node_2]
        assert p_check[max_node] >= p_check[node_3]
        assert p_check[max_node] >= p_check[node_4], (p_check[max_node], p_check[node_4])
        assert p_check[max_node] >= p_check[node_5]
        assert p_check[max_node] >= p_check[node_6]

    for bifurcation in convergent_check:
        edge_1, edge_2, edge_3, central_node = bifurcation[0], bifurcation[1], bifurcation[2], bifurcation[3]
        q_1, q_2, q_3 = Q_check[edge_1], Q_check[edge_2], Q_check[edge_3]
        assert abs(q_1) > abs(q_2)
        assert abs(q_1) > abs(q_3)
        node_1, node_2 = edges_check[edge_1][1], edges_check[edge_1][2]
        if node_1 == central_node:
            min_node = node_2
        elif node_2 == central_node:
            min_node = node_1
        else:
            assert 0
        node_3, node_4, node_5, node_6 = edges_check[edge_2][1], edges_check[edge_2][2], edges_check[edge_3][1], edges_check[edge_3][2]
        assert p_check[min_node] <= p_check[node_1]
        assert p_check[min_node] <= p_check[node_2]
        assert p_check[min_node] <= p_check[node_3]
        assert p_check[min_node] <= p_check[node_4]
        assert p_check[min_node] <= p_check[node_5]
        assert p_check[min_node] <= p_check[node_6]
    return


if __name__ == "__main__":
    pass