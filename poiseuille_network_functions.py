import numpy as np
import networkx as nx
from decimal import Decimal


def process_network(edges_in_network, nodes_in_network):
    """
    Function that runs all of the network preprocessing required to solve the poiseuille matrix and the haematocrit split
    Identifies bifurcations and straigh connections in the network, and returns relevant information pertaining to them
    :param edges_in_network: (edge_#, node_0, node_1) in every row, standard edges notation
    :param nodes_in_network: (node_#, x, y, z) in every row, standard nodes notation
    :return network_bifurcations: [edge_1_#, edge_2_#, edge_3_#, central_node] at every row returns each bifurcation in the network, with all three edges in the bifurcation and the central node to the bifurcation
    :return network_bifurcation_neighbour_nodes: [central node, neigh_node1, neigh_node2, neigh_node3] in every row for every bifurcation in network
    :return network_straights: [edge_1, edge_2, central_node] in every row, represents a straight
    :return network_straights_neighbour_nodes: [central_node, neighbour_node_1, neighbour_node_2] in every row
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

            network_bifurcations.append(bifurcation)
            network_bifurcation_neighbour_nodes.append(bifurcation_nodes)

    return network_bifurcations, network_bifurcation_neighbour_nodes, network_straights, network_straights_neighbour_nodes


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


def check_mass_balance(flowrates, haematocrits, straights, divergings, convergings, upper_bound=1.001, lower_bound=0.999):
    """
    Performs a mass balance at every degree 2 and 3 node on both blood flowrate and RBC flowrate to ensure mass is conserved
    :param flowrates: List of flowrates at all edges, indexed by edge number
    :param haematocrits: List of haematocrits at all edges, indexed by edge number
    :param straights: Degree 2 nodes with connected edges [edge1, edge2, central node]
    :param divergings: Diverging degree 3 nodes with connecting edges [Diverging edge, edge2, edge3, central node]
    :param convergings: Converging degree 3 nodes with connecting edges [Converging edge, edge2, edge3, central node]
    :param upper_bound: Error marging upper bound as a fraction
    :param lower_bound: Error margin lower bound as a fraction
    :return: Boolean, always True, but test will fail if one of the assertions fail
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

        assert upper_bound >= round(Decimal(q0) / Decimal(q1 + q2), 2) >= lower_bound, ("flowrate balance fail in converging ", converging, round( Decimal(q0) / Decimal(q1 + q2), 2 ), q0, q1, q2 )
        if h1 + h2 == 0.:
            assert h0 < 1e-5, h0
        else:
            assert upper_bound >= round(Decimal(h0 * q0) / Decimal(h1 * q1 + h2 * q2), 2) >= lower_bound, ("RBC balance fail in converging ", converging, round(Decimal(h0 * q0) / Decimal( h1 * q1 + h2 * q2), 2))

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


def check_bifurcations(divergent_check, convergent_check, q_check, p_check, edges_check):
    """
    A load of assert statements to check if my bifurcations are rightly classified
    :param divergent_check: list divergeant bifurcations (edge_1, edge_2, edge_3, central_node) in every row
    :param convergent_check:list convergeant bifurcations (edge_1, edge_2, edge_3, central_node) in every row
    :param q_check: list of edge flowrates with index edge_#
    :param p_check: list of node flowrates with index node_#
    :param edges_check: (edges_#, node_1, node_2) in every row, standard edges notation
    :return: Boolean, always True, but test will fail if one of the assertions fail
    """
    for bifurcation in divergent_check:
        edge_1, edge_2, edge_3, central_node = bifurcation[0], bifurcation[1], bifurcation[2], bifurcation[3]
        q_1, q_2, q_3 = q_check[edge_1], q_check[edge_2], q_check[edge_3]
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
        q_1, q_2, q_3 = q_check[edge_1], q_check[edge_2], q_check[edge_3]
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
    return True


def check_converged(q_results, error=0.1, print_option=False):
    """
    Checks whether the current iteration has converged with the previous iteration
    :param q_results: a nested list with each entry containing a list with the flowrate at every edge for each iteration
    :param error: percentage difference between previous and current iteration for convergence
    :return converged: Boolean for whether solution of current iteration has converged with previous iteration
    """
    converged = True
    if len(q_results) == 1:
        converged = False
    else:
        current_Q = q_results[-1]
        previous_Q = q_results[-2]
        for count, (current, previous) in enumerate(zip(current_Q, previous_Q)):
            if abs(current) == abs(previous):
                pass
            elif abs(current) - abs(previous) >= abs(current)*error/100.:
                if print_option:
                    print(100.*(abs(current) - abs(previous)) / abs(current), current, previous, count, abs(current - previous))
                converged = False
                break
    return converged


def check_straights(q, straights):
    """
    Checks that the straigh edges are conserving mass from a flowrate point of view
    :param q: a list containing the flowrate at every edge, indexed by edge
    :param straights: a nested list containing [edge 1 #, edge 2 #, central node #] at every entry, i.e. the two edges in a straight and the central node
    :return: Boolean for whether mass is conserved in every straight or not
    """
    for straight in straights:
        edge_1 = straight[0]
        edge_2 = straight[1]
        if abs(abs(q[edge_1]) - abs(q[edge_2])) > abs(q[edge_1])*0.01/100:
            return False
    return True


def check_converging(q, convergings):
    """
    Check that the converging bifurcations are conserving mass from a flowrate point of view
    :param q: a list containing the flowrate at every edge, indexed by edge
    :param convergings: a nested list containing [edge_1, edge_2, edge_3, central_node] where edge 2 and edge 3 converge into edge 1, at every entry
    :return: Boolean for whether mass is conserved in every converging or not
    """
    for converging in convergings:
        inlet_1 = converging[1]
        inlet_2 = converging[2]
        outlet = converging[0]
        q_in = abs(q[inlet_1]) + abs(q[inlet_2])
        q_out = abs(q[outlet])
        if q_in == q_out:
            return True
        elif q_in - q_out >= q_in*0.01/100:
            return False
    return True


def check_diverging(q, divergings):
    """
    Check that the diverging bifurcations are conserving mass
    :param q: a list containing the flowrate at every edge, indexed by edge
    :param divergings: a nested list contaning [edge_1, edge_2, edge_3, central_node] where edge 1 bifurcates into edge 2 and edge 3, at every entry
    :return: Boolean for whether mass is conserved in every diverging or not
    """
    for diverging in divergings:
        inlet = diverging[0]
        outlet_1 = diverging[1]
        outlet_2 = diverging[2]
        q_in = abs(q[inlet])
        q_out = abs(q[outlet_1]) + abs(q[outlet_2])
        if q_in == q_out:
            return True
        elif q_in - q_out >= q_in * 0.01 / 100:
            return False
    return True


def check_flow_to_all_junctions_is_zero(poiseuille_class):
    """
    This checks that according to the solution, the flow to every junction is 0 (q1 + q2 +q3=0) based on notation in Fry
    :param poiseuille_class: NetworkClass defined in poiseuille_class.py
    :return: no return, just fails if not
    """
    for count, bifurcation in enumerate(poiseuille_class.bifurcations):
        q1, q2, q3 = poiseuille_class.Q[bifurcation[0]], poiseuille_class.Q[bifurcation[1]], poiseuille_class.Q[bifurcation[2]]
        bif_neigh = poiseuille_class.bifurcation_neighbour_nodes[count]
        central = bif_neigh[0]
        neigh1, neigh2, neigh3 = bif_neigh[1], bif_neigh[2], bif_neigh[3]
        pressures = poiseuille_class.p
        p_central = pressures[central]
        p1, p2, p3 = pressures[neigh1], pressures[neigh2], pressures[neigh3]
        if p1 > p_central:
            l1 = 1.
        else:
            l1 = -1.
        if p2 > p_central:
            l2 = 1.
        else:
            l2 = -1.
        if p3 > p_central:
            l3 = 1.
        else:
            l3 = -1.
        q1, q2, q3 = l1*abs(q1), l2*abs(q2), l3*abs(q3)
        print(sum([q1, q2, q3]), q1, q2, q3)
        assert abs(sum([q1, q2, q3])) < 10e-22
    for count, straight in enumerate(poiseuille_class.straights):
        q1, q2 = poiseuille_class.Q[straight[0]], poiseuille_class.Q[straight[1]]
        bif_neigh = poiseuille_class.straight_neighbour_nodes[count]
        central = bif_neigh[0]
        neigh1, neigh2 = bif_neigh[1], bif_neigh[2]
        pressures = poiseuille_class.p
        p_central = pressures[central]
        p1, p2 = pressures[neigh1], pressures[neigh2]
        if p1 > p_central:
            l1 = 1.
        else:
            l1 = -1.
        if p2 > p_central:
            l2 = 1.
        else:
            l2 = -1.
        q1, q2 = l1*abs(q1), l2*abs(q2)
        print(sum([q1, q2]), q1, q2)
        assert abs(sum([q1, q2])) < 10e-22
    return


if __name__ == "__main__":
    pass
