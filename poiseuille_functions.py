import math
import numpy as np
import scipy.linalg
from scipy.sparse.linalg import spsolve
from scipy.sparse import coo_matrix
from scipy.linalg import solve as solve_matrix


def exp(x):
    return math.exp(x)


def logit(x):
    return math.log(x/(1.-x))


def solve_conductance(NetworkClass):
    """
    Generates conductance values based on Poiseuille equation and viscosity empirical relations from Secomb 2017 review
    :param NetworkClass: NetworkClass defined in poiseuille_class.py
    :return conductance_list: a list containing the conductance at every edge, indexed by edge
    :return viscosity_list: a list containing viscosity at every edge, indexed by edge
    """
    plasma_viscosity = 1.2*10**-3  # value from paper, Secomb 2017 (Pa s)
    conductance_list = []
    viscosity_list = []
    for edge in NetworkClass.edges:
        edge_number = edge[0]
        miu45 = 220.*exp(-1.3*NetworkClass.D[edge_number])+3.2-2.44*exp(-0.06*NetworkClass.D[edge_number]**0.645)
        C = (0.8 + exp(-0.075*NetworkClass.D[edge_number])) * (-1. + 1./(1.+10.**(-11)*NetworkClass.D[edge_number]**(12)))+1./(1.+10.**(-11)*NetworkClass.D[edge_number]**12)
        miurel = 1.+(miu45-1.)*((1.-NetworkClass.H[edge_number])**C-1.)/((1.-0.45)**C-1.)
        if np.isnan(miurel):
            print(miurel, miu45, NetworkClass.H[edge_number], C)
            assert 0
        miu = miurel*plasma_viscosity
        conductance = math.pi*NetworkClass.radius[edge_number]**4/(8.*NetworkClass.length[edge_number]*miu)
        conductance_list.append(conductance)
        viscosity_list.append(miu)
    return conductance_list, viscosity_list


def generate_solve_poiseuille_matrix(NetworkClass, conductance, sparse=True):
    """
    Generates a zero matrix of number_nodes x number_nodes and populates it with the conductance and known pressure (BC), converts it to a sparse matrix, and solves for pressure
    :param NetworkClass: NetworkClass defined in poiseuille_class.py
    :param conductance: a list containing the conductance at every edge, indexed by edge, generated through solve_conductance
    :return pressure_matrix: List of pressure values indexed by node number
    """
    if sparse:
        row = []
        col = []
        data = []
        for bifurcation, bifurcation_neighbours in zip(NetworkClass.bifurcations, NetworkClass.bifurcation_neighbour_nodes):
            row.append(bifurcation[3]); col.append(bifurcation[3]); data.append(-(conductance[bifurcation[0]] + conductance[bifurcation[1]] + conductance[bifurcation[2]]))
            row.append(bifurcation_neighbours[0]); col.append(bifurcation_neighbours[1]); data.append(conductance[bifurcation[0]])
            row.append(bifurcation_neighbours[0]); col.append(bifurcation_neighbours[2]); data.append(conductance[bifurcation[1]])
            row.append(bifurcation_neighbours[0]); col.append(bifurcation_neighbours[3]); data.append(conductance[bifurcation[2]])
        for straight, straight_neighbours in zip(NetworkClass.straights, NetworkClass.straight_neighbour_nodes):
            row.append(straight[2]); col.append(straight[2]); data.append(-(conductance[straight[0]] + conductance[straight[1]]))
            row.append(straight_neighbours[0]); col.append(straight_neighbours[1]); data.append(conductance[straight[0]])
            row.append(straight_neighbours[0]); col.append(straight_neighbours[2]); data.append(conductance[straight[1]])
        for nodes_iolets in NetworkClass.iolets: #This imposes the BCs in the matrix so that it doesn't change BCs to an unknown to be solved should it be linked to a straight/bifurcation
            row.append(nodes_iolets[0]), col.append(nodes_iolets[0]), data.append(1)

        matrix_size = len(NetworkClass.nodes)
        sparse_matrix = coo_matrix((data, (row, col)), shape=(matrix_size, matrix_size)).tocsr() # this converts the matrix to another format scipy can solve, reason for ussing coo is that its easy for me...
        pressure_matrix = spsolve(sparse_matrix, NetworkClass.p_0)
    else:
        """The Fry formulation below leads to identical results to above, implementation is for reference for the Fry notation and to check how it works"""
        """Above just bypasses several steps by pre-processing the network and getting the neighbour nodes, which yields the signs for M,L and not having to loop twice through all nodes"""
        """Sparse matrix also makes the solver much quicker on much larger networks"""
        matrix_K = np.zeros(shape=(len(NetworkClass.nodes), len(NetworkClass.nodes)))
        for bifurcation_i in NetworkClass.nodes:  # Fry, for i in N
            for bifurcation_k in NetworkClass.nodes:  # Fry, for k in N
                for edge_j in NetworkClass.edges:  # Fry, for j in S
                    edge_index = edge_j[0]
                    node_i = bifurcation_i[0]
                    node_k = bifurcation_k[0]
                    Mjk = getMjk(node_k, edge_j) * conductance[edge_index]
                    Lij = getLij(node_i, edge_j)
                    matrix_K[node_i, node_k] += Mjk * Lij
        for nodes_iolets in NetworkClass.iolets:  # this sets the boundary conditions with a 1 in the diagonal, so the pressure is known, the pressure is already in the NetworkClass.p0
            matrix_K[nodes_iolets[0], nodes_iolets[0]] = 1
        pressure_matrix = solve_matrix(matrix_K, NetworkClass.p_0)
    return pressure_matrix


def getMjk(node_k, segment_j):
    start_node_segment = segment_j[1]
    end_node_segment = segment_j[2]
    if node_k == start_node_segment:
        M_jk = 1.
    elif node_k == end_node_segment:
        M_jk = -1.
    else:
        M_jk = 0.
    return M_jk


def getLij(node_i, segment_j):
    start_node_segment = segment_j[1]
    end_node_segment = segment_j[2]
    if node_i == start_node_segment:
        L_ij = -1
    elif node_i == end_node_segment:
        L_ij = 1
    else:
        L_ij = 0.
    return L_ij


def solve_flow(pressure_matrix, conductance, edges_list):
    """
    Use available data to solve for flowrate through Poiseuille's equation
    :param pressure_matrix: Matrix containing pressure values indexed by node_number, generated through generate_solve_poiseuille_matrix
    :param conductance: a list containing the conductance at every edge, indexed by edge, generated through solve_conductance
    :param edges_list: a nested list containing [edge #, node 1, node 2] at every entry
    :return flow_list: a list containing the flowrate at every edge, indexed by edge
    """
    flow_list = []
    for edge in edges_list:
        flow = conductance[edge[0]] * (pressure_matrix[edge[1]] - pressure_matrix[edge[2]])
        flow_list.append(flow)
    return flow_list


def solve_haematocrit(flowrates, NetworkClass, diverging_bifurcations, converging_bifurcations, p_matrix):
    """
    Implements a mass balance for converging and straights. For diverging bifurcations,
    calculates child branch haematocrit according to Pries solver, see Secomb 2017 blood flow in microcirculation
    :param flowrates: a list containing the flowrate at every edge, indexed by edge, generated through solve_flow
    :param NetworkClass: NetworkClass defined in poiseuille_class.py
    :param diverging_bifurcations: a list containing [edge 1, edge 2, edge 3, central_node] in diverging bifurcations, edge 1 bifurcates to 2 and 3
    :param converging_bifurcations: a list contaning [edge_1, edge_2, edge_3, central_node] in converging bifurcations edge 2 and 3 converge into 1
    :param p_matrix: a list of pressure values indexed by node number, generated through generate_solve_poiseuille_matrix
    :return h_list: a list of haematocrit at every edge, indexed by edge number
    """

    h_list = NetworkClass.H
    straight_bifurcations = NetworkClass.straights
    edges = NetworkClass.edges
    diameters = NetworkClass.D

    descending_pressure_list = sorted(p_matrix, reverse=True)
    pressure_descending_index = [np.where(p_matrix == pressure)[0][0] for pressure in descending_pressure_list]
    central_node_straight = [edges_node[2] for edges_node in straight_bifurcations]
    central_node_diverging = [edges_node[3] for edges_node in diverging_bifurcations]
    central_node_converging = [edges_node[3] for edges_node in converging_bifurcations]
    for index in pressure_descending_index:
        if index in central_node_straight:
            straight = straight_bifurcations[central_node_straight.index(index)]
            if abs(flowrates[straight[0]]) and abs(flowrates[straight[1]]) < 10 ** (-50):
                h_list[straight[0]] = 0
                h_list[straight[1]] = 0

            if p_matrix[edges[straight[0], 1]] >= p_matrix[edges[straight[0], 2]] and p_matrix[edges[straight[0], 1]] >= p_matrix[edges[straight[1], 1]] and p_matrix[edges[straight[0], 1]] >= p_matrix[edges[straight[1], 2]]:
                h_list[straight[1]] = h_list[straight[0]]
            elif p_matrix[edges[straight[0], 2]] >= p_matrix[edges[straight[0], 1]] and p_matrix[edges[straight[0], 2]] >= p_matrix[edges[straight[1], 1]] and p_matrix[edges[straight[0], 2]] >= p_matrix[edges[straight[1], 2]]:
                h_list[straight[1]] = h_list[straight[0]]
            else:
                h_list[straight[0]] = h_list[straight[1]]
        elif index in central_node_diverging:
            divergent = diverging_bifurcations[central_node_diverging.index(index)]
            if abs(flowrates[divergent[0]]) < 10. ** (-50):
                h_list[divergent[1]] = 0
                h_list[divergent[2]] = 0
            else:
                h_list[divergent[1]], h_list[divergent[2]] = pries_solver(flowrates[divergent[0]], flowrates[divergent[1]], flowrates[divergent[2]], h_list[divergent[0]], diameters[divergent[0]], diameters[divergent[1]], diameters[divergent[2]])
                q_in = abs(flowrates[divergent[0]]) * h_list[divergent[0]]
                q_out = abs(flowrates[divergent[1]]) * h_list[divergent[1]] + abs(flowrates[divergent[2]]) * h_list[divergent[2]]
                if q_in != 0.:
                    assert 0.99999 < q_in / q_out < 1.00001, (q_in / q_out)
                elif q_in == 0.:
                    assert q_in == q_out, (q_in, q_out)
                else:
                    assert 0
        elif index in central_node_converging:
            convergent = converging_bifurcations[central_node_converging.index(index)]
            if abs(flowrates[convergent[0]]) < 10**(-50):
                h_list[convergent[0]] = 0
            else:
                h_list[convergent[0]] = (abs(flowrates[convergent[1]])*h_list[convergent[1]] + abs(flowrates[convergent[2]])*h_list[convergent[2]])/abs(flowrates[convergent[0]])
                q_in = abs(flowrates[convergent[1]])*h_list[convergent[1]] + abs(flowrates[convergent[2]]) * h_list[convergent[2]]
                q_out = abs(flowrates[convergent[0]]) * h_list[convergent[0]]
                if q_in != 0.:
                    assert 0.99999 < q_in / q_out < 1.00001, (q_in / q_out)
                elif q_in == 0.:
                    assert q_in == q_out, (q_in, q_out)
                else:
                    assert 0
        else:
            pass
    return h_list


def pries_solver(q_0, q_1, q_2, h_0, d_0, d_1, d_2):
    """
    Implementation of the Pries model from Secomb 2017 review paper
    :param q_0: float, flowrate of parent (diverging) branch
    :param q_1: float, flowrate child branch 1
    :param q_2: float, flowrate child rbanch 2
    :param h_0: float, haematocrit of parent branch
    :param d_0: float, diameter parent branch in um
    :param d_1: float, diameter child branch 1 in um
    :param d_2: float, diameter child branch 2 in um
    :return h_1: float, haematocrit in child branch 1
    :return h_2: float, haematocrit in child branch 2
    """
    e = math.e
    FQB = abs(q_1 / q_0) # the absolute is necessary as the convention for flowrates can lead to a negative FQB, which is of course unphysical and doesn't give right results
    x0 = 0.964 * (1. - h_0) / d_0
    A = -13.29 * ((d_1 ** 2 / d_2 ** 2) - 1.) / ((d_1 ** 2 / d_2 ** 2) + 1.) * (1. - h_0) / d_0
    B = 1 + 6.98 * (1. - h_0) / d_0
    if FQB <= x0:
        FQE = 0
    elif (1. - x0) <= FQB:
        FQE = 1.
    else:
        logitFQE = A + B * logit((FQB - x0) / (1 - 2 * x0))
        FQE = e ** logitFQE / (1. + e ** logitFQE)
    h_1 = FQE * abs(q_0) * h_0 / abs(q_1)
    if FQE == 1.: # This is necessary as h2 calculation can sometimes be a very small negative value, which leads to a crash. Might be easier to calculate h2 through FQE2 = 1 - FQE1 ?
        h_2 = 0.
    else:
        h_2 = (abs(q_0 * h_0) - abs(q_1 * h_1)) / abs(q_2)
    assert 1. > h_1 >= 0., (h_1, q_1, q_2, q_0)
    assert 1. > h_2 >= 0., (h_2, q_1, q_2, q_0)
    return h_1, h_2


if __name__ == "__main__":
    pass
