import math
import numpy as np
from scipy.sparse.linalg import spsolve
# import scipy.sparse.linalg # these are for the sparse matrix solution, which isn't really used so comment out
from scipy.sparse import coo_matrix


def exp(x):
    return math.exp(x)


def logit(x):
    return math.log(x/(1.-x))


def solve_conductance(NetworkClass):
    """
    Generates conductance values based on Poiseuille equation and viscosity empirical relations from Secomb 2017 review
    :param conductance_edges: (edge_no, node_1, node_2) list at every entry
    :param conductance_r: list with all radii indexed by edge_no
    :param conductance_d: list with all diameters indexed by edge_no
    :param conductance_l: list with all lengths indexed by edge_no
    :param conductance_H: list with all hct's indexed by edge_no
    :return: list with conductance and viscosity, indexed by edge_no
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
        if NetworkClass.compression_resistance[edge_number] == False:
            conductance = math.pi*NetworkClass.radius[edge_number]**4/(8.*NetworkClass.length[edge_number]*miu)
        elif NetworkClass.compression_resistance[edge_number] == True:
            equivalent_compressed_radius = NetworkClass.equivalent_radius[edge_number]
            major_r, minor_r = NetworkClass.major[edge_number], NetworkClass.minor[edge_number]
            conductance = math.pi * major_r**3. * minor_r**3. / (4. * NetworkClass.length[edge_number] * miu * (major_r**2. + minor_r**2.))
            # conductance = math.pi * equivalent_compressed_radius ** 4 / (8 * NetworkClass.length[edge_number] * miu)
        else:
            assert 0
        conductance_list.append(conductance)
        viscosity_list.append(miu)
    return conductance_list, viscosity_list


def generate_solve_poiseuille_matrix(network_class, conductance):
    """
    Generates a zero matrix of number_nodes x number_nodes and populates it with the conductance and known pressure (BC) and solves for pressure
    :param matrix_nodes: (node_number, x_coord, y_coord) at every row indexed by node number
    :param bifurcations_list: (edge_1, edge_2, edge_3, central_node) at every row forming a bifurcation
    :param bifurcations_neighbours: (central_node, node_1, node_2, node_3) at every row of a bifurcation
    :param straights_list: (edge_1, edge_2, central_node) at every row of a straight
    :param straights_neighbours: (central_node, node_1, node_2) at every row of a straight
    :param conductance: List of conductances indexed by edge number
    :param iolets: (node_number, pressure_BC) at every row
    :param p_0: Current pressure list wih BCs already in place
    :return: List of pressure values indexed by node number
    """


    row = []
    col = []
    data = []
    for bifurcation, bifurcation_neighbours in zip(network_class.bifurcations, network_class.bifurcation_neighbour_nodes):
        row.append(bifurcation[3]); col.append(bifurcation[3]); data.append(-(conductance[bifurcation[0]] + conductance[bifurcation[1]] + conductance[bifurcation[2]]))
        row.append(bifurcation_neighbours[0]); col.append(bifurcation_neighbours[1]); data.append(conductance[bifurcation[0]])
        row.append(bifurcation_neighbours[0]); col.append(bifurcation_neighbours[2]); data.append(conductance[bifurcation[1]])
        row.append(bifurcation_neighbours[0]); col.append(bifurcation_neighbours[3]); data.append(conductance[bifurcation[2]])
    for straight, straight_neighbours in zip(network_class.straights, network_class.straight_neighbour_nodes):
        row.append(straight[2]); col.append(straight[2]); data.append(-(conductance[straight[0]] + conductance[straight[1]]))
        row.append(straight_neighbours[0]); col.append(straight_neighbours[1]); data.append(conductance[straight[0]])
        row.append(straight_neighbours[0]); col.append(straight_neighbours[2]); data.append(conductance[straight[1]])
    for nodes_iolets in network_class.iolets: #This imposes the BCs in the matrix so that it doesn't change BCs to an unknown to be solved should it be linked to a straight/bifurcation
        row.append(nodes_iolets[0]), col.append(nodes_iolets[0]), data.append(1)

    matrix_size = len(network_class.nodes)

    sparse_matrix = coo_matrix((data, (row, col)), shape=(matrix_size, matrix_size)).tocsr() # this converts the matrix to another format scipy can solve, reason for ussing coo is that its easy for me...

    pressure_matrix = spsolve(sparse_matrix, network_class.p_0)

    return pressure_matrix


def solve_flow(pressure_matrix, conductance, edges_list):
    """
    Use available data to solve for flowrate through Poiseuille
    :param pressure_matrix: Matrix containing pressure values indexed by node_number
    :param conductance: List of conductances indexed by edge_number
    :param edges_list: standard edges input (edge_no, node_0, node_1)
    :return: Flowrates indexed by edge number
    """
    flow_list = []
    for edge in edges_list:
        flow = conductance[edge[0]] * (pressure_matrix[edge[1]] - pressure_matrix[edge[2]])
        flow_list.append(flow)
    return flow_list


def solve_haematocrit(flowrates, network_class, diverging_bifurcations, converging_bifurcations, p_matrix, h_solver):
    """
    Implements a mass balance for converging and straights. For diverging bifurcations, chooses which model to use and implements HCT split model
    :param flowrates: Flowrates indexed by edges
    :param h_list: Current hct indexed by edges
    :param diameters: Diameter indexed by edges, not D is in um
    :param diverging_bifurcations: (edge_1, edge_2, edge_3, central_node) in diverging bifurcations, edge_1 splits to 2 and 3
    :param converging_bifurcations: (edge_1, edge_2, edge_3, central_node) in converging bifurcations edge_2 and 3 converge into 1
    :param straight_bifurcations: (edge_1, edge_2, central_node) where there is no bifurcation
    :param edges: List of edges
    :param p_matrix: Pressure list indexed by node number
    :param h_solver: Choice of string to determine whether to use Pries solver or our solver
    :return: Returns the updated HCT list indexed by edge number
    """

    h_list = network_class.H
    straight_bifurcations = network_class.straights
    edges = network_class.edges
    compression_choice = network_class.ROM_choice
    diameters = network_class.D

    """NEW"""
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
                # assert 0
            if p_matrix[edges[straight[0], 1]] >= p_matrix[edges[straight[0], 2]] and p_matrix[edges[straight[0], 1]] >= \
                    p_matrix[edges[straight[1], 1]] and p_matrix[edges[straight[0], 1]] >= p_matrix[
                edges[straight[1], 2]]:
                h_list[straight[1]] = h_list[straight[0]]
            elif p_matrix[edges[straight[0], 2]] >= p_matrix[edges[straight[0], 1]] and p_matrix[
                edges[straight[0], 2]] >= p_matrix[edges[straight[1], 1]] and p_matrix[edges[straight[0], 2]] >= \
                    p_matrix[edges[straight[1], 2]]:
                h_list[straight[1]] = h_list[straight[0]]
            else:
                h_list[straight[0]] = h_list[straight[1]]
        elif index in central_node_diverging:
            divergent = diverging_bifurcations[central_node_diverging.index(index)]
            if abs(flowrates[divergent[0]]) < 10. ** (-50):
                h_list[divergent[1]] = 0
                h_list[divergent[2]] = 0
            else:
                if h_solver == "pries":
                    h_list[divergent[1]], h_list[divergent[2]] = pries_solver(flowrates[divergent[0]], flowrates[divergent[1]], flowrates[divergent[2]], h_list[divergent[0]], diameters[divergent[0]], diameters[divergent[1]], diameters[divergent[2]])
                    q_in = abs(flowrates[divergent[0]]) * h_list[divergent[0]]
                    q_out = abs(flowrates[divergent[1]]) * h_list[divergent[1]] + abs(flowrates[divergent[2]]) * h_list[divergent[2]]

                    if q_in != 0.:
                        assert 0.99999 < q_in / q_out < 1.00001, (q_in / q_out)
                    elif q_in == 0.:
                        assert q_in == q_out, (q_in, q_out)
                    else:
                        assert 0
                elif h_solver == "updated_pries" and compression_choice[divergent[0]] == False:
                    h_list[divergent[1]], h_list[divergent[2]] = pries_solver(flowrates[divergent[0]], flowrates[divergent[1]], flowrates[divergent[2]], h_list[divergent[0]], diameters[divergent[0]], diameters[divergent[1]], diameters[divergent[2]])
                elif h_solver == "updated_pries" and compression_choice[divergent[0]] == True:
                    h_list[divergent[1]], h_list[divergent[2]] = pries_solver_updated(flowrates[divergent[0]], flowrates[divergent[1]], flowrates[divergent[2]], h_list[divergent[0]], diameters[divergent[0]], diameters[divergent[1]], diameters[divergent[2]])
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
    """END NEW"""

    # convergence_check = False
    # divergence_check = False
    # n = 1
    # while (convergence_check is False) or (divergence_check is False): # this in reality could be ordered by decreasing pressure, and would largely solve the slow solutions. Try this afternoon ?
    #     print("n is ", n)
    #     n += 1
    #     for convergent in converging_bifurcations:
    #         if abs(flowrates[convergent[0]]) < 10**(-50):
    #             h_list[convergent[0]] = 0
    #         else:
    #             h_list[convergent[0]] = (abs(flowrates[convergent[1]])*h_list[convergent[1]] + abs(flowrates[convergent[2]])*h_list[convergent[2]])/abs(flowrates[convergent[0]])
    #             q_in = abs(flowrates[convergent[1]])*h_list[convergent[1]] + abs(flowrates[convergent[2]]) * h_list[convergent[2]]
    #             q_out = abs(flowrates[convergent[0]]) * h_list[convergent[0]]
    #             if q_in != 0.:
    #                 assert 0.99999 < q_in / q_out < 1.00001, (q_in / q_out)
    #             elif q_in == 0.:
    #                 assert q_in == q_out, (q_in, q_out)
    #             else:
    #                 assert 0
    #
    #     for divergent in diverging_bifurcations:
    #         if abs(flowrates[divergent[0]]) < 10. ** (-50):
    #             h_list[divergent[1]] = 0
    #             h_list[divergent[2]] = 0
    #         else:
    #             if h_solver == "pries":
    #                 h_list[divergent[1]], h_list[divergent[2]] = pries_solver(flowrates[divergent[0]], flowrates[divergent[1]], flowrates[divergent[2]], h_list[divergent[0]], diameters[divergent[0]], diameters[divergent[1]], diameters[divergent[2]])
    #                 q_in = abs(flowrates[divergent[0]]) * h_list[divergent[0]]
    #                 q_out = abs(flowrates[divergent[1]]) * h_list[divergent[1]] + abs(flowrates[divergent[2]]) * h_list[divergent[2]]
    #
    #                 if q_in != 0.:
    #                     assert 0.99999 < q_in / q_out < 1.00001, (q_in / q_out)
    #                 elif q_in == 0.:
    #                     assert q_in == q_out, (q_in, q_out)
    #                 else:
    #                     assert 0
    #             elif h_solver == "updated_pries" and compression_choice[divergent[0]] == False:
    #                 h_list[divergent[1]], h_list[divergent[2]] = pries_solver(flowrates[divergent[0]], flowrates[divergent[1]], flowrates[divergent[2]], h_list[divergent[0]], diameters[divergent[0]], diameters[divergent[1]], diameters[divergent[2]])
    #             elif h_solver == "updated_pries" and compression_choice[divergent[0]] == True:
    #                 h_list[divergent[1]], h_list[divergent[2]] = pries_solver_updated(flowrates[divergent[0]], flowrates[divergent[1]], flowrates[divergent[2]], h_list[divergent[0]], diameters[divergent[0]], diameters[divergent[1]], diameters[divergent[2]])
    #             else:
    #                 assert 0
    #
    #     straight_iteration = False
    #     while straight_iteration is False:
    #         for straight in straight_bifurcations:
    #             if abs(flowrates[straight[0]]) and abs(flowrates[straight[1]]) < 10**(-50):
    #                 h_list[straight[0]] = 0
    #                 h_list[straight[1]] = 0
    #                 # assert 0
    #             if p_matrix[edges[straight[0], 1]] >= p_matrix[edges[straight[0], 2]] and p_matrix[edges[straight[0], 1]] >= p_matrix[edges[straight[1], 1]] and p_matrix[edges[straight[0], 1]] >= p_matrix[edges[straight[1], 2]]:
    #                 h_list[straight[1]] = h_list[straight[0]]
    #             elif p_matrix[edges[straight[0], 2]] >= p_matrix[edges[straight[0], 1]] and p_matrix[edges[straight[0], 2]] >= p_matrix[edges[straight[1], 1]] and p_matrix[edges[straight[0], 2]] >= p_matrix[edges[straight[1], 2]]:
    #                 h_list[straight[1]] = h_list[straight[0]]
    #             else:
    #                 h_list[straight[0]] = h_list[straight[1]]
    #         rbc_flux = [h * abs(q) for h, q in zip(h_list, flowrates)]
    #         straight_iteration = check_straights(rbc_flux, straight_bifurcations)
    #
    #         rbc_flux = [h * abs(q) for h, q in zip(h_list, flowrates)]
    #         convergence_check = check_converging(rbc_flux, converging_bifurcations)
    #         divergence_check = check_diverging(rbc_flux, diverging_bifurcations)
    # assert convergence_check is True
    # assert divergence_check is True
    return h_list


def pries_solver(q_0, q_1, q_2, h_0, d_0, d_1, d_2):
    """
    Implementation of the Pries model from Secomb 2017 review paper
    :param q_0: flowrate of diverging branch
    :param q_1: flowrate child branch 1
    :param q_2: flowrate child rbanch 2
    :param h_0: HCT parent branch
    :param d_0: diameter parent branch in um
    :param d_1: diameter child branch 1 in um
    :param d_2: diameter child branch 2 in um
    :return: HCT of the child branches
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


def pries_solver_updated(q_0, q_1, q_2, h_0, d_0, d_1, d_2):
    """
    Implementation of the Pries model from Secomb 2017 review paper updated with the new X0 term calculated in "fit_validation_to_pries.py"
    :param q_0: flowrate of diverging branch
    :param q_1: flowrate child branch 1
    :param q_2: flowrate child rbanch 2
    :param h_0: HCT parent branch
    :param d_0: diameter parent branch in um
    :param d_1: diameter child branch 1 in um
    :param d_2: diameter child branch 2 in um
    :return: HCT of the child branches
    """
    e = math.e
    FQB = abs(q_1 / q_0) # the absolute is necessary as the convention for flowrates can lead to a negative FQB, which is of course unphysical and doesn't give right results
    x0 = 4.1627 * (1. - h_0) / d_0
    A = -13.29 * ((d_1 ** 2 / d_2 ** 2) - 1.) / ((d_1 ** 2 / d_2 ** 2) + 1.) * (1. - h_0) / d_0
    B = 1 + 6.98 * (1. - h_0) / d_0
    if FQB <= x0:
        FQE = 0.
    elif (1. - x0) <= FQB:
        FQE = 1.
    else:
        logitFQE = A + B * logit((FQB - x0) / (1. - 2. * x0))
        FQE = e ** logitFQE / (1. + e ** logitFQE)
    h_1 = FQE * abs(q_0) * h_0 / abs(q_1)
    if FQE == 1.: # This is necessary as h2 calculation can sometimes be a very small negative value, which leads to a crash. Might be easier to calculate h2 through FQE2 = 1 - FQE1 ?
        h_2 = 0.
    else:
        h_2 = (abs(q_0 * h_0) - abs(q_1 * h_1)) / abs(q_2)
    assert 1. > h_1 >= 0., h_1
    assert 1. > h_2 >= 0., h_2
    return h_1, h_2


def check_converged(Q_results, error = 0.1):
    """
    Checks whether the current iteration has converged with the previous iteration
    :param Q_results: List of flowrates lists indexed by iteration
    :return: Boolean for whether solution has converged
    """
    converged = True
    if len(Q_results) == 1:
        converged = False
    else:
        current_Q = Q_results[-1]
        previous_Q = Q_results[-2]
        for count, (current, previous) in enumerate(zip(current_Q, previous_Q)):
            if abs(current) == abs(previous):
                pass
            elif abs(current) - abs(previous) >= abs(current)*error/100.:
                print(100.*(abs(current) - abs(previous)) / abs(current), current, previous, count, abs(current - previous))
                converged = False
                break
    return converged


def check_straights(Q, straights):
    """
    Checks that the straigh edges are conserving mass from a flowrate point of view
    :param Q:
    :param straights:
    :return:
    """
    for straight in straights:
        edge_1 = straight[0]
        edge_2 = straight[1]
        if abs(abs(Q[edge_1]) - abs(Q[edge_2])) > abs(Q[edge_1])*0.01/100:
            return False
    return True


def check_converging(Q, convergings):
    """
    Check that the converging bifurcations are converging mass
    :param Q:
    :param convergings:
    :return:
    """
    for converging in convergings:
        inlet_1 = converging[1]
        inlet_2 = converging[2]
        outlet = converging[0]
        Q_in = abs(Q[inlet_1]) + abs(Q[inlet_2])
        Q_out = abs(Q[outlet])
        if Q_in == Q_out:
            return True
        elif Q_in - Q_out >= Q_in*0.01/100:
            return False
    return True


def check_diverging(Q, divergings):
    """
    Check that the diverging bifurcations are conserving mass
    :param Q:
    :param divergings:
    :return:
    """
    for diverging in divergings:
        inlet = diverging[0]
        outlet_1 = diverging[1]
        outlet_2 = diverging[2]
        Q_in = abs(Q[inlet])
        Q_out = abs(Q[outlet_1]) + abs(Q[outlet_2])
        if Q_in == Q_out:
            return True
        elif Q_in - Q_out >= Q_in * 0.01 / 100:
            return False
    return True


def check_flow_reversal(Q_current, Q_previous):
    """

    :param Q_current:
    :param Q_previous:
    :return:
    """
    for q_current, q_previous in zip(Q_current, Q_previous):
        if q_current < 0.:
            q_current = -1
        elif q_current > 0.:
            q_current = 1
        else:
            assert 0

        if q_previous < 0.:
            q_previous = -1
        elif q_previous > 0.:
            q_previous = 1
        else:
            assert 0

        if q_previous != q_current:
            return True

    return False


if __name__ == "__main__":
    pass
