import math
import random
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
                    Mjk = getMsn(edge_j, node_k) * conductance[edge_index]
                    Lij = getLns(node_i, edge_j)
                    matrix_K[node_i, node_k] += Mjk * Lij
        for nodes_iolets in NetworkClass.iolets:  # this sets the boundary conditions with a 1 in the diagonal, so the pressure is known, the pressure is already in the NetworkClass.p0
            matrix_K[nodes_iolets[0], nodes_iolets[0]] = 1
        pressure_matrix = solve_matrix(matrix_K, NetworkClass.p_0)
    return pressure_matrix


def optimise_BC_efficient(NetworkClass, target_wss, target_pressure, k_p=1., k_tau=4.):
    """This implements the method described in Fry et al. 2012 to optimise the BCs based on target pressure/shear stress values
    currently works for both pressure and shear stress targets as single entry or list (for each node/edge)"""

    if type(target_pressure) is float: # This manages the fact that targets can be either a single entry, or a list
        target_pressure = [target_pressure]*len(NetworkClass.nodes)
    else:
        assert len(target_pressure) == len(NetworkClass.nodes)
    if type(target_wss) is float:
        target_wss = [target_wss]*len(NetworkClass.edges)
    else:
        assert len(target_wss) == len(NetworkClass.edges)

    matrix_dim = len(NetworkClass.nodes) + len(NetworkClass.IUB_nodes)
    matrix = np.zeros(shape=(matrix_dim, matrix_dim))
    k_matrix = np.zeros(shape=(len(NetworkClass.IUB_nodes), len(NetworkClass.nodes)))

    bvector = [0]*matrix_dim
    iolet_nodes = [node[0] for node in NetworkClass.iolets]

    """build k matrix and populate general matrix with h-matrix terms"""
    for bifurcation, bifurcation_neighbours in zip(NetworkClass.bifurcations, NetworkClass.bifurcation_neighbour_nodes):
        node_i = bifurcation_neighbours[0]
        node_k1, node_k2, node_k3 = bifurcation_neighbours[1], bifurcation_neighbours[2], bifurcation_neighbours[3]
        edge_j1, edge_j2, edge_j3 = bifurcation[0], bifurcation[1], bifurcation[2]
        w_i = get_wi(node_i, NetworkClass.edges, NetworkClass.length)
        if node_i in NetworkClass.IUB_nodes:
            index_IUB_node = NetworkClass.IUB_nodes.index(node_i)
            k_matrix[index_IUB_node, node_i] = -(NetworkClass.j[bifurcation[0]] + NetworkClass.j[bifurcation[1]] + NetworkClass.j[bifurcation[2]])
            k_matrix[index_IUB_node, bifurcation_neighbours[1]] = NetworkClass.j[bifurcation[0]]
            k_matrix[index_IUB_node, bifurcation_neighbours[2]] = NetworkClass.j[bifurcation[1]]
            k_matrix[index_IUB_node, bifurcation_neighbours[3]] = NetworkClass.j[bifurcation[2]]
        if node_i not in iolet_nodes:
            bvector[node_i] += target_pressure[node_i] * k_p * w_i
            bvector[node_i] += k_tau * target_wss[edge_j1] * getMsn(NetworkClass.edges[edge_j1], node_i) * NetworkClass.j[edge_j1] * NetworkClass.length[edge_j1] * 4 * NetworkClass.apparent_viscosity[edge_j1] / (math.pi * NetworkClass.radius[edge_j1] ** 3)
            bvector[node_i] += k_tau * target_wss[edge_j2] * getMsn(NetworkClass.edges[edge_j2], node_i) * NetworkClass.j[edge_j2] * NetworkClass.length[edge_j2] * 4 * NetworkClass.apparent_viscosity[edge_j2] / (math.pi * NetworkClass.radius[edge_j2] ** 3)
            bvector[node_i] += k_tau * target_wss[edge_j3] * getMsn(NetworkClass.edges[edge_j3], node_i) * NetworkClass.j[edge_j3] * NetworkClass.length[edge_j3] * 4 * NetworkClass.apparent_viscosity[edge_j3] / (math.pi * NetworkClass.radius[edge_j3] ** 3)
            matrix[node_i, node_i] += k_p * w_i
            c_j = 4 * NetworkClass.apparent_viscosity[edge_j1] / (math.pi * NetworkClass.radius[edge_j1] ** 3)
            matrix[node_i, node_k1] += k_tau * NetworkClass.length[edge_j1] * (c_j ** 2) * getMsn(NetworkClass.edges[edge_j1], node_i) * NetworkClass.j[edge_j1] * getMsn(NetworkClass.edges[edge_j1], node_k1) * NetworkClass.j[edge_j1]
            matrix[node_i, node_i] -= matrix[node_i, node_k1]
            c_j = 4 * NetworkClass.apparent_viscosity[edge_j2] / (math.pi * NetworkClass.radius[edge_j2] ** 3)
            matrix[node_i, node_k2] += k_tau * NetworkClass.length[edge_j2] * (c_j ** 2) * getMsn(NetworkClass.edges[edge_j2], node_i) * NetworkClass.j[edge_j2] * getMsn(NetworkClass.edges[edge_j2], node_k2) * NetworkClass.j[edge_j2]
            matrix[node_i, node_i] -= matrix[node_i, node_k2]
            c_j = 4 * NetworkClass.apparent_viscosity[edge_j3] / (math.pi * NetworkClass.radius[edge_j3] ** 3)
            matrix[node_i, node_k3] += k_tau * NetworkClass.length[edge_j3] * (c_j ** 2) * getMsn(NetworkClass.edges[edge_j3], node_i) * NetworkClass.j[edge_j3] * getMsn(NetworkClass.edges[edge_j3], node_k3) * NetworkClass.j[edge_j3]
            matrix[node_i, node_i] -= matrix[node_i, node_k3]

    for straight, straight_neighbours in zip(NetworkClass.straights, NetworkClass.straight_neighbour_nodes):
        node_i = straight_neighbours[0]
        node_k1, node_k2 = straight_neighbours[1], straight_neighbours[2]
        edge_j1, edge_j2 = straight[0], straight[1]
        w_i = get_wi(node_i, NetworkClass.edges, NetworkClass.length)
        if node_i in NetworkClass.IUB_nodes:
            index_IUB_node = NetworkClass.IUB_nodes.index(node_i)
            k_matrix[index_IUB_node, node_i] = -(NetworkClass.j[straight[0]] + NetworkClass.j[straight[1]])
            k_matrix[index_IUB_node, straight_neighbours[1]] = NetworkClass.j[straight[0]]
            k_matrix[index_IUB_node, straight_neighbours[2]] = NetworkClass.j[straight[1]]
        if node_i not in iolet_nodes:
            bvector[node_i] += target_pressure[node_i] * k_p * w_i
            bvector[node_i] += k_tau * target_wss[edge_j1] * getMsn(NetworkClass.edges[edge_j1], node_i) * NetworkClass.j[edge_j1] * NetworkClass.length[edge_j1] * 4 * NetworkClass.apparent_viscosity[edge_j1] / (math.pi * NetworkClass.radius[edge_j1] ** 3)
            bvector[node_i] += k_tau * target_wss[edge_j2] * getMsn(NetworkClass.edges[edge_j2], node_i) * NetworkClass.j[edge_j2] * NetworkClass.length[edge_j2] * 4 * NetworkClass.apparent_viscosity[edge_j2] / (math.pi * NetworkClass.radius[edge_j2] ** 3)
            matrix[node_i, node_i] += k_p * w_i
            c_j = 4 * NetworkClass.apparent_viscosity[edge_j1] / (math.pi * NetworkClass.radius[edge_j1] ** 3)
            matrix[node_i, node_k1] += k_tau * NetworkClass.length[edge_j1] * (c_j ** 2) * getMsn(NetworkClass.edges[edge_j1], node_i) * NetworkClass.j[edge_j1] * getMsn(NetworkClass.edges[edge_j1], node_k1) * NetworkClass.j[edge_j1]
            matrix[node_i, node_i] -= matrix[node_i, node_k1]
            c_j = 4 * NetworkClass.apparent_viscosity[edge_j2] / (math.pi * NetworkClass.radius[edge_j2] ** 3)
            matrix[node_i, node_k2] += k_tau * NetworkClass.length[edge_j2] * (c_j ** 2) * getMsn(NetworkClass.edges[edge_j2], node_i) * NetworkClass.j[edge_j2] * getMsn(NetworkClass.edges[edge_j2], node_k2) * NetworkClass.j[edge_j2]
            matrix[node_i, node_i] -= matrix[node_i, node_k2]

    for single, single_neighbours in zip(NetworkClass.singles, NetworkClass.single_neighbour_nodes):
        node_i = single_neighbours[0]
        node_k1 = single_neighbours[1]
        edge_j1 = single[0]
        w_i = get_wi(node_i, NetworkClass.edges, NetworkClass.length)
        if node_i in NetworkClass.IUB_nodes:
            index_IUB_node = NetworkClass.IUB_nodes.index(node_i)
            k_matrix[index_IUB_node, single_neighbours[1]] = NetworkClass.j[single[0]]
        if node_i not in iolet_nodes:
            bvector[node_i] += target_pressure[node_i] * k_p * w_i
            bvector[node_i] += k_tau * target_wss[edge_j1] * getMsn(NetworkClass.edges[edge_j1], node_i) * NetworkClass.j[edge_j1] * NetworkClass.length[edge_j1] * 4 * NetworkClass.apparent_viscosity[edge_j1] / (math.pi * NetworkClass.radius[edge_j1] ** 3)
            matrix[node_i, node_i] += k_p * w_i
            c_j = 4 * NetworkClass.apparent_viscosity[edge_j1] / (math.pi * NetworkClass.radius[edge_j1] ** 3)
            matrix[node_i, node_k1] += k_tau * NetworkClass.length[edge_j1] * (c_j ** 2) * getMsn(NetworkClass.edges[edge_j1], node_i) * NetworkClass.j[edge_j1] * getMsn(NetworkClass.edges[edge_j1], node_k1) * NetworkClass.j[edge_j1]
            matrix[node_i, node_i] -= matrix[node_i, node_k1]

    """Add BCs to relevant parts"""
    for nodes_iolets in NetworkClass.iolets:
        bvector[nodes_iolets[0]] = nodes_iolets[1]
        matrix[nodes_iolets[0], nodes_iolets[0]] = 1.
        index_IUB_node = NetworkClass.IUB_nodes.index(nodes_iolets[0])
        k_matrix[index_IUB_node, nodes_iolets[0]] = 1.
        bvector[NetworkClass.IUB_matrix_idx[index_IUB_node]] = nodes_iolets[1]

    """Place k-matrix into general matrix"""
    matrix[len(NetworkClass.nodes):matrix_dim, 0:len(NetworkClass.nodes)] = k_matrix
    matrix[0:len(NetworkClass.nodes), len(NetworkClass.nodes):matrix_dim] = k_matrix.transpose()

    pressure_lambda_matrix = solve_matrix(matrix, bvector)

    return pressure_lambda_matrix[0:len(NetworkClass.nodes)]


def optimise_BC_original(NetworkClass, target_wss, target_pressure, k_p=1., k_tau=4.):
    """This implements the method described in Fry et al. 2012 to optimise the BCs based on target pressure/shear stress values
    currently works for both pressure and shear stress targets as single entry or list (for each node/edge)
    Doesn't work yet for known BCs (congrats, that's the easy part) I THINK IT WORKS, NEED TO DO MORE TESTS
    Also need to have the loop to determine flow direction to find the minima (see paper and TWS code for details)"""

    import sys
    np.set_printoptions(precision=30, linewidth=sys.maxsize)
    # np.set_printoptions(suppress=True)

    if type(target_pressure) is float: # This manages the fact that targets can be either a single entry, or a list
        target_pressure = [target_pressure]*len(NetworkClass.nodes)
    else:
        assert len(target_pressure) == len(NetworkClass.nodes)
    if type(target_wss) is float:
        target_wss = [target_wss]*len(NetworkClass.edges)
    else:
        assert len(target_wss) == len(NetworkClass.edges)

    matrix_dim = len(NetworkClass.nodes) + len(NetworkClass.IUB_nodes)
    matrix = np.zeros(shape=(matrix_dim, matrix_dim))
    bvector = [0]*matrix_dim

    iolet_nodes = [node[0] for node in NetworkClass.iolets]

    """Purpose for the moment is getting it to work
    Then optimise matrix setup/solution
    Then add flow directions (look at TWS setup on how to do that, but it starts with some sort of random assignment and iterates until flow direction no longer changes)"""

    for nodes_iolets in NetworkClass.iolets:  # this sets the boundary conditions with a 1 in the diagonal, so the pressure is known, the pressure is already in the NetworkClass.p0
        bvector[nodes_iolets[0]] = nodes_iolets[1]
        matrix[nodes_iolets[0], nodes_iolets[0]] = 1.

    for c, node_i in enumerate(NetworkClass.nodes):
        # print(c, len(NetworkClass.nodes))
        node_i = node_i[0]
        w_i = get_wi(node_i, NetworkClass.edges, NetworkClass.length)
        if node_i not in iolet_nodes:
            bvector[node_i] += target_pressure[node_i] * k_p * w_i

        for edge_j in NetworkClass.edges:
            edge_index = edge_j[0]
            Mji = getMsn(edge_j, node_i) * NetworkClass.j[edge_index]
            l_j = NetworkClass.length[edge_index]
            c_j = 4 * NetworkClass.apparent_viscosity[edge_index] / (math.pi * NetworkClass.radius[edge_index] ** 3)
            if node_i not in iolet_nodes:
                bvector[node_i] += k_tau * target_wss[edge_index] * Mji * l_j * c_j

        for node_k in NetworkClass.nodes:
            node_k = node_k[0]
            if node_k == node_i and node_i not in iolet_nodes:
                matrix[node_i, node_k] += k_p * w_i

            for edge_j in NetworkClass.edges:
                edge_index = edge_j[0]
                l_j = NetworkClass.length[edge_index]
                c_j = 4*NetworkClass.apparent_viscosity[edge_index]/(math.pi*NetworkClass.radius[edge_index]**3)
                Mji = getMsn(edge_j, node_i) * NetworkClass.j[edge_index]
                Mjk = getMsn(edge_j, node_k) * NetworkClass.j[edge_index]
                if node_i not in iolet_nodes:
                    matrix[node_i, node_k] += k_tau * l_j * (c_j ** 2) * Mji * Mjk
        for node_k, matrix_index in zip(NetworkClass.IUB_nodes, NetworkClass.IUB_matrix_idx):
            # print(node_k, matrix_index)
            for edge_j in NetworkClass.edges:
                edge_index = edge_j[0]
                Lkj = getLns(node_k, edge_j)
                Mji = getMsn(edge_j, node_i) * NetworkClass.j[edge_index]
                Kki = Lkj * Mji

                if node_i not in iolet_nodes:
                    matrix[node_i, matrix_index] += Kki
                if node_i in iolet_nodes and node_k in iolet_nodes:
                    index = iolet_nodes.index(node_i)
                    # bvector[matrix_index] -= Kki * NetworkClass.iolets[index][1] #!TODO what is this double entry ?? check if it works without? I think below is right and remove this one
                    bvector[matrix_index] -= getLns(node_k, edge_j) * getMsn(edge_j, node_i) * NetworkClass.iolets[index][1]
                if node_i in iolet_nodes:
                    if node_k == node_i:
                        matrix[node_i, matrix_index] = 1. # this gets added three times, youpee, and everywhere too, get rid of it due to = instead of +=, but still buggy in formulation
                    else:
                        matrix[node_i, matrix_index] += Kki

    """Transpose the k (conductance) portion of matrix, this save quite a bit of time and works, TWS/PS use the same technique"""
    k_transpose = matrix[0:len(NetworkClass.nodes), len(NetworkClass.nodes):matrix_dim].transpose()
    matrix[len(NetworkClass.nodes):matrix_dim, 0:len(NetworkClass.nodes)] = k_transpose

    for row in matrix:
        print(row)
    print("matrix above, bvector below")
    print(bvector)

    pressure_lambda_matrix = solve_matrix(matrix, bvector)
    print(pressure_lambda_matrix[0:len(NetworkClass.nodes)])  # this looks at the solution of the pressure values after optimised, it doesn't print the lambda values at the end

    return pressure_lambda_matrix[0:len(NetworkClass.nodes)]


def flow_opt_outer_loop(NetworkClass, target_wss, target_pressure, k_p=1., k_tau=4.):
    """This introduces a heuristic approach to determine flow direction to minimise D
    Based on method described in Fry et al. 2012
    But looks at minimised D based on 50 random flow direction initialisations
    1. 50 random flow direction initliasation
    2. Each of 50 converges until no flow direction change with increasing k_tau
    3. calculate deviation from target for each converged iteration
    4. pick the one with the lowest deviation as the optimum
    I think this allows to stochastically pick the global minimum from a set of local minimas
    Although it depends on the random process being sufficiently representative"""

    D_list, fd_list = [], []

    undirected_target_wss = [abs(wss) for wss in target_wss]
    k_start = 0.002
    k_steps = 50

    for counter in range(50):
        print(counter)
        flow_directions = [1] * len(NetworkClass.edges)
        random_choice = [-1., 1.]
        random.seed(counter)  # tried different values and still gives correct flow directions in the end, so the method works to some degree
        for c, fd in enumerate(flow_directions):
            flow_directions[c] = fd * random.choice(random_choice)

        for c, k_step in enumerate(range(k_steps)):
            if c == 0:
                k_tau_steps = k_start
            else:
                k_tau_steps = k_tau_steps*2.
            current_target_wss = [tau*fd for tau, fd in zip(undirected_target_wss, flow_directions)]
            optimised_pressure = optimise_BC_efficient(NetworkClass, current_target_wss, target_pressure, k_tau=k_tau_steps, k_p=k_p)
            optimised_flowrates = solve_flow(optimised_pressure, NetworkClass.j, NetworkClass.edges)
            direction_change = 0
            for count, (q, fd) in enumerate(zip(optimised_flowrates, flow_directions)):
                if q < 0 and fd > 0:
                    direction_change += 1
                    flow_directions[count] = -1.
                elif q > 0 and fd < 0:
                    direction_change += 1
                    flow_directions[count] = 1.

            if direction_change == 0:
                current_target_wss = [tau * fd for tau, fd in zip(undirected_target_wss, flow_directions)]
                optimised_pressure = optimise_BC_efficient(NetworkClass, current_target_wss, target_pressure, k_tau=k_tau, k_p=k_p)
                optimised_flowrates = solve_flow(optimised_pressure, NetworkClass.j, NetworkClass.edges)
                predicted_wss = [4. * mu * flowrate / (np.pi * r ** 3) for mu, flowrate, r in zip(NetworkClass.apparent_viscosity, optimised_flowrates, NetworkClass.radius)]
                calculate_deviation(target_pressure, target_wss, k_p, k_tau, optimised_pressure, predicted_wss, NetworkClass)
                D_list.append(calculate_deviation(target_pressure, target_wss, k_p, k_tau, optimised_pressure, predicted_wss, NetworkClass))
                fd_list.append(flow_directions)
                break

    min_index = D_list.index(min(D_list))
    flow_directions = fd_list[min_index]

    current_target_wss = [tau * fd for tau, fd in zip(undirected_target_wss, flow_directions)]
    optimised_pressure = optimise_BC_efficient(NetworkClass, current_target_wss, target_pressure, k_tau=k_tau, k_p=k_p)
    flowrates = solve_flow(optimised_pressure, NetworkClass.j, NetworkClass.edges)
    predicted_wss = [4. * mu * flowrate / (np.pi * r ** 3) for mu, flowrate, r in zip(NetworkClass.apparent_viscosity, flowrates, NetworkClass.radius)]
    print(D_list)
    print("D value is", calculate_deviation(target_pressure, target_wss, k_p, k_tau, optimised_pressure, predicted_wss, NetworkClass))
    return optimised_pressure


def calculate_deviation(target_p, target_wss, k_p, k_tau, predicted_p, predicted_wss, NWClass, pressure_calc=True):
    """This just quantifies the deviation from target, which is what we are minimising in the scheme
    Way of doing a check of how we're doing really
    Note that target p section only really works when there is an actual target p somewhere, else pressure is all over the place (but relative values work)"""
    deviation = 0.
    if pressure_calc:
        for node in NWClass.nodes:
            node_index = node[0]
            p = predicted_p[node_index]
            t_p = target_p[node_index]
            w_k = get_wi(node_index, NWClass.edges, NWClass.length)
            deviation += 0.5*k_p*w_k*(p - t_p)**2
    for edge in NWClass.edges:
        edge_index = edge[0]
        p_wss = predicted_wss[edge_index]
        t_wss = target_wss[edge_index]
        deviation += 0.5*k_tau*NWClass.length[edge_index]*(p_wss - t_wss)**2
    return deviation


def getMsn(segment_s, node_n):
    start_node_segment = segment_s[1]
    end_node_segment = segment_s[2]
    if node_n == start_node_segment:
        M_sn = 1.
    elif node_n == end_node_segment:
        M_sn = -1.
    else:
        M_sn = 0.
    return M_sn


def getLns(node_n, segment_s):
    start_node_segment = segment_s[1]
    end_node_segment = segment_s[2]
    if node_n == start_node_segment:
        L_ns = -1.
    elif node_n == end_node_segment:
        L_ns = 1.
    else:
        L_ns = 0.
    return L_ns


def get_wi(node_i, edges, lengths):
    l_j = 0
    for edge in edges:
        if node_i in [edge[1], edge[2]]:
            l_j += lengths[edge[0]]
    return 0.5*l_j


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
        logitFQE = A + B * logit((FQB - x0) / (1. - 2. * x0))
        FQE = e ** logitFQE / (1. + e ** logitFQE)
    h_1 = FQE * abs(q_0) * h_0 / abs(q_1)
    if FQE == 1.: # This is necessary as h2 calculation can sometimes be a very small negative value, which leads to a crash. Might be easier to calculate h2 through FQE2 = 1 - FQE1 ?
        h_2 = 0.
    else:
        h_2 = (abs(q_0 * h_0) - abs(q_1 * h_1)) / abs(q_2)
    assert 1. > h_1 >= 0., (h_1, q_1, q_2, q_0)
    assert 1. > h_2 >= 0., (h_2, q_1, q_2, q_0)
    return h_1, h_2


def test_3_branch_system():
    from poiseuille_class import PoiseuilleNetwork
    import poiseuille_reader
    path_vtp = "/Users/romainenjalbert/Documents/code/poiseuille_solver/test_fry_optimiser/test_Fry_optimiser.vtp"
    path_0_bc = "/Users/romainenjalbert/Documents/code/poiseuille_solver/test_fry_optimiser/test_Fry_optimiser_unknown.csv"
    path_1_bc = "/Users/romainenjalbert/Documents/code/poiseuille_solver/test_fry_optimiser/test_Fry_optimiser_1known.csv"
    path_2_bc = "/Users/romainenjalbert/Documents/code/poiseuille_solver/test_fry_optimiser/test_Fry_optimiser_2known.csv"
    path_3_bc = "/Users/romainenjalbert/Documents/code/poiseuille_solver/test_fry_optimiser/test_Fry_optimiser_known.csv"
    class_0_bc = PoiseuilleNetwork(path_vtp, *poiseuille_reader.read_iolets_csv(path_0_bc))
    class_1_bc = PoiseuilleNetwork(path_vtp, *poiseuille_reader.read_iolets_csv(path_1_bc))
    class_2_bc = PoiseuilleNetwork(path_vtp, *poiseuille_reader.read_iolets_csv(path_2_bc))
    class_3_bc = PoiseuilleNetwork(path_vtp, *poiseuille_reader.read_iolets_csv(path_3_bc))

    class_3_bc.j, class_3_bc.apparent_viscosity = solve_conductance(class_3_bc)
    p_3_bc = generate_solve_poiseuille_matrix(class_3_bc, class_3_bc.j, sparse=False)
    q_3_bc = solve_flow(p_3_bc, class_3_bc.j, class_3_bc.edges)
    wss_3_bc = [4. * mu * flowrate / (np.pi * r ** 3) for mu, flowrate, r in zip(class_3_bc.apparent_viscosity, q_3_bc, class_3_bc.radius)]

    class_0_bc.j, class_0_bc.apparent_viscosity = solve_conductance(class_0_bc)
    p_0_bc = optimise_BC_efficient(class_0_bc, wss_3_bc, p_3_bc)
    q_0_bc = solve_flow(p_0_bc, class_0_bc.j, class_0_bc.edges)
    wss_0_bc = [4. * mu * flowrate / (np.pi * r ** 3) for mu, flowrate, r in zip(class_0_bc.apparent_viscosity, q_0_bc, class_0_bc.radius)]

    assert np.array(wss_0_bc).all() == np.array(wss_3_bc).all(), (wss_0_bc, wss_3_bc)
    assert p_0_bc.all() == p_3_bc.all(), (p_0_bc == p_3_bc)

    return


def test_large_network():
    from poiseuille_class import PoiseuilleNetwork
    import poiseuille_reader
    import matplotlib.pyplot as plt
    path_vtp = "/Users/romainenjalbert/Documents/code/poiseuille_solver/test_fry_optimiser/complex_network_6iolets.vtp"
    path_bck = "/Users/romainenjalbert/Documents/code/poiseuille_solver/test_fry_optimiser/complex_network_6iolets_bck.csv"
    path_bcu = "/Users/romainenjalbert/Documents/code/poiseuille_solver/test_fry_optimiser/complex_network_6iolets_bcu.csv"

    class_bck = PoiseuilleNetwork(path_vtp, *poiseuille_reader.read_iolets_csv(path_bck))
    class_bcu = PoiseuilleNetwork(path_vtp, *poiseuille_reader.read_iolets_csv(path_bcu))

    class_bck.j, class_bck.apparent_viscosity = solve_conductance(class_bck)
    p_bck = generate_solve_poiseuille_matrix(class_bck, class_bck.j)
    q_bck = solve_flow(p_bck, class_bck.j, class_bck.edges)
    wss_bck = [4. * mu * flowrate / (np.pi * r ** 3) for mu, flowrate, r in zip(class_bck.apparent_viscosity, q_bck, class_bck.radius)]

    class_bcu.j, class_bcu.apparent_viscosity = solve_conductance(class_bcu)
    p_bcu = optimise_BC_efficient(class_bcu, wss_bck, p_bck)
    q_bcu = solve_flow(p_bcu, class_bcu.j, class_bcu.edges)
    wss_bcu = [4. * mu * flowrate / (np.pi * r ** 3) for mu, flowrate, r in zip(class_bcu.apparent_viscosity, q_bcu, class_bcu.radius)]

    assert np.array(wss_bcu).all() == np.array(wss_bck).all(), (wss_bcu, wss_bck)
    for pk, pu in zip(p_bck, p_bcu):
        if pu < 10**(-10):
            pu = 0.
        if pk < 10**(-10):
            pk = 0.
        assert pu*1.0001 >= pk >= pu*.9999, (pu*1.0001, pk, pu*.9999)

    """write target pressure and wss for use later"""
    import csv
    with open("/Users/romainenjalbert/Documents/code/poiseuille_solver/test_fry_optimiser/complex_network_6iolets_target_p.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for counter, pressure in enumerate(p_bck):
            writer.writerow([counter, pressure])
    with open("/Users/romainenjalbert/Documents/code/poiseuille_solver/test_fry_optimiser/complex_network_6iolets_target_wss.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        for counter, wss_bck in enumerate(wss_bck):
            writer.writerow([counter, wss_bck])

    assert 0
    plt.scatter(p_bck, p_bcu)
    plt.show()
    plt.close()

    plt.scatter(wss_bck, wss_bck)
    plt.show()
    plt.close()

    return


if __name__ == "__main__":
    pass
