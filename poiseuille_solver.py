import poiseuille_network_functions
import poiseuille_reader
import poiseuille_functions
from poiseuille_class import PoiseuilleNetwork
import numpy as np
from argparse import ArgumentParser


def main(polyline_path, bc_node_list, pressure_bc_list, h_bc_list, alpha=0.05, print_option=False):
    """
    Main function, takes the network and boundary conditions and returns the solved flow/haematocrit/pressure in the network
    Uses Poiseuille's law for the flow, and Pries' empirical model for phase separation
    See Secomb 2017 annual review on blood flow in the microcirculation as a starting point
    :param polyline_path: VTK polyline of the network, each edge contains the radius under string "Radius"
    :param bc_node_list: list of boundary nodes index values, [BC node 1 index, BC node 2 index .... , final BC node index]
    :param pressure_bc_list: list of pressure at boundary nodes [pressure BC node 1, ... , pressure final BC node], same length as bc_node_list with matching index between the two
    :param h_bc_list: list haematocrits at bounday nodes [haematocrit BC node 1, ..., haematocrit at final BC node], note haematocrit is defined at edge, so class finds edge corresponding to boundary node
    :param alpha: constant for predictor-corrector scheme, see Lorthois et al. NeuroImage 2011, part 1. Lower values lead to more stable solver but solver is then slower to converge
    :return: the PoiseuilleClass of solved network, containing pressure, flowrate, and haematocrits in the network
    """
    poiseuille_class = PoiseuilleNetwork(polyline_path, bc_node_list, pressure_bc_list, h_bc_list)

    solution_converged = False
    iteration = 1

    while solution_converged is False:
        j, current_miu = poiseuille_functions.solve_conductance(poiseuille_class)
        poiseuille_class.p = poiseuille_functions.generate_solve_poiseuille_matrix(poiseuille_class, j)
        poiseuille_class.Q = poiseuille_functions.solve_flow(poiseuille_class.p, j, poiseuille_class.edges)
        convergent_bifurcations, divergent_bifurcation = poiseuille_network_functions.bifurcation_types(poiseuille_class.Q, poiseuille_class.bifurcations, poiseuille_class.edges, poiseuille_class.p)
        poiseuille_class.H = poiseuille_functions.solve_haematocrit(poiseuille_class.Q, poiseuille_class, divergent_bifurcation, convergent_bifurcations, poiseuille_class.p)

        if iteration > 1:
            poiseuille_class.Q = [alpha * q_current + (1. - alpha) * q_previous for q_current, q_previous in zip(poiseuille_class.Q, previous_q)]
            poiseuille_class.H = [alpha * h_current + (1. - alpha) * h_previous for h_current, h_previous in zip(poiseuille_class.H, previous_h)]

            poiseuille_class.apparent_viscosity = current_miu
            poiseuille_class.wss = [4.*mu*abs(flowrate) / (np.pi*r**3) for mu, flowrate, r in zip(poiseuille_class.apparent_viscosity, poiseuille_class.Q, poiseuille_class.radius)]

            if poiseuille_network_functions.check_converged([previous_q, poiseuille_class.Q], error=0.05, print_option=print_option):
                print("Passing convergence check, checking mass balance...")
                assert poiseuille_network_functions.check_bifurcations(divergent_bifurcation, convergent_bifurcations, poiseuille_class.Q, poiseuille_class.p, poiseuille_class.edges) is True
                assert poiseuille_network_functions.check_mass_balance(poiseuille_class.Q, poiseuille_class.H, poiseuille_class.straights, divergent_bifurcation, convergent_bifurcations) is True
                solution_converged = True

        if print_option:
            print(iteration)
        iteration += 1
        previous_h, previous_q = poiseuille_class.H, poiseuille_class.Q
    return poiseuille_class


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_polyline', type=str, help='Input polyline with nodes, edges, and radius')
    parser.add_argument('input_iolets', type=str, help='Input csv with iolets')
    parser.add_argument('--print_option', type=int, help='Flag to print each iteration information in Poiseuille solver, argparse does not support a bool type, as it returns True for both True and False...', default=0, required=False, choices=[0, 1])
    args = parser.parse_args()

    nodes, pressure, haematocrits = poiseuille_reader.read_iolets_csv(args.input_iolets)
    poiseuille_reader.write_to_vtk(main(args.input_polyline, nodes, pressure, haematocrits, print_option=args.print_option))
