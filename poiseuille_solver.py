import poiseuille_network_functions
import poiseuille_reader
import poiseuille_functions
from poiseuille_class import PoiseuilleNetwork
import time
import matplotlib.pyplot as plt
import numpy as np
import vtk
from scipy import stats
from vtk.util import numpy_support

from argparse import ArgumentParser


def main(polyline_path, iolet_csv, alpha=0.05, print_option=False, poiseuille_class=None, target_p=None, target_wss = None, k_p=1., k_tau=4.):
    """
    Main function, takes the network and boundary conditions and returns the solved flow/haematocrit/pressure in the network
    Uses Poiseuille's law for the flow, and Pries' empirical model for phase separation
    See Secomb 2017 annual review on blood flow in the microcirculation as a starting point
    :param polyline_path: VTK polyline of the network, each edge contains the radius under string "Radius"
    :param iolet_csv: csv containing boundary node index, pressure, and hct information
    :param alpha: constant for predictor-corrector scheme, see Lorthois et al. NeuroImage 2011, part 1. Lower values lead to more stable solver but solver is then slower to converge
    :return: the PoiseuilleClass of solved network, containing pressure, flowrate, and haematocrits in the network
    """
    poiseuille_class = PoiseuilleNetwork(polyline_path, *poiseuille_reader.read_iolets_csv(iolet_csv))

    solution_converged = False
    iteration = 1
    print(poiseuille_class.iolets)
    if len(poiseuille_class.unknown_BC_nodes) != 0:
        assert target_p != None, ("need target pressures to optimise")
        assert target_wss != None, ("need target wss to optimise")
        poiseuille_class.j, poiseuille_class.apparent_viscosity = poiseuille_functions.solve_conductance(poiseuille_class)
        optimised_pressures = poiseuille_functions.flow_opt_outer_loop(poiseuille_class, poiseuille_reader.read_target_csv(target_wss), poiseuille_reader.read_target_csv(target_p),k_p=k_p, k_tau=k_tau)
        for unknown_node in poiseuille_class.unknown_BC_nodes:
            optimised_pressure = optimised_pressures[unknown_node]
            poiseuille_class.iolets.append([unknown_node, optimised_pressure])

    while solution_converged is False:
        j, current_miu = poiseuille_functions.solve_conductance(poiseuille_class)
        poiseuille_class.apparent_viscosity = current_miu
        poiseuille_class.p = poiseuille_functions.generate_solve_poiseuille_matrix(poiseuille_class, j)
        poiseuille_class.Q = poiseuille_functions.solve_flow(poiseuille_class.p, j, poiseuille_class.edges)
        convergent_bifurcations, divergent_bifurcation = poiseuille_network_functions.bifurcation_types(poiseuille_class.Q, poiseuille_class.bifurcations, poiseuille_class.edges, poiseuille_class.p)
        poiseuille_class.H = poiseuille_functions.solve_haematocrit(poiseuille_class.Q, poiseuille_class, divergent_bifurcation, convergent_bifurcations, poiseuille_class.p)

        if iteration > 1:
            poiseuille_class.Q = [alpha * q_current + (1. - alpha) * q_previous for q_current, q_previous in zip(poiseuille_class.Q, previous_q)]
            poiseuille_class.H = [alpha * h_current + (1. - alpha) * h_previous for h_current, h_previous in zip(poiseuille_class.H, previous_h)]

            poiseuille_class.wss = [4.*mu*flowrate / (np.pi*r**3) for mu, flowrate, r in zip(poiseuille_class.apparent_viscosity, poiseuille_class.Q, poiseuille_class.radius)]

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
    parser.add_argument('--target_p', type=str, default=None, help="CSV file with node number and pressure target at node")
    parser.add_argument('--target_wss', type=str, default=None, help="CSV file with edge number and wss target at edge")
    parser.add_argument('--print_option', type=int, help='Flag to print each iteration information in Poiseuille solver, argparse does not support a bool type, as it returns True for both True and False...', default=0, required=False, choices=[0, 1])
    args = parser.parse_args()

    poiseuille_reader.write_to_vtk(main(args.input_polyline, args.input_iolets, print_option=args.print_option, target_p=args.target_p, target_wss=args.target_wss), args.input_polyline.replace(".vtp", "_poiseuille.vtp"))
