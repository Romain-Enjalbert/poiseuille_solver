import poiseuille_network_functions
import poiseuille_reader
import poiseuille_functions
from poiseuille_class import PoiseuilleNetwork
from argparse import ArgumentParser


def main(polyline_path, bc_node_list, pressure_bc_list, h_bc_list, alpha=0.05):
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

    p_results, q_results, h_results, miu_results, j_results, rbc_results = [], [], [], [], [], []
    solution_converged = False
    iteration = 1

    while solution_converged is False:
        j, current_miu = poiseuille_functions.solve_conductance(poiseuille_class)
        p = poiseuille_functions.generate_solve_poiseuille_matrix(poiseuille_class, j)
        q = poiseuille_functions.solve_flow(p, j, poiseuille_class.edges)

        if iteration == 1:
            q = q
            convergent_bifurcations, divergent_bifurcation = poiseuille_network_functions.bifurcation_types(q, poiseuille_class.bifurcations, poiseuille_class.edges, p)
            new_h = poiseuille_functions.solve_haematocrit(q, poiseuille_class, divergent_bifurcation, convergent_bifurcations, p)
            poiseuille_class.H = new_h
        else:
            q = [alpha * q_current + (1. - alpha) * q_previous for q_current, q_previous in zip(q, q_results[-1])]
            convergent_bifurcations, divergent_bifurcation = poiseuille_network_functions.bifurcation_types(q, poiseuille_class.bifurcations, poiseuille_class.edges, p)
            new_h = poiseuille_functions.solve_haematocrit(q, poiseuille_class, divergent_bifurcation, convergent_bifurcations, p)
            poiseuille_class.H = [alpha * h_current + (1. - alpha) * h_previous for h_current, h_previous in zip(new_h, h_results[-1])]

        rbc_flux = [abs(q * h) for q, h in zip(q, poiseuille_class.H)]

        j_results.append(j); miu_results.append(current_miu); q_results.append(q);
        h_results.append(poiseuille_class.H); p_results.append(p); rbc_results.append(rbc_flux)
        poiseuille_class.Q = q; poiseuille_class.p = p

        print(iteration)
        iteration += 1

        if iteration > 2:
            if poiseuille_network_functions.check_converged(q_results, error=0.05):
                print("Passing convergence check, checking mass balance...")
                assert poiseuille_network_functions.check_bifurcations(divergent_bifurcation, convergent_bifurcations, q, p, poiseuille_class.edges) is True
                assert poiseuille_network_functions.check_mass_balance(q, poiseuille_class.H, poiseuille_class.straights, divergent_bifurcation, convergent_bifurcations) is True
                solution_converged = True
    return poiseuille_class


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_polyline', type=str, help='Input polyline with nodes, edges, and radius')
    parser.add_argument('input_iolets', type=str, help='Input csv with iolets')
    args = parser.parse_args()

    nodes, pressure, haematocrits = poiseuille_reader.read_iolets_csv(args.input_iolets)
    poiseuille_reader.write_to_vtk(main(args.input_polyline, nodes, pressure, haematocrits))
