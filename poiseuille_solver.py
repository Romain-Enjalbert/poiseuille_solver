import poiseuille_network_functions
import poiseuille_reader
import poiseuille_functions
from poiseuille_class import PoiseuilleNetwork
from argparse import ArgumentParser


def main(polyline_path, nodelist, pressure_bc_list, h_bc_list):
    poiseuille_class = PoiseuilleNetwork(polyline_path, nodelist, pressure_bc_list, h_bc_list)

    p_results, Q_results, H_results, miu_results, j_results, rbc_results = [], [], [], [], [], []
    solution_converged = False
    iteration = 1
    alpha = 0.05

    while solution_converged is False:
        j, current_miu = poiseuille_functions.solve_conductance(poiseuille_class)

        p = poiseuille_functions.generate_solve_poiseuille_matrix(poiseuille_class, j)
        Q = poiseuille_functions.solve_flow(p, j, poiseuille_class.edges)

        if iteration == 1:
            Q = Q
        else:
            Q = [alpha * q_current + (1. - alpha) * q_previous for q_current, q_previous in zip(Q, Q_results[-1])]

        convergent_bifurcations, divergent_bifurcation = poiseuille_network_functions.bifurcation_types(Q,
                                                                                                        poiseuille_class.bifurcations,
                                                                                                        poiseuille_class.edges,
                                                                                                        p)
        new_h = poiseuille_functions.solve_haematocrit(Q, poiseuille_class, divergent_bifurcation,
                                                       convergent_bifurcations, p)

        if iteration == 1:
            poiseuille_class.H = new_h
        else:
            poiseuille_class.H = [alpha * h_current + (1. - alpha) * h_previous for h_current, h_previous in
                                  zip(new_h, H_results[-1])]

        rbc_flux = [abs(q * h) for q, h in zip(Q, poiseuille_class.H)]

        j_results.append(j); miu_results.append(current_miu); Q_results.append(Q); H_results.append(poiseuille_class.H); p_results.append(p); rbc_results.append(rbc_flux)

        poiseuille_class.Q = Q; poiseuille_class.p = p

        print(iteration)
        iteration += 1

        if iteration > 2:
            if poiseuille_functions.check_converged(Q_results, error=0.05) is True:
                try:
                    print("Passing convergence check, mass balance...")
                    poiseuille_network_functions.bifurcation_check(divergent_bifurcation, convergent_bifurcations, Q, p,
                                                                   poiseuille_class.edges)
                    poiseuille_network_functions.check_mass_balance(Q, poiseuille_class.H, poiseuille_class.straights,
                                                                    divergent_bifurcation, convergent_bifurcations,
                                                                    upper_bound=1.001, lower_bound=0.999)
                    solution_converged = True
                except:
                    solution_converged = False
    return poiseuille_class


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_polyline', type=str, help='Input polyline with nodes, edges, and radius')
    parser.add_argument('input_iolets', type=str, help='Input csv with iolets')
    args = parser.parse_args()

    nodes, pressure, haematocrits = poiseuille_reader.read_iolets_csv(args.input_iolets)
    poiseuille_reader.write_to_vtk(main(args.input_polyline, nodes, pressure, haematocrits))
