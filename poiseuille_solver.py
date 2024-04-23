import poiseuille_network_functions
import poiseuille_reader
import poiseuille_functions
from poiseuille_class import PoiseuilleNetwork
from argparse import ArgumentParser


def sign(number):
    if number > 0:
        return 1
    elif number < 0:
        return -1
    else:
        assert 0

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('input_edges', type=str, help='Input csv with edges')
    parser.add_argument('input_nodes', type=str, help='Input csv with nodes')
    parser.add_argument('input_iolets', type=str, help='Input csv with iolets')
    parser.add_argument('h_solver', type=str, help='Haematocrit solver, choice between pries or updated_pries', choices=["pries", "updated_pries"])
    parser.add_argument('--write_vtk', type=bool, help='Write data to vtk', default=None, required=False)
    args = parser.parse_args()

    poiseuille_class = poiseuille_reader.get_network_class(args.input_edges, args.input_nodes, args.input_iolets)

    p_results, Q_results, H_results, miu_results, j_results, rbc_results = [], [], [], [], [], []
    solution_converged = False
    iteration = 1
    alpha = 0.05

    while solution_converged is False:
        j, current_miu = poiseuille_functions.solve_conductance(poiseuille_class)

        p = poiseuille_functions.generate_solve_poiseuille_matrix(poiseuille_class, j)
        Q = poiseuille_functions.solve_flow(p, j, poiseuille_class.edges)

        if iteration < 6:
            Q = Q
        elif iteration >= 6:
            Q = [alpha * q_current + (1. - alpha) * q_previous for q_current, q_previous in zip(Q, Q_results[-1])]
        else:
            assert 0

        convergent_bifurcations, divergent_bifurcation = poiseuille_network_functions.bifurcation_types(Q, poiseuille_class.bifurcations, poiseuille_class.edges, p)
        new_h = poiseuille_functions.solve_haematocrit(Q, poiseuille_class, divergent_bifurcation, convergent_bifurcations, p, args.h_solver)

        if iteration == 1:
            poiseuille_class.H = new_h
        elif iteration > 1:
            poiseuille_class.H = [alpha * h_current + (1. - alpha) * h_previous for h_current, h_previous in zip(new_h, H_results[-1])]
        else:
            assert 0

        rbc_flux = [abs(q*h) for q, h in zip(Q, poiseuille_class.H)]

        j_results.append(j); miu_results.append(current_miu); Q_results.append(Q); H_results.append(poiseuille_class.H); p_results.append(p); rbc_results.append(rbc_flux)

        print(iteration)
        iteration += 1

        if iteration > 2:
            poiseuille_reader.write_to_vtk(args.write_vtk, poiseuille_class.edges, poiseuille_class.nodes, Q_results[-1], H_results[-1], poiseuille_class.D, p_results[-1], poiseuille_class, poiseuille_class.length)
            if poiseuille_functions.check_converged(Q_results, error=0.05) is True:
                try:
                    print("Passing convergence check, mass balance...")
                    poiseuille_network_functions.bifurcation_check(divergent_bifurcation, convergent_bifurcations, Q, p, poiseuille_class.edges)
                    poiseuille_network_functions.check_mass_balance(Q, poiseuille_class.H, poiseuille_class.straights, divergent_bifurcation, convergent_bifurcations, upper_bound=1.001, lower_bound=0.999)
                    solution_converged = True
                except:
                    solution_converged = False
    poiseuille_reader.write_to_vtk(args.write_vtk, poiseuille_class.edges, poiseuille_class.nodes, Q_results[-1], H_results[-1], poiseuille_class.D, p_results[-1], poiseuille_class, poiseuille_class.length)
