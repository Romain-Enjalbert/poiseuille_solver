import poiseuille_network_functions
import poiseuille_reader


class PoiseuilleNetwork(object):
    def __init__(self, edges_csv, nodes_csv, iolets_csv):
        self.edges, self.radius, self.D, self.h_inlets, self.ROM_choice, self.compression_resistance = poiseuille_reader.read_edges_file(edges_csv)
        self.nodes = poiseuille_reader.read_nodes_file(nodes_csv)
        self.iolets, self.BC = poiseuille_reader.read_iolets_file(iolets_csv)
        self.length, self.H, self.p_0 = poiseuille_reader.process_input(self.edges, self.nodes, self.h_inlets, self.iolets)
        self.equivalent_radius, self.major, self.minor = poiseuille_reader.equivalent_circle_radius_list(4.26, self.radius)
        self.bifurcations, self.bifurcation_neighbour_nodes, self.straights, self.straight_neighbour_nodes = poiseuille_network_functions.new_process_network(self.edges, self.nodes)


if __name__ == "__main__":
    pass
