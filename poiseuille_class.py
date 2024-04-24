import poiseuille_network_functions
import poiseuille_reader
import csv
import numpy as np
from scipy.spatial.distance import euclidean
import vtk
from vtk.util import numpy_support


class PoiseuilleNetwork(object):
    """
    edges: at every entry contains [edge #, node 1 index, node 2 index] OK
    radius: at every entry contains radius in m indexed by edges OK
    D: at every entry contains diameter in um indexed by edges OK
    h_inlets: every entry contains [edge #, h BC value (0 - 1. range)] OK
    nodes: List (node_#, x_coord, y_coord) indexed by node number OK
    iolets: Return list (node_#, pressure_bc) OK
    BC: type of BC pressure or velocity, no longer used I think but would have been useful to still have
    length: Length of each channel indexed by edge
    H: initial hct values indexed by edge
    p_0: pressure list with BCs and 0s everywhere else
    bifurcations:
    bifurcation_neighbour_nodes:
    straights:
    straight_neighbour_nodes:
    """
    def __init__(self, polydata_path, csv_iolets_path):
        self.read_polydata(polydata_path)
        self.read_csv_iolets(csv_iolets_path)
        self.process_inlets()
        self.bifurcations, self.bifurcation_neighbour_nodes, self.straights, self.straight_neighbour_nodes = poiseuille_network_functions.new_process_network(self.edges, self.nodes)

    def read_polydata(self, polydata_path):
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(polydata_path)
        reader.Update()
        polydata = reader.GetOutput()

        cleanFilter = vtk.vtkCleanPolyData()
        cleanFilter.SetInputData(polydata)
        cleanFilter.Update()
        polydata = cleanFilter.GetOutput()

        edges = numpy_support.vtk_to_numpy(polydata.GetLines().GetData())
        edges = np.reshape(edges, (int(len(edges)/3), 3))
        edges = [[line[1], line[2]] for line in edges]
        self.edges = []
        for count, edge in enumerate(edges):
            self.edges.append([count, edge[0], edge[1]])
        self.edges = np.array(self.edges)
        nodes = numpy_support.vtk_to_numpy(polydata.GetPoints().GetData())
        self.nodes = []
        for count, node in enumerate(nodes):
            self.nodes.append([count, node[0], node[1], node[2]])
        radius = numpy_support.vtk_to_numpy(polydata.GetCellData().GetArray("Radius"))
        self.radius = [r*10**-6 for r in radius]
        self.D = [2.*r for r in radius]
        return

    def read_csv_iolets(self, csv_iolets):
        self.h_inlets = []
        self.iolets = []
        with open(csv_iolets, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                self.iolets.append([int(row[0]), float(row[1])])
                for count, edge in enumerate(self.edges):
                    if int(row[0]) == int(edge[1]) or int(row[0]) == int(edge[2]):
                        self.h_inlets.append([count, float(row[2])])
                        continue
        return

    def process_inlets(self):
        self.length = []
        for edge in self.edges:
            node_1, node_2 = self.nodes[edge[1]], self.nodes[edge[2]]
            node_1 = [node_1[1], node_1[2], node_1[3]]
            node_2 = [node_2[1], node_2[2], node_2[3]]
            self.length.append(euclidean(node_1, node_2)*10**-6)

        self.p_0 = []
        for i in range(len(self.nodes)):
            self.p_0.append(0)
        for pressure in self.iolets:
            self.p_0[pressure[0]] = pressure[1]

        self.H = []
        for i in range(len(self.edges)):
            self.H.append(0.0)
        for input_h in self.h_inlets:
            self.H[input_h[0]] = input_h[1]
        return


if __name__ == "__main__":
    pass
