import poiseuille_network_functions
import numpy as np
from scipy.spatial.distance import euclidean
import vtk
from vtk.util import numpy_support


class PoiseuilleNetwork(object):
    """
    A class representing a microvascular network

    ...
Methods
----------
read_polydata(self, polydata_path)
    Read the polydata of the network into the class
    The polydata contains the graph (edges, nodes) and the radius at every edge

create_h_inlets(self, h_list, BC_nodes)
    Creates the h_inlets for the instance attribute

process_inlets(self)
    Initiliases some of the instance attributes for the network class

"""

    def __init__(self, polydata_path, BC_nodes, BC_p, BC_haematocrit):
        """
        Parameters
        ----------
        polydata_path : str
            path to the .vtp network to create the class of the network
        BC_nodes : list
            a list containing the node indices that are boundary nodes
        BC_p : list
            a list containing the pressure value (pa) of the boundary nodes, indexed by BC_nodes list
        BC_haematocrit :
            a list containing the haematcrit [0,1.] at the boundary nodes, indexed by BC_nodes list

        Attributes
        ----------
        edges : list
            a nested list containing [edge #, node 1 index, node 2 index] at every entry
        nodes : list
            a nested list containing [node #, x_coord, y_coord, z_coord] at every entry
        radius : list
            a list containing the radius (in metres) at every entry, by edge index
        D : list
            a list containing the diameter (in um) at every entry, by edge index
        iolets : list
            a nested list containing [BC node index, pressure value of BC node in pa] at every entry
        h_inlets : list
            a nested list containing [BC edge index, haematocrit at index ranging from 0. to 1.] at every entry
        length : list
            a list containing the length of each edge (in metres) at every entry, by edge index
        H : list
            a list containing the haematocrit at every edge, by edge index, before first iteration initialises at 0.2 everywhere, except at edges defined by h_inlets
        p_0 : list
            a list initialising the pressure values at every node at 0, except where the nodes are defined by iolets
        p : list
            a list cintaning the pressure values at every node after Poiseuille solution, by node index
            doesn't exist upon initialisation but updated after each iteration
        wss : list
            a list containing the wall shear stress at every edge, by edge index, as per Fry et al. Microcirculation 2012, in SI units
            doesn't exist upon initialisation but updated after each iteration
        apparent_viscosity : list
            a list containing the apparent viscosity at every edge, by edge index, as per Secomb 2017 annual review, in SI units
            doesn't exist upon initialisation but updated after each iteration
        bifurcations : list
            a nested list containing [edge 1 #, edge 2 #, edge 3 #, central node #] at every entry, i.e. all three edges in the bifurcation and the central node to the bifurcation
        bifurcation_neighbour_nodes : list
            a nested list containing np.array([central node #, neighbour  node 1 #, neighbour  node 2 #, neighbour  node 3 #]) at every entry, i.e. the central node to the bifurcation, and the three other nodes at the end of the edges
        straights : list
            a nested list containing [edge 1 #, edge 2 #, central node #] at every entry, i.e. the two edges in a straight and the central node
        straight_neighbour_nodes : list
            a nested list containing np.array([central node #, neighbour  node 1 #, neighbour  node 2 #]) at every entry, i.e. the central node to the straight, and the two other nodes at the end of the edges
        """
        self.read_polydata(polydata_path)
        self.iolets = [[n, p] for n, p in zip(BC_nodes, BC_p)]
        self.create_h_inlets(BC_haematocrit, BC_nodes)
        self.process_inlets()
        self.bifurcations, self.bifurcation_neighbour_nodes, self.straights, self.straight_neighbour_nodes = poiseuille_network_functions.process_network(self.edges, self.nodes)

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

    def create_h_inlets(self, h_list, BC_nodes):
        self.h_inlets = []
        for h, n in zip(h_list, BC_nodes):
            for count, edge in enumerate(self.edges):
                if n == int(edge[1]) or n == int(edge[2]):
                    self.h_inlets.append([count, h])
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
