import csv
import numpy as np
import vtk
import os
from vtk.util import numpy_support
import pickle
from poiseuille_class import PoiseuilleNetwork


def get_network_class(edges_csv, nodes_csv, iolets_csv):
    if "network.pkl" in os.listdir(os.getcwd()):
        network_class = pickle.load(open("network.pkl", "rb"))
    else:
        network_class = PoiseuilleNetwork(edges_csv, nodes_csv, iolets_csv)
        pickle.dump(network_class, open("network.pkl", "wb"))
    return network_class


def process_inlets(edges_csv, nodes_csv, iolets_csv):
    """
    Governing function to process input CSV files and create relevant lists from them
    :param edges_csv: csv file containing edges. Format (edge_#, node_0, node_1) can contain a fourth entry to import hct in that edge for first loop/iolet
    :param nodes_csv: csv file containing nodes. Format (node_#, x_coord, y_coord)
    :param iolets_csv: csv file containing pressure values at iolers. Format (node_#, pressure_bc)
    :return: All relevant lists
    """
    edges_reader, r_reader, D_reader, Hinlets_reader, ROM_choice_reader, compression_resistance_reader = read_edges_file(edges_csv)
    nodes_reader = read_nodes_file(nodes_csv)
    iolets_reader, bc_reader = read_iolets_file(iolets_csv)
    l_reader, H_reader, p_0_reader = process_input(edges_reader, nodes_reader, Hinlets_reader, iolets_reader)
    equivalent_radius_list = equivalent_circle_radius_list(4.26, r_reader)
    return edges_reader, r_reader, D_reader, Hinlets_reader, nodes_reader, iolets_reader, l_reader, H_reader, p_0_reader, ROM_choice_reader, compression_resistance_reader, equivalent_radius_list, bc_reader


def read_edges_file(edges_file):
    """
    Processes the edges csv file
    :param edges_file: csv file containing edges. Format (edge_#, node_0, node_1) can contain a fourth entry to import hct in that edge for first loop/iolet
    :return: Relevant lists indexed by edges to be read from csv. Except reader_h_inlet with (edge_#, hct) at every row, usually only for inlet
    """
    edges_reader = []
    radius_reader = []
    reader_h_inlets = []
    reader_D = []
    reader_ROM_choice = []
    reader_compression_resistance = []
    with open(edges_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            # print row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip(), row[5].strip()
            edges_reader.append([int(row[0].strip()), int(row[1].strip()), int(row[2].strip())])
            radius_reader.append(float(row[3].strip())*10**-6)
            reader_D.append(float(row[3].strip())*2)
            if str(row[4].strip()) == "True":
                reader_ROM_choice.append(True)
            elif str(row[4].strip()) == "False":
                reader_ROM_choice.append(False)
            else:
                assert 0
            if str(row[5].strip()) == "True":
                reader_compression_resistance.append(True)
            elif str(row[5].strip()) == "False":
                reader_compression_resistance.append(False)
            else:
                assert 0
            if len(row) == 7:
                reader_h_inlets.append([int(row[0].strip()), float(row[6].strip())])
    return np.array(edges_reader), radius_reader, reader_D, reader_h_inlets, reader_ROM_choice, reader_compression_resistance


def read_nodes_file(nodes_file):
    """
    Process the nodes csv file
    :param nodes_file: csv file containing nodes. Format (node_#, x_coord, y_coord)
    :return: List (node_#, x_coord, y_coord) indexed by node number
    """
    nodes_reader = []
    with open(nodes_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            nodes_reader.append([int(row[0]), float(row[1]), float(row[2])])
    return (np.array(nodes_reader))


def read_iolets_file(iolets_file):
    """
    Process the iolets csv file
    :param iolets_file: csv file containing pressure values at iolers. Format (node_#, pressure_bc)
    :return: Return list (node_#, pressure_bc)
    """
    reader_iolets = []
    with open(iolets_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            reader_iolets.append([int(row[0]), float(row[1])])
            if len(row) >= 3:
                boundary_condition = str(row[2]) # This here reads the node that is an inlet and determines whether the value associated is for pressure BC or flowrate
    return reader_iolets, boundary_condition


def process_input(input_edges, input_nodes, input_Hinlets, input_iolets):
    """
    Processes the available lists to return additional information
    :param input_edges: (edge_#, node_0, node_1) indexed by edge number
    :param input_nodes: (node_#, x_coord, y_coord) indexed by node number
    :param input_Hinlets: (edge_#, hct_value)
    :param input_iolets: (node_#, pressure_bc)
    :return: Length of each channel indexed by edge, initial hct values indexed by edge, and pressure list with BCs and 0s everywhere else, indexed by node
    """
    reader_l = []
    reader_H = []
    reader_p0 = []
    for edge in input_edges:
        delta_x = abs(input_nodes[edge[1], 1] - input_nodes[edge[2], 1])
        delta_y = abs(input_nodes[edge[1], 2] - input_nodes[edge[2], 2])
        length_squared = delta_x ** 2 + delta_y ** 2
        length = length_squared ** 0.5
        reader_l.append(length * 10 ** (-6))
    for i in range(len(input_edges)):
        reader_H.append(0.2)
    for input_h in input_Hinlets:
        reader_H[input_h[0]] = input_h[1]
    for i in range(len(input_nodes)):
        reader_p0.append(0)
    for pressure in input_iolets:
        reader_p0[pressure[0]] = pressure[1]
    return reader_l, reader_H, reader_p0


def equivalent_circle_radius_list(aspect_ratio, radii):
    return_equivalent_circle_radius_list = []
    major_list = []
    minor_list = []
    for radius in radii:
        circle_perimeter = np.pi*radius*2
        minor = np.linspace(0.1e-6, radius, 1000)
        for minor_r in minor:
            major_r = minor_r * aspect_ratio
            h = (major_r - minor_r) ** 2 / (major_r + minor_r) ** 2
            ellipse_perimeter = np.pi * (major_r + minor_r) * (1 + 0.25 * h + h * h / 64 + h * h * h / 256 + h * h * h * h / 16384) #  https://www.mathsisfun.com/geometry/ellipse-perimeter.html
            if abs(ellipse_perimeter - circle_perimeter) / circle_perimeter < 1. / 100:
                major_axis = major_r
                minor_axis = minor_r
                minor_list.append(minor_axis)
                major_list.append(major_axis)
                break
        equivalent_radius = (minor_axis*major_axis)**0.5
        return_equivalent_circle_radius_list.append(equivalent_radius)
    return return_equivalent_circle_radius_list, major_list, minor_list


def create_points(array):
    vtk_points = vtk.vtkPoints()
    for point_id, point in enumerate(array):
        vtk_points.InsertPoint(point_id, point)
    return vtk_points


def create_cells(array):
    vtk_cells = vtk.vtkCellArray()
    for segment in array:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, segment[0])
        line.GetPointIds().SetId(1, segment[1])
        vtk_cells.InsertNextCell(line)
    return vtk_cells


def write_to_vtk(boolean, input_edges, nodes, flowrates, haematocrit, diameter, pressure, poiseuille_class, length):
    """
    Write result of the network simulation to VTK
    :param boolean: True or False, True writes the file
    :param input_edges: (edge_#, node_0, node_1) at every row indexed by edge number
    :param nodes: (node_#, x_coord, y_coord) at every row indexed by node number
    :param flowrates: Flowrate of last iteration indexed by edge number
    :param haematocrit: HCT of last iteration indexed by edge number
    :param diameter: Diameter indexed by edge number
    :param pressure: Pressure of last iteration indexed by node number
    :return: No return, but saves a VTP file in current directory called 'poiseuille_vtk.vtp'
    """
    if boolean is True:
        absolute_flowrate = []
        for q in flowrates:
            absolute_flowrate.append(abs(q))

        rbc_flux = []
        for hct, q in zip(haematocrit, absolute_flowrate):
            flux = hct*q
            rbc_flux.append(flux)

        segments = []
        for edge in input_edges:
            segments.append([edge[1], edge[2]])
        points = []
        for node in nodes:
            points.append([node[1], node[2], 0])

        is_tumour = []
        for rom_c, compression_c in zip(poiseuille_class.ROM_choice, poiseuille_class.compression_resistance):
            if rom_c is True or compression_c is True:
                is_tumour.append(1)
            else:
                is_tumour.append(0)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(create_points(points))
        polydata.SetLines(create_cells(segments))

        flowrate_array = numpy_support.numpy_to_vtk(flowrates)
        flowrate_array.SetName('flowrate')
        polydata.GetCellData().AddArray(flowrate_array)

        absolute_flowrate_array = numpy_support.numpy_to_vtk(absolute_flowrate)
        absolute_flowrate_array.SetName('absolute_flowrate')
        polydata.GetCellData().AddArray(absolute_flowrate_array)

        haematocrit_array = numpy_support.numpy_to_vtk(haematocrit)
        haematocrit_array.SetName('haematocrit')
        polydata.GetCellData().AddArray(haematocrit_array)

        diameter_array = numpy_support.numpy_to_vtk(diameter)
        diameter_array.SetName('diameter')
        polydata.GetCellData().AddArray(diameter_array)

        rbc_flux_array = numpy_support.numpy_to_vtk(rbc_flux)
        rbc_flux_array.SetName('rbc_flux')
        polydata.GetCellData().AddArray(rbc_flux_array)

        is_tumour_array = numpy_support.numpy_to_vtk(is_tumour)
        is_tumour_array.SetName('is_tumour')
        polydata.GetCellData().AddArray(is_tumour_array)

        length_array = numpy_support.numpy_to_vtk(length)
        length_array.SetName('length')
        polydata.GetCellData().AddArray(length_array)

        pressure_array = numpy_support.numpy_to_vtk(pressure)
        pressure_array.SetName('pressure')
        polydata.GetPointData().AddArray(pressure_array)

        file_name = 'poiseuille_vtk.vtp'
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(file_name)
        writer.SetInputData(polydata)
        writer.Write()
    return


if __name__ == "__main__":
    pass
