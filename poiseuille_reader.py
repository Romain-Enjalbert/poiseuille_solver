import vtk
from vtk.util import numpy_support


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


def write_to_vtk(poiseuille_class):
    """
    Write result of the network simulation to VTK
    :param poiseuille_class: Contains all the data to write to disk
    :return: No return, but saves a VTP file in current directory called 'poiseuille_vtk.vtp'
    """
    absolute_flowrate = []
    for q in poiseuille_class.Q:
        absolute_flowrate.append(abs(q))

    rbc_flux = []
    for hct, q in zip(poiseuille_class.H, absolute_flowrate):
        flux = hct*q
        rbc_flux.append(flux)

    segments = []
    for edge in poiseuille_class.edges:
        segments.append([edge[1], edge[2]])
    points = []
    for node in poiseuille_class.nodes:
        points.append([node[1], node[2], 0])

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(create_points(points))
    polydata.SetLines(create_cells(segments))

    flowrate_array = numpy_support.numpy_to_vtk(poiseuille_class.Q)
    flowrate_array.SetName('flowrate')
    polydata.GetCellData().AddArray(flowrate_array)

    absolute_flowrate_array = numpy_support.numpy_to_vtk(absolute_flowrate)
    absolute_flowrate_array.SetName('absolute_flowrate')
    polydata.GetCellData().AddArray(absolute_flowrate_array)

    haematocrit_array = numpy_support.numpy_to_vtk(poiseuille_class.H)
    haematocrit_array.SetName('haematocrit')
    polydata.GetCellData().AddArray(haematocrit_array)

    diameter_array = numpy_support.numpy_to_vtk(poiseuille_class.D)
    diameter_array.SetName('diameter')
    polydata.GetCellData().AddArray(diameter_array)

    rbc_flux_array = numpy_support.numpy_to_vtk(rbc_flux)
    rbc_flux_array.SetName('rbc_flux')
    polydata.GetCellData().AddArray(rbc_flux_array)


    length_array = numpy_support.numpy_to_vtk(poiseuille_class.length)
    length_array.SetName('length')
    polydata.GetCellData().AddArray(length_array)

    pressure_array = numpy_support.numpy_to_vtk(poiseuille_class.p)
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
