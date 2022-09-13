
import os
import sys
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Legacy VTK Reader'
working_dir = sys.argv[1]
porous1vtk = LegacyVTKReader(FileNames=[os.path.join(working_dir, 'porous.1.vtk')])

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1371, 794]

# get layout
layout1 = GetLayout()

# show data in view
porous1vtkDisplay = Show(porous1vtk, renderView1, 'UnstructuredGridRepresentation')

# trace defaults for the display properties.
porous1vtkDisplay.Representation = 'Surface'
porous1vtkDisplay.ColorArrayName = [None, '']
porous1vtkDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
porous1vtkDisplay.SelectOrientationVectors = 'None'
porous1vtkDisplay.ScaleFactor = 20.1
porous1vtkDisplay.SelectScaleArray = 'None'
porous1vtkDisplay.GlyphType = 'Arrow'
porous1vtkDisplay.GlyphTableIndexArray = 'None'
porous1vtkDisplay.GaussianRadius = 1.0050000000000001
porous1vtkDisplay.SetScaleArray = [None, '']
porous1vtkDisplay.ScaleTransferFunction = 'PiecewiseFunction'
porous1vtkDisplay.OpacityArray = [None, '']
porous1vtkDisplay.OpacityTransferFunction = 'PiecewiseFunction'
porous1vtkDisplay.DataAxesGrid = 'GridAxesRepresentation'
porous1vtkDisplay.PolarAxes = 'PolarAxesRepresentation'
porous1vtkDisplay.ScalarOpacityUnitDistance = 1.340787906659877

# reset view to fit data
renderView1.ResetCamera()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Extract Surface'
extractSurface1 = ExtractSurface(Input=porous1vtk)

# hide data in view
Hide(porous1vtk, renderView1)

# show data in view
extractSurface1Display = Show(extractSurface1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
extractSurface1Display.Representation = 'Surface'
extractSurface1Display.ColorArrayName = [None, '']
extractSurface1Display.OSPRayScaleFunction = 'PiecewiseFunction'
extractSurface1Display.SelectOrientationVectors = 'None'
extractSurface1Display.ScaleFactor = 20.1
extractSurface1Display.SelectScaleArray = 'None'
extractSurface1Display.GlyphType = 'Arrow'
extractSurface1Display.GlyphTableIndexArray = 'None'
extractSurface1Display.GaussianRadius = 1.0050000000000001
extractSurface1Display.SetScaleArray = [None, '']
extractSurface1Display.ScaleTransferFunction = 'PiecewiseFunction'
extractSurface1Display.OpacityArray = [None, '']
extractSurface1Display.OpacityTransferFunction = 'PiecewiseFunction'
extractSurface1Display.DataAxesGrid = 'GridAxesRepresentation'
extractSurface1Display.PolarAxes = 'PolarAxesRepresentation'

# reset view to fit data
renderView1.ResetCamera()

# hide data in view
Hide(porous1vtk, renderView1)

# update the view to ensure updated data information
renderView1.Update()

# save data
SaveData(os.path.join(working_dir, 'surface.vtk'), proxy=extractSurface1)