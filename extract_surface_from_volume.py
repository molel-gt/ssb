import os
import sys

from paraview.simple import *


paraview.simple._DisableFirstRenderCameraReset()

working_dir = sys.argv[1]
tetrxdmf = XDMFReader(FileNames=[os.path.join(working_dir, 'tetr_unscaled.xdmf')])
tetrxdmf.CellArrayStatus = ['name_to_read']

# Properties modified on tetrxdmf
tetrxdmf.GridStatus = ['Grid']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1371, 794]

# get layout
layout1 = GetLayout()

# show data in view
tetrxdmfDisplay = Show(tetrxdmf, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'name_to_read'
name_to_readLUT = GetColorTransferFunction('name_to_read')

# get opacity transfer function/opacity map for 'name_to_read'
name_to_readPWF = GetOpacityTransferFunction('name_to_read')

# trace defaults for the display properties.
tetrxdmfDisplay.Representation = 'Surface'
tetrxdmfDisplay.ColorArrayName = ['CELLS', 'name_to_read']
tetrxdmfDisplay.LookupTable = name_to_readLUT
tetrxdmfDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
tetrxdmfDisplay.SelectOrientationVectors = 'None'
tetrxdmfDisplay.ScaleFactor = 0.5
tetrxdmfDisplay.SelectScaleArray = 'name_to_read'
tetrxdmfDisplay.GlyphType = 'Arrow'
tetrxdmfDisplay.GlyphTableIndexArray = 'name_to_read'
tetrxdmfDisplay.GaussianRadius = 0.025
tetrxdmfDisplay.SetScaleArray = [None, '']
tetrxdmfDisplay.ScaleTransferFunction = 'PiecewiseFunction'
tetrxdmfDisplay.OpacityArray = [None, '']
tetrxdmfDisplay.OpacityTransferFunction = 'PiecewiseFunction'
tetrxdmfDisplay.DataAxesGrid = 'GridAxesRepresentation'
tetrxdmfDisplay.PolarAxes = 'PolarAxesRepresentation'
tetrxdmfDisplay.ScalarOpacityFunction = name_to_readPWF
tetrxdmfDisplay.ScalarOpacityUnitDistance = 0.08271659144605703

# reset view to fit data
renderView1.ResetCamera()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
tetrxdmfDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Extract Surface'
extractSurface1 = ExtractSurface(Input=tetrxdmf)

# show data in view
extractSurface1Display = Show(extractSurface1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
extractSurface1Display.Representation = 'Surface'
extractSurface1Display.ColorArrayName = ['CELLS', 'name_to_read']
extractSurface1Display.LookupTable = name_to_readLUT
extractSurface1Display.OSPRayScaleFunction = 'PiecewiseFunction'
extractSurface1Display.SelectOrientationVectors = 'None'
extractSurface1Display.ScaleFactor = 0.5
extractSurface1Display.SelectScaleArray = 'name_to_read'
extractSurface1Display.GlyphType = 'Arrow'
extractSurface1Display.GlyphTableIndexArray = 'name_to_read'
extractSurface1Display.GaussianRadius = 0.025
extractSurface1Display.SetScaleArray = [None, '']
extractSurface1Display.ScaleTransferFunction = 'PiecewiseFunction'
extractSurface1Display.OpacityArray = [None, '']
extractSurface1Display.OpacityTransferFunction = 'PiecewiseFunction'
extractSurface1Display.DataAxesGrid = 'GridAxesRepresentation'
extractSurface1Display.PolarAxes = 'PolarAxesRepresentation'

# hide data in view
Hide(tetrxdmf, renderView1)

# show color bar/color legend
extractSurface1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# save data
SaveData(os.path.join(working_dir, 'tria_unscaled.xmf'), proxy=extractSurface1, CellDataArrays=['name_to_read'])

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [2.2737555, 2.2737555, 18.23714773910607]
renderView1.CameraFocalPoint = [2.2737555, 2.2737555, 2.5]
renderView1.CameraParallelScale = 4.073073550472726

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).