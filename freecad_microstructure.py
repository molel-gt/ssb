#!/usr/bin/python                                                                          

App.newDocument()
# App.setActiveDocument("Unnamed")
# App.ActiveDocument=App.getDocument("Unnamed")
# Gui.ActiveDocument=Gui.getDocument("Unnamed")
# Gui.activeDocument().activeView().viewDefaultOrientation()
### End command Std_New
# Gui.runCommand('Std_OrthographicCamera',1)
### Begin command Std_SaveAs
# Gui.SendMsgToActiveView("SaveAs")
App.getDocument("Unnamed").saveAs(u"/home/lmolel/work/ssb/output/segmentation/half-cell.FCStd")
### End command Std_SaveAs
### Begin command Std_Workbench
# Gui.activateWorkbench("MeshWorkbench")
import PartDesignGui
### End command Std_Workbench
### Begin command Std_Import
import Mesh
from freecad import module_io
module_io.OpenInsertObject("Mesh", "/home/lmolel/work/ssb/output/segmentation/voids.stl", "insert", "Unnamed")

### End command Std_Import
# Gui.runCommand('Mesh_Evaluation',0)
### Begin command Std_Save
# Gui.SendMsgToActiveView("Save")
App.getDocument("Unnamed").save()
### End command Std_Save
# Gui.Selection.addSelection('Unnamed','voids')
# Gui.runCommand('Mesh_Export',0)
### Begin command Std_Workbench
# Gui.activateWorkbench("PartWorkbench")
### End command Std_Workbench
### Begin command Part_ShapeFromMesh
import Part
App.getDocument('Unnamed').addObject('Part::Feature', 'voids001')
__shape__ = Part.Shape()
__shape__.makeShapeFromMesh(FreeCAD.getDocument('Unnamed').getObject('voids').Mesh.Topology, 0.100000, False)
FreeCAD.getDocument('Unnamed').getObject('voids001').Shape = __shape__
FreeCAD.getDocument('Unnamed').getObject('voids001').purgeTouched()
del __shape__
### End command Part_ShapeFromMesh
### Begin command Std_Save
# Gui.SendMsgToActiveView("Save")
App.getDocument("Unnamed").save()
### End command Std_Save
