from collections import namedtuple


Facet = namedtuple('Facet', 'indices values')


class Phases:
    def __init__(self, void=0, electrolyte=1, active_material=2):
        self.void = void
        self.electrolyte = electrolyte
        self.active_material = active_material


class SurfaceMarkers:
    def __init__(self, left_cc=1, right_cc=2, insulated=3, active=4, inactive=5, am_se_interface=6):
        self.left_cc = left_cc
        self.right_cc = right_cc
        self.insulated = insulated
        self.active = active
        self.inactive = inactive
        self.am_se_interface = am_se_interface


class CellTypes:
    def __init__(self, triangle="triangle", tetra="tetra", line="line"):
        self.triangle = triangle
        self.tetra = tetra
        self.line = line
