from collections import namedtuple

from cv2 import triangulatePoints


Facet = namedtuple('Facet', 'indices values')


class Phases:
    def __init__(self, void=0, electrolyte=1, active_material=2):
        self.void = void
        self.electrolyte = electrolyte
        self.active_material = active_material


class SurfaceMarkers:
    def __init__(self, left_cc=1, right_cc=2, insulated=3, active=4, inactive=5):
        self.left_cc = left_cc
        self.right_cc = right_cc
        self.insulated = insulated
        self.active = active
        self.inactive = inactive


class CellTypes:
    def __init__(self, triangle="triangle", tetra="tetra"):
        self.triangle = triangle
        self.tetra = tetra