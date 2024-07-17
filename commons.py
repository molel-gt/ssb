from collections import namedtuple


Facet = namedtuple('Facet', 'indices values')

# studies
STUDIES = (
           'contact_loss_lma',
           'contact_loss_ref',
           'full_cell_2d',
           'lmb_3d_cc',
           'reaction_distribution',
           'separator_mechanics',
           )

class Study(str):
    def __init__(self, name):
        if name not in STUDIES:
            raise ValueError(f"{name} not in STUDIES\n")
        self._name = name

    @property
    def name(self):
        return self._name

# markers for subdomains and boundaries
class Markers:
    def __init__(self):
        help = "Markers typical of a solid-state battery"

    @property
    def void(self):
        return 255

    @property
    def negative_cc(self):
        return 1

    @property
    def negative_am(self):
        return 2

    @property
    def electrolyte(self):
        return 3

    @property
    def positive_am(self):
        return 4

    @property
    def positive_cc(self):
        return 5

    @property
    def negative_cc_v_negative_am(self):
        return 6

    @property
    def negative_am_v_electrolyte(self):
        return 7

    @property
    def electrolyte_v_positive_am(self):
        return 8

    @property
    def positive_am_v_positive_cc(self):
        return 9

    @property
    def left(self):
        return 10

    @property
    def middle(self):
        return 11

    @property
    def right(self):
        return 12

    @property
    def insulated_negative_cc(self):
        return 13

    @property
    def insulated_negative_am(self):
        return 14

    @property
    def insulated_electrolyte(self):
        return 15

    @property
    def insulated_positive_am(self):
        return 16

    @property
    def insulated_positive_cc(self):
        return 17

    @property
    def insulated(self):
        return 18

    @property
    def three_d_cc_well(self):
        return 19

    @property
    def external(self):
        return 111


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
