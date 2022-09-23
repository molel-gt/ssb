PHASES = (0, 1, 2)

phase_key = {
    "void": 0,
    "electrolyte": 1,
    "activematerial": 2,
}

surface_tags = {
    "left_cc": 1,
    "right_cc": 2,
    "insulated": 3,
    "active_area": 4,
    "inactive_area": 5,
}

HFW = 80.4/1768  # micrometers for FIB-SEM per pixel width

# Decimal places in scientific notation
EXP_DIGITS = 4

KAPPA0 = 0.1  # bulk ionic conductivity Siemens/meter