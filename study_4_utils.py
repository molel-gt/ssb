
import os

import configs


def meshfile_subfolder_path(name_of_study="study_4", eps=0.5,  dimensions="10-10-10", max_resolution=1):
    """
    standardization of meshfile_subfolder path for use by mesh builder and FEA solvers
    """
    return os.path.join(configs.get_configs()['LOCAL_PATHS']['data_dir'], name_of_study, str(eps), dimensions, str(max_resolution))