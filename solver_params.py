gamg = {
    "pc_gamg_type": 'agg',
    "pc_gamg_threshold": 0.01,
    "pc_gamg_repartition": True,
    "pc_gamg_aggressive_coarsening": 4,
    "pc_gamg_aggressive_square_graph": 1,
    "pc_gamg_agg_nsmooths": 1,
    "pc_gamg_coarse_eq_limit": 10000,
    "pc_gamg_parallel_coarse_grid_solver": True,
    "pc_gamg_eigenvalues": [1e-4, 5],
    "pc_gamg_use_sa_esteig": True,
}

boomeramg = {
    "pc_hypre_type": "boomeramg",
    "pc_hypre_boomeramg_coarsen_type": "hmis",
    "pc_hypre_boomeramg_interp_type": "ext",
    "pc_hypre_boomeramg_strong_threshold": 0.7,
    "pc_hypre_boomeramg_agg_nl": 2,
    "pc_hypre_boomeramg_agg_num_paths": 5,
    "pc_hypre_boomeramg_truncfactor": 0.75,
    "pc_hypre_boomeramg_smooth_num_levels": 2,
    # "pc_hypre_boomeramg_smooth_type": "pilut",
    "pc_hypre_boomeramg_vec_interp_variant": 3,
    "pc_hypre_boomeramg_nodal_coarsen": 4,
    "pc_hypre_boomeramg_max_iter": 4,
    "pc_hypre_boomeramg_max_levels": 5,
    "pc_hypre_boomeramg_relax_type_all": "chebyshev",
    'pc_hypre_boomeramg_p_max': 4,
    "pc_hypre_boomeramg_print_statistics": 2,
    "pc_hypre_boomeramg_cycle_type": 'v',
}

AMG_TYPES = {
    "gamg": gamg,
    "hypre": boomeramg,
}

LINESEARCH = {
    "snes_linesearch_type": "none", #"basic",
    'snes_linesearch_monitor': None,
    'snes_monitor': None,
}
