#!/usr/bin/env python3
import json
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_opts, utils

plt.rcParams.update(plot_opts.params)

data_cols = ['I left [A]', 'I interface [A]', 'I right [A]', 'solve time [s]', 'Positive Wa', 'Kr', 'dofs', 'n_procs']


def json_data(cols=data_cols, kinetics_type="butler_volmer"):
    process = subprocess.Popen('find output/tertiary_current/40-40-75 -name simulation.json',
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    rows = []

    output, error = process.communicate()
    sim_jsons = output.split("\n")

    for f in sim_jsons:
        if kinetics_type not in f:
            continue
        with open(f, "r") as fp:
            data = json.load(fp)
            row_data = {k: data[k] for k in cols}
        rows.append(row_data)

    return rows


if __name__ == '__main__':
    rows_data = json_data()
    df = pd.DataFrame(rows_data)
    workdir = "output/tertiary_current/40-40-75/summary-plots"
    utils.make_dir_if_missing(workdir)

    current_ratios_plots = os.path.join(workdir, "current-ratios.eps")
    solve_time_nprocs_plots = os.path.join(workdir, "solve-time-nprocs.eps")
    solve_time_dofs_plots = os.path.join(workdir, "solve-time-dofs.eps")
    fig, ax = plt.subplots()
    ax.plot(df['dofs'], df['I left [A]']/df['I interface [A]'], 'o')
    ax.set_xlabel('DOFs')
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(current_ratios_plots, bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.plot(df['n_procs'], df['dofs'] * df['n_procs'] / df['solve time [s]'], 'o')
    ax.set_xlabel('nprocs')
    ax.set_ylabel('dofs/s')
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(solve_time_nprocs_plots, bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.plot(df['dofs'], df['dofs'] * df['n_procs'] / df['solve time [s]'], 'o')
    ax.set_xlabel('DOFs')
    ax.set_ylabel('dofs/s')
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(solve_time_dofs_plots, bbox_inches='tight')
