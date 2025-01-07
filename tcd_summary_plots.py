#!/usr/bin/env python3
import json
import os
import subprocess

import matplotlib.pyplot as plt
from matplotlib import ticker

import numpy as np
import pandas as pd

import plot_opts, utils

plt.rcParams.update(plot_opts.params)

data_cols = ['I left [A]', 'I interface [A]', 'I right [A]', 'solve time [s]', 'Positive Wa', 'Kr', 'dofs', 'n_procs', 'polynomial approximation order (p)', 'penalty parameter (gamma)']


def json_data(cols=data_cols, kinetics_type="butler_volmer", refined=False):
    process = subprocess.Popen('find output/tertiary_current/40-40-75 -name simulation.json',
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    rows = []

    output, error = process.communicate()
    sim_jsons = output.split("\n")

    for f in sim_jsons:
        if kinetics_type not in f:
            continue
        if not refined and 'refined' in f:
            continue
        with open(f, "r") as fp:
            try:
                data = json.load(fp)
                row_data = {k: data.get(k) for k in cols}
            except json.decoder.JSONDecodeError:
                print(f"Could not decode {f}")
        if row_data.get('polynomial approximation order (p)') is None:
            continue
        if row_data.get('polynomial approximation order (p)') == 4:
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
    weak_scaling_plot = os.path.join(workdir, "weak-scaling.eps")
    strong_scaling_plot = os.path.join(workdir, "strong-scaling.eps")
    fig, ax = plt.subplots()
    ax.semilogx(df['dofs'], df['I interface [A]']/df['I left [A]'], 'o')
    ax.set_xlabel('DOFs')
    ax.set_ylabel(r'$\frac{I_c}{I_{\phi}}$')
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(current_ratios_plots, bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.plot(df['n_procs'], df['dofs'] / df['n_procs'] / df['solve time [s]'], 'o')
    ax.set_xlabel('nprocs')
    ax.set_ylabel('dofs/s')
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(solve_time_nprocs_plots, bbox_inches='tight')

    fig, ax = plt.subplots()
    ax.plot(df['dofs'], df['dofs'] / df['n_procs'] / df['solve time [s]'], 'o')
    ax.set_xlabel('DOFs')
    ax.set_ylabel('dofs/s')
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(solve_time_dofs_plots, bbox_inches='tight')

    strong_scaling_df = df[np.logical_and(np.isclose(df['dofs'], 3193836, atol=1e4), np.isclose(df['Positive Wa'], 1))]
    strong_scaling_df = strong_scaling_df[np.isclose(strong_scaling_df['penalty parameter (gamma)'], 1e-4)]
    strong_scaling_df.sort_values("n_procs", inplace=True)
    strong_scaling_df.reset_index(drop=True, inplace=True)
    print(strong_scaling_df)
    fig, ax = plt.subplots()
    solve_time_1proc = strong_scaling_df[np.isclose(strong_scaling_df['n_procs'], 1)]["solve time [s]"].values[0]
    ax.plot(strong_scaling_df['n_procs'], solve_time_1proc/strong_scaling_df['solve time [s]'], 'kx-', label='Actual')
    ax.plot([1, np.max(strong_scaling_df['n_procs'])], [1, np.max(strong_scaling_df['n_procs'])], 'r--', label='Ideal')
    ax.set_xlabel('No. of Processors')
    ax.set_ylabel("Speedup")
    plt.xscale('log', base=2)
    ax.set_xlim([1, np.max(strong_scaling_df['n_procs']) + 1])
    ax.set_ylim([1, np.max(strong_scaling_df['n_procs']) + 1])
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    ax.legend()
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(strong_scaling_plot, bbox_inches='tight')
