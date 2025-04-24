#!/usr/bin/env
import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import pandas as pd

import plot_opts, utils
plt.rcParams.update(plot_opts.params)

galvanostatic = "galvanostatic"
potentiostatic = "potentiostatic"
hold_voltage = "hold_voltage"
cyclic_voltammetry = "cyclic_voltammetry"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reaction distribution')
    parser.add_argument('--results_folder', help='parent folder containing mesh folder', required=True)
    parser.add_argument('--cycling_mode', help='parent folder containing mesh folder', required=True)
    args = parser.parse_args()
    stats_json = os.path.join(args.results_folder, "stats.json")
    stats = pd.read_csv(stats_json)

    figures_dir = os.path.join("figures", "reaction_distribution", args.mode)
    utils.make_dir_if_missing(figures_dir)

    time = stats["t [s]"]

    # cell voltage and open-circuit potential
    V_ocp = stats["V (ocp) (avg) [V]"]
    V_cell = stats["u (avg) right [V]"]
    fig, ax = plt.subplots()
    ax.plot(time[1:], V_ocp[1:], label=r'$V_{\mathrm{OCP}}$')
    ax.plot(time[1:], V_cell[1:], 'r', label=r'$V_{\mathrm{cell}}$')
    ax.set_box_aspect(1)
    ax.set_ylabel(r'Voltage [V]')
    ax.set_xlabel(r'time [s]')
    ax.legend(loc="upper center")
    ax.yaxis.set_major_formatter(plticker.FuncFormatter(lambda x, pos: f"{float(x):.3f}"))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'cell-voltage-cv.eps'), format='eps')

    # ohmic and kinetic losses
    eta_ohm = stats["u (avg) right [V]"] - ocv_chen2020(stats["c surf (avg) (normalized)"]) - stats["surface overpotential (avg) [V]"]
    eta_surf = stats["surface overpotential (avg) [V]"]
    fig, ax = plt.subplots()
    ax.plot(time[1:], mV * eta_surf[1:], 'r--', label=r'$\eta_s$')
    ax.plot(time, mV * eta_ohm, 'b-.', label=r'$\eta_{\Omega}$')
    ax.set_box_aspect(1)
    ax.set_ylabel('overpotential [mV]')
    ax.set_xlabel('time [s]')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'eta-s-avg.eps'), format='eps')

    # normalized avg bulk and surface concentration of lithium
    c_bulk_avg = stats["c (avg) (normalized)"]
    c_surf_avg = stats["c surf (avg) (normalized)"]
    fig, ax = plt.subplots()
    ax.plot(time, c_bulk_avg, 'k', label='volume')
    ax.plot(time, c_surf_avg, 'k--', label='surface')
    ax.set_ylabel(r'$\tilde{c}$')
    ax.set_xlabel('time [s]')
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'c-normalized.eps'), format='eps')

    # normalized stdev of surface concentration of lithium
    c_surf_stdev = stats["c surf (stdev) (normalized)"]
    fig, ax = plt.subplots()
    ax.plot(time, c_surf_stdev, 'b')
    ax.set_ylabel(r'$\sigma(\tilde{c}_{\mathrm{surf}})$')
    ax.set_xlabel('time [s]')
    ax.set_box_aspect(1)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'c-surf-stdev-normalized.eps'), format='eps')

    # interfacial errors
    error_u = stats["I_interface error norm (normalized)"]
    error_c = stats["I_interface error norm c (normalized)"]
    fig, ax = plt.subplots()
    ax.plot(time, error_u, 'k', label='potential')
    ax.plot(time, error_c, 'k-.', label='concentration')
    ax.set_box_aspect(1)
    ax.set_ylabel(r'$\tilde{e}_2$')
    ax.set_xlabel('time [s]')
    ax.legend()
    ax.axhline(y=0.05, linestyle='--', color='r', linewidth=0.15)
    ax.yaxis.set_major_formatter(plticker.FuncFormatter(set_dp))
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'rmse-evolution.eps'), format='eps')

    # normalized stdev of current density at charge transfer boundary
    i_se_am_stdev = np.abs(stats["i (stdev) se/am [A/m2]"]/stats["i (avg) se/am [A/m2]"])
    fig, ax = plt.subplots()
    ax.plot(time, i_se_am_stdev, 'k')
    ax.set_box_aspect(1)
    ax.set_ylabel(r'$\tilde{\sigma}_{\mathrm{SE/AM}}$')
    ax.set_xlabel('time [s]')
    ax.grid(color='gray', linewidth=0.05)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'i-stdev-se-am-normalized-evolution.eps'), format='eps')

    # normalized stdev of current density at other interfaces
    i_left_stdev = np.abs(stats["i (stdev) left [A/m2]"]/stats["i (avg) left [A/m2]"])
    i_right_stdev = np.abs(stats["i (stdev) right [A/m2]"]/stats["i (avg) right [A/m2]"])
    fig, ax = plt.subplots()
    ax.plot(time, i_left_stdev, 'r', label='Left')
    ax.plot(time, i_right_stdev, 'b--', label='Right')
    ax.set_box_aspect(1)
    ax.set_ylabel(r'$\tilde{\sigma}$')
    ax.set_xlabel('time [s]')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, 'i-stdev-left-n-right-normalized-evolution.eps'), format='eps')

    if args.cycle_mode == cyclic_voltammetry:
        I_right = stats["I right [A]"]
        fig, ax = plt.subplots()
        ax.plot(V_cell[1:]-3.5, 1e9 * np.abs(I_right[1:]), 'k')
        ax.set_box_aspect(1)
        ax.set_ylabel(r'I [nA]')
        ax.set_xlabel('Overpotential [V]')
        ax.legend()
        # ax.set_ylim([0, 0.002])
        # ax.set_ylim([0, 150])
        # ax.yaxis.set_major_formatter(plticker.FuncFormatter(set_dp))
        # ax.legend(title='$K_r$', bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, 'cyclic-voltammetry.eps'), format='eps')
