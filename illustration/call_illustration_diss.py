"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 21.12.2021
"""

import numpy as np
from utils import plot_1dv2


def main():
    # print_sod_regime()
    # print_sod_solution()
    # print_shear_layer_regime()
    # print_shear_layer_solution()
    print_shear_layer_distribution()
    # paper_illustrations()
    return 0


def print_sod_regime():
    folder_name = "paper_data/sod1D/regime/"
    x_data = np.load(folder_name + "sod_x.npy")
    n_jump = 2
    # ----- KN=1e-4 ----------
    regime = "1e-4"
    nn_data = np.load(folder_name + "sod_Kn_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "sod_Kn_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "sod_Kn_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', 'NN', 'KnGLL'], legend_pos="upper left",
              name='Regime_labels' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=["-", "o", "^"],
              show_fig=False, xlim=(0, 1), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime", black_first=True)

    # ----- KN=1e-3 ----------
    regime = "1e-3"
    nn_data = np.load(folder_name + "sod_Kn_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "sod_Kn_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "sod_Kn_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', 'NN', 'KnGLL'], legend_pos="upper left",
              name='Regime_labels' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=["-", "o", "^"],
              show_fig=False, xlim=(0, 1), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime", black_first=True)

    # ----- KN=1e-2 ----------
    regime = "1e-2"
    nn_data = np.load(folder_name + "sod_Kn_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "sod_Kn_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "sod_Kn_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', 'NN', 'KnGLL'], legend_pos="upper left",
              name='Regime_labels' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=["-", "o", "^"],
              show_fig=False, xlim=(0, 1), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime label", black_first=True)

    return 0


def print_sod_solution():
    folder_name = "paper_data/sod1D/solution/"
    x_data = np.load(folder_name + "sod_sol_x_1.npy")
    n_jump = 1
    zoom_idx = range(120, 160)

    # ----- KN=1e-4 ----------
    regime = "1e-4"
    s = "1"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper right",
              name='solution_rho' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0, 1), ylim=(.1, 1.05))
    plot_1dv2(xs=[x_data[zoom_idx]],
              ys=[kinetic_data[zoom_idx], ns_data[zoom_idx], nn_data[zoom_idx], kn_gll_data[zoom_idx]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper right",
              name='solution_rho_zoom' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.6, 0.8), ylim=(.1, .5))

    s = "2"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper left",
              name='solution_U' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0, 1), ylim=(-0.05, 0.9))
    plot_1dv2(xs=[x_data[zoom_idx]],
              ys=[kinetic_data[zoom_idx], ns_data[zoom_idx], nn_data[zoom_idx], kn_gll_data[zoom_idx]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper left",
              name='solution_U_zoom' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.6, 0.8), ylim=(-0.05, 0.9))

    s = "3"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper left",
              name='solution_T' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True)  # , xlim=(0, 1), ylim=(.1, 1.05))
    plot_1dv2(xs=[x_data[zoom_idx]],
              ys=[kinetic_data[zoom_idx], ns_data[zoom_idx], nn_data[zoom_idx], kn_gll_data[zoom_idx]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper left",
              name='solution_T_zoom' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.6, 0.8), ylim=(1.24, 2.7))

    # ----- KN=1e-3 ----------
    regime = "1e-3"
    s = "1"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper right",
              name='solution_rho' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0, 1), ylim=(.1, 1.05))
    plot_1dv2(xs=[x_data[zoom_idx]],
              ys=[kinetic_data[zoom_idx], ns_data[zoom_idx], nn_data[zoom_idx], kn_gll_data[zoom_idx]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper right",
              name='solution_rho_zoom' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.6, 0.8), ylim=(.1, .5))

    s = "2"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper left",
              name='solution_U' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0, 1), ylim=(-0.05, 0.9))
    plot_1dv2(xs=[x_data[zoom_idx]],
              ys=[kinetic_data[zoom_idx], ns_data[zoom_idx], nn_data[zoom_idx], kn_gll_data[zoom_idx]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper left",
              name='solution_U_zoom' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.6, 0.8), ylim=(-0.05, 0.9))

    s = "3"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper left",
              name='solution_T' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True)  # , xlim=(0, 1), ylim=(.1, 1.05))
    plot_1dv2(xs=[x_data[zoom_idx]],
              ys=[kinetic_data[zoom_idx], ns_data[zoom_idx], nn_data[zoom_idx], kn_gll_data[zoom_idx]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper left",
              name='solution_T_zoom' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.6, 0.8), ylim=(1.24, 2.7),
              symbolsize=2)

    # ----- KN=1e-2 ----------
    regime = "1e-2"
    s = "1"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper right",
              name='solution_rho' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0, 1), ylim=(.1, 1.05))
    plot_1dv2(xs=[x_data[zoom_idx]],
              ys=[kinetic_data[zoom_idx], ns_data[zoom_idx], nn_data[zoom_idx], kn_gll_data[zoom_idx]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper right",
              name='solution_rho_zoom' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.6, 0.8), ylim=(.1, .5))

    s = "2"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper left",
              name='solution_U' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0, 1), ylim=(-0.05, 0.9))
    plot_1dv2(xs=[x_data[zoom_idx]],
              ys=[kinetic_data[zoom_idx], ns_data[zoom_idx], nn_data[zoom_idx], kn_gll_data[zoom_idx]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper left",
              name='solution_U_zoom' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.6, 0.8), ylim=(-0.05, 0.9))

    s = "3"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper left",
              name='solution_T' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True)  # , xlim=(0, 1), ylim=(.1, 1.05))
    plot_1dv2(xs=[x_data[zoom_idx]],
              ys=[kinetic_data[zoom_idx], ns_data[zoom_idx], nn_data[zoom_idx], kn_gll_data[zoom_idx]],
              labels=['Kinetic', 'Navier-Stokes', 'NN', 'KnGLL'], legend_pos="upper left",
              name='solution_T_zoom' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.6, 0.8), ylim=(1.24, 2.7))

    return 0


def print_shear_layer_regime():
    folder_name = "paper_data/shear_layer_2d/regime/"
    save_folder_name = "illustration/shear_layer_2d"
    x_data = np.load(folder_name + "layer_f_regime_x_1.npy")
    n_jump = 4
    # ----- KN=1tau ----------
    regime = "1"
    nn_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', 'NN', 'KnGLL'], legend_pos="upper left",
              name='Regime_labels_shear_layer' + regime, log=False, folder_name=save_folder_name,
              linetypes=["-", "o", "^"],
              show_fig=False, xlim=(-0.5, 0.5), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime", black_first=True)

    # ----- KN=1tau ----------
    regime = "2"
    nn_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', 'NN', 'KnGLL'], legend_pos="upper left",
              name='Regime_labels_shear_layer' + regime, log=False, folder_name=save_folder_name,
              linetypes=["-", "o", "^"],
              show_fig=False, xlim=(-0.5, 0.5), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime", black_first=True)

    # ----- KN=50tau ----------
    regime = "3"
    nn_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', 'NN', 'KnGLL'], legend_pos="upper left",
              name='Regime_labels_shear_layer' + regime, log=False, folder_name=save_folder_name,
              linetypes=["-", "o", "^"],
              show_fig=False, xlim=(-0.5, 0.5), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime label",
              black_first=True)

    return 0


def print_shear_layer_solution():
    folder_name = "paper_data/shear_layer_2d/solution/"
    save_folder_name = "illustration/shear_layer_2d"

    n_jump = 1

    # ----- t=\tau----------
    regime = "1"
    # density
    x_data = np.load(folder_name + "layer_n_t_x_" + regime + ".npy")
    nn_data = np.load(folder_name + "layer_n_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_n_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_n_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_rho_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(-0.06, 0.06), ylim=(.8, 1.2))
    # U_1
    nn_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_u1_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_1$", black_first=True, xlim=(-0.06, 0.06), ylim=(-0.015, .25))
    # U_2
    nn_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_u2_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_2$", black_first=True, xlim=(-0.06, 0.06), ylim=(-1.05, 1.05))
    # T
    nn_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_T_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(-0.06, 0.06), ylim=(.48, 1.37))

    # ----- t=10\tau----------
    n_jump = 2
    regime = "2"
    x_lim = (-0.1, .1)
    # rho
    x_data = np.load(folder_name + "layer_n_t_x_" + regime + ".npy")
    nn_data = np.load(folder_name + "layer_n_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_n_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_n_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='shear_rho_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=x_lim, ylim=(.65, 1.35))
    # U_1
    nn_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='shear_u1_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_1$", black_first=True, xlim=x_lim, ylim=(-0.01, .21))
    # U_2
    nn_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_u2_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_2$", black_first=True, xlim=x_lim, ylim=(-1.05, 1.05))
    # T
    nn_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_T_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=x_lim, ylim=(.48, 1.27))

    # ----- t=50\tau----------
    n_jump = 7
    regime = "3"
    x_lim = (-0.5, .5)
    # rho
    x_data = np.load(folder_name + "layer_n_t_x_" + regime + ".npy")
    nn_data = np.load(folder_name + "layer_n_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_n_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_n_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='shear_rho_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=x_lim, ylim=(.65, 1.37))
    # U_1
    nn_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='shear_u1_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_1$", black_first=True, xlim=x_lim, ylim=(-0.01, .22))
    # U_2
    nn_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_u2_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_2$", black_first=True, xlim=x_lim, ylim=(-1.05, 1.05))
    # T
    nn_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_T_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=x_lim, ylim=(.48, 1.19))

    return 0


def print_shear_layer_distribution():
    folder_name = "paper_data/shear_layer_2d/solution/"
    save_folder_name = "illustration/shear_layer_2d"
    n_jump = 1
    x_lim = (-5, 5)
    # t=tau
    regime = "1"
    x_data = np.load(folder_name + "layer_f_t_x_" + regime + ".npy")
    nn_data = np.load(folder_name + "layer_f_t_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_f_t_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_f_t_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_f_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$f$", black_first=True, xlim=x_lim, ylim=(0, .08))

    # t=10tau
    regime = "2"
    x_data = np.load(folder_name + "layer_f_t_x_" + regime + ".npy")
    nn_data = np.load(folder_name + "layer_f_t_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_f_t_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_f_t_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_f_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$f$", black_first=True, xlim=x_lim, ylim=(0, .08))

    # t=50tau
    regime = "3"
    x_data = np.load(folder_name + "layer_f_t_x_" + regime + ".npy")
    nn_data = np.load(folder_name + "layer_f_t_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_f_t_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_f_t_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_f_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$f$", black_first=True, xlim=x_lim, ylim=(0, 0.095))
    return 0


def paper_illustrations():
    # ---- illustrate moment dynamics
    x_dat = np.load('paper_data/sod1D/X.npy')
    y_dat = np.load('paper_data/sod1D/Y.npy')
    z_dat = np.load('paper_data/sod1D/Z.npy')
    iter_dat = np.load('paper_data/sod1D/I.npy')

    # 1) moment - Sod test case
    moment_u = x_dat[:3, :]
    # normalize everything
    u_normal = moment_u / moment_u[0, :]
    scatter_plot_2d_N2(x_in=u_normal[1:, :].reshape((u_normal.shape[1], 2)), z_in=iter_dat[0], show_fig=False,
                       log=False,
                       folder_name='illustration', color_map=1, name='moment_dynamics', label_x='velocity',
                       label_y='temperature', title='macroscopic variables over time')

    # 2) gradient - sod test case
    grad_u = x_dat[3:6, :]
    scatter_plot_3d(xyz_in=grad_u.reshape((u_normal.shape[1], 3)), color_in=iter_dat[0], show_fig=False, log=False,
                    folder_name='illustration', color_map=1, name='grad_moment_dynamics',
                    title='gradient macroscopic variables over time', lim_x=(-1, 10),
                    lim_y=(0, 10), lim_z=(0, 10))
    scatter_plot_2d(x_in=grad_u[1:, :].reshape((u_normal.shape[1], 2)), z_in=iter_dat[0], show_fig=False, log=False,
                    folder_name='illustration', color_map=1, name='grad_moment_dynamics2D', label_x=r"$\nabla_x U$",
                    label_y=r"$\nabla_x T$", title='gradient macroscopic variables over time')

    # II ----- illustrate generated paper_data
    x_dat = np.load('paper_data/generator1D/X.npy')
    y_dat = np.load('paper_data/generator1D/Y.npy')
    # 1) moment - generated
    moment_u = x_dat[:3, :]
    # normalize everything
    u_normal = moment_u / moment_u[0, :]
    scatter_plot_2d_N2(x_in=u_normal[1:, :].reshape((u_normal.shape[1], 2)), z_in=np.zeros(shape=(u_normal.shape[1])),
                       show_fig=False, log=False,
                       folder_name='illustration', color_map=1, name='moment_dynamics_genData', label_x='velocity',
                       label_y='temperature', title='macroscopic variables (generated)')

    # 2) gradient - generated
    grad_u = x_dat[3:6, :]
    scatter_plot_3d(xyz_in=grad_u.reshape((u_normal.shape[1], 3)),
                    color_in=np.zeros(shape=(u_normal.shape[1])), show_fig=False, log=False,
                    folder_name='illustration', color_map=1, name='grad_moment_dynamics_genData',
                    title='gradient macroscopic variables (generated)', lim_x=(-1, 10),
                    lim_y=(0, 10), lim_z=(0, 10))
    scatter_plot_2d(x_in=grad_u[1:, :].reshape((u_normal.shape[1], 2)),
                    z_in=np.zeros(shape=(u_normal.shape[1])), show_fig=False, log=False,
                    folder_name='illustration', color_map=1, name='grad_moment_dynamics2D_genData',
                    label_x=r"$\nabla_x U$", label_y=r"$\nabla_x T$",
                    title='gradient macroscopic variables (generated)')
    return 0


if __name__ == '__main__':
    main()
