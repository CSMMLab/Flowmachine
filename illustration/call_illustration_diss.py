"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 21.12.2021
"""

import numpy as np
import pandas as pd

from utils import plot_1dv2, load_density_function  # , scatter_plot_2d_N2,  scatter_plot_3d, scatter_plot_2d


def main():
    # ------ 0. Data Generator -------
    # print_densities()
    # print_macroscopic_var()
    # ------ 1. Sod 1D ---------------
    print_sod_regime()
    print_sod_solution()
    # ------ 2. Shear Layer 2D -------
    # print_shear_layer_regime()
    # print_shear_layer_solution()
    # print_shear_layer_distribution()
    # ------ 3. Cylinder 2D ----------
    # print_cylinder_solution()
    # paper_illustrations()
    return 0


def print_densities():
    save_folder_name = "illustration/generator"

    """ Skript that performs all illustration for the paper done by Steffen"""
    [v_x, weights, f_kinetic] = load_density_function("paper_data/pdfs.csv")
    f_ns = pd.read_csv("paper_data/fns.csv").to_numpy()
    grads = pd.read_csv("paper_data/gradient_w.csv").to_numpy()
    params = pd.read_csv("paper_data/paras.csv").to_numpy()
    conservative_variables = pd.read_csv("paper_data/w.csv").to_numpy()

    # ----- Ilustrate Upwing Merging of two pdf -----
    v_x = np.reshape(v_x, newshape=(v_x.shape[1]))

    plot_1dv2(xs=[v_x],
              ys=[f_kinetic[0, :].reshape(len(f_kinetic[0]), 1), f_kinetic[1, :].reshape(len(f_kinetic[0]), 1),
                  f_kinetic[2, :].reshape(len(f_kinetic[0]), 1), f_ns[:, 0].reshape(len(f_kinetic[0]), 1)],
              labels=['left cell', 'right cell', 'interface', 'BGK reconstruction'], name='maxwell_fusion', log=False,
              folder_name=save_folder_name, linetypes=['-', '--', 'o', '-.'], show_fig=False, xlim=(-5, 5),
              ylim=(0, 0.5), xlabel=r"$v$", ylabel=r"$f(v)$", black_first=True)
    plot_1dv2(xs=[v_x],
              ys=[f_kinetic[3, :].reshape(len(f_kinetic[0]), 1), f_kinetic[4, :].reshape(len(f_kinetic[0]), 1),
                  f_kinetic[5, :].reshape(len(f_kinetic[0]), 1), f_ns[:, 1].reshape(len(f_kinetic[0]), 1)],
              labels=['left cell', 'right cell', 'interface', 'BGK reconstruction'], name='bimodal_fusion', log=False,
              folder_name=save_folder_name, linetypes=['-', '--', 'o', '-.'], show_fig=False, xlim=(-5, 5),
              ylim=(0, 0.6), xlabel=r"$v$", ylabel=r"$f(v)$", black_first=True)
    plot_1dv2(xs=[v_x],
              ys=[f_kinetic[0, :].reshape(len(f_kinetic[0]), 1), f_kinetic[3, :].reshape(len(f_kinetic[0]), 1),
                  f_kinetic[8, :].reshape(len(f_kinetic[0]), 1)], labels=['Maxwellian', 'bimodal', 'anisotropic'],
              name='Entropy_Sampling', log=False, folder_name=save_folder_name, linetypes=None, show_fig=False,
              xlim=(-5, 5), ylim=(0, 0.75), xlabel=r"$v$", ylabel=r"$f(v)$", black_first=True)
    return 0


def print_macroscopic_var():
    save_folder_name = "illustration/generator"

    # ---- illustrate moment dynamics
    w_data_generator = np.load('paper_data/generator1D/generated_macroscopic_variables_normalized.npy')
    w_data_generator_d = np.load('paper_data/generator1D/generated_macroscopic_variables_normalized_gradients.npy')
    sod_data_generator = np.load('paper_data/generator1D/sod_macroscopic_variables_normalized.npy')
    sod_data_generator_d = np.load('paper_data/generator1D/sod_macroscopic_variables_gradients.npy')

    plot_1dv2(xs=[w_data_generator[:, 0]], ys=[w_data_generator[:, 1]], labels=None, name='generated_moments',
              log=False, folder_name=save_folder_name, linetypes=['o'], show_fig=False, xlim=(-1.5, 1.5),
              ylim=(0, 1), xlabel=r"$U$", ylabel=r"$T$", black_first=False)
    plot_1dv2(xs=[sod_data_generator[:, 0]], ys=[sod_data_generator[:, 1]], labels=None, name='sod_moments',
              log=False, folder_name=save_folder_name, linetypes=['o'], show_fig=False, xlim=(-1.5, 1.5),
              ylim=(0, 1), xlabel=r"$U$", ylabel=r"$T$", black_first=False)
    plot_1dv2(xs=[w_data_generator_d[:, 0]], ys=[w_data_generator_d[:, 1]], labels=None, name='generated_moments_grad',
              log=False, folder_name=save_folder_name, linetypes=['o'], show_fig=False, xlim=(-10, 10),
              ylim=(-10, 10), xlabel=r"$\nabla U$", ylabel=r"$\nabla  T$", black_first=False)
    plot_1dv2(xs=[sod_data_generator_d[:, 0]], ys=[sod_data_generator_d[:, 1]], labels=None, name='sod_moments_grad',
              log=False, folder_name=save_folder_name, linetypes=['o'], show_fig=False, xlim=(-10, 10),
              ylim=(-10, 10), xlabel=r"$\nabla  U$", ylabel=r"$\nabla T$", black_first=False)

    return 0


def print_sod_regime():
    folder_name = "paper_data/sod1D/regime/"
    x_data = np.load(folder_name + "sod_x.npy")
    n_jump = 4
    # ----- KN=1e-4 ----------
    regime = "1e-4"
    nn_data = np.load(folder_name + "sod_Kn_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "sod_Kn_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "sod_Kn_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', r'$\rm{N}_\theta$', r'$\rm{Kn}_{\rm{GLL}}$'], legend_pos="upper left",
              name='Regime_labels' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=["-", "o", "^"],
              show_fig=False, xlim=(0, 1), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime", black_first=True,
              font_size=20, yticks=[0, 1])

    # ----- KN=1e-3 ----------
    regime = "1e-3"
    nn_data = np.load(folder_name + "sod_Kn_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "sod_Kn_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "sod_Kn_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', r'$\rm{N}_\theta$', r'$\rm{Kn}_{\rm{GLL}}$'], legend_pos="upper left",
              name='Regime_labels' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=["-", "o", "^"],
              show_fig=False, xlim=(0, 1), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime", black_first=True,
              font_size=20, yticks=[0, 1])

    # ----- KN=1e-2 ----------
    regime = "1e-2"
    nn_data = np.load(folder_name + "sod_Kn_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "sod_Kn_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "sod_Kn_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', r'$\rm{N}_\theta$', r'$\rm{Kn}_{\rm{GLL}}$'], legend_pos="upper left",
              name='Regime_labels' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=["-", "o", "^"],
              show_fig=False, xlim=(0, 1), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime", black_first=True,
              font_size=20, yticks=[0, 1])

    return 0


def print_sod_solution():
    folder_name = "paper_data/sod1D/solution/"
    x_data = np.load(folder_name + "sod_sol_x_1.npy")
    n_jump = 1

    # ----- KN=1e-4 ----------
    regime = "1e-4"
    s = "1"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0, 1), ylim=(.1, 1.05),
              font_size=20)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.58, 0.67), ylim=(.2, .5),
              font_size=20)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.4, 0.5), ylim=(.47, .7),
              font_size=20)

    s = "2"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_U' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0, 1), ylim=(-0.05, 0.9),
              font_size=20)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="lower right",
              name='solution_U_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.4, 0.6), ylim=(0.5, 0.85),
              font_size=20)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_U_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.25, 0.35), ylim=(-0.001, 0.25),
              font_size=20)

    s = "3"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_T' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True,
              font_size=20)  # , xlim=(0, 1), ylim=(.1, 1.05))
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_T_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.44, 0.55), ylim=(1.22, 1.4),
              font_size=20)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_T_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.58, 0.68), ylim=(1.22, 2.65),
              font_size=20)

    # ----- KN=1e-3 ----------
    regime = "1e-3"
    s = "1"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0, 1), ylim=(.1, 1.05),
              font_size=20)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.55, 0.68), ylim=(.2, .5),
              font_size=20)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.7, 0.85), ylim=(.122, .24),
              font_size=20)

    s = "2"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_U' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0, 1), ylim=(-0.05, 0.9),
              font_size=20)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="lower left",
              name='solution_U_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.56, 0.76), ylim=(0.8, 0.88),
              font_size=20, xticks=[0.57, 0.62, 0.67, 0.72, 0.77])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_U_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.27, 0.33), ylim=(-0.001, 0.15),
              font_size=20, xticks=[0.28, 0.3, 0.32], yticks=[0.0, 0.05, 0.1, 0.15])

    s = "3"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_T' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True,
              font_size=20)  # , xlim=(0, 1), ylim=(.1, 1.05))
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_T_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.44, 0.55), ylim=(1.22, 1.4),
              font_size=20)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_T_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.58, 0.68), ylim=(1.22, 2.65),
              font_size=20)

    # ----- KN=1e-2 ----------
    regime = "1e-2"
    s = "1"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0, 1), ylim=(.1, 1.05),
              font_size=20)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="lower left",
              name='solution_rho_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.2, 0.35), ylim=(.85, 1.01),
              font_size=20, yticks=[0.85, 0.9, 0.95, 1.0])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.7, 0.9), ylim=(.122, .24),
              font_size=20, xticks=[0.70, 0.75, 0.80, 0.85, 0.90])

    s = "2"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_U' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0, 1), ylim=(-0.05, 0.9),
              font_size=20)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="lower left",
              name='solution_U_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.56, 0.76), ylim=(0.8, 0.88),
              font_size=20, xticks=[0.57, 0.62, 0.67, 0.72, 0.77])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_U_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.27, 0.33), ylim=(-0.001, 0.15),
              font_size=20, xticks=[0.28, 0.3, 0.32], yticks=[0.0, 0.05, 0.1, 0.15])

    s = "3"
    nn_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NN_" + s + ".npy")
    kn_gll_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_KnGLL_" + s + ".npy")
    kinetic_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_Kinetic_" + s + ".npy")
    ns_data = np.load(folder_name + "sod_sol_Kn_" + regime + "_NS_" + s + ".npy")

    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_T' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True,
              font_size=20)  # , xlim=(0, 1), ylim=(.1, 1.05))
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="lower left",
              name='solution_T_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.2, 0.35), ylim=(1.75, 2.01),
              font_size=20)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_T_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.7, 0.9), ylim=(1.6, 2.65),
              font_size=20)

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
              labels=['true', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='Regime_labels_shear_layer' + regime, log=False, folder_name=save_folder_name,
              linetypes=["-", "o", "^"],
              show_fig=False, xlim=(-0.5, 0.5), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime", black_first=True)

    # ----- KN=1tau ----------
    regime = "2"
    nn_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='Regime_labels_shear_layer' + regime, log=False, folder_name=save_folder_name,
              linetypes=["-", "o", "^"],
              show_fig=False, xlim=(-0.5, 0.5), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime", black_first=True)

    # ----- KN=50tau ----------
    regime = "3"
    nn_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
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


def print_cylinder_solution():
    folder_name = "paper_data/cylinder_2d/solution/"
    save_folder_name = "illustration/cylinder_2d"

    n_jump = 1

    # ----- kn=1e-2----------
    regime = "2"
    x_lim = (-6, -1)
    # density
    x_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_x.npy")
    nn_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_kinetic.npy")
    ns_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='cylinder_rho_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=x_lim, ylim=(0, 14))
    # U velocity
    x_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_x.npy")
    nn_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_kinetic.npy")
    ns_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="lower left", linetypes=["-", "--", "o"],
              name='cylinder_u_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_1$", black_first=True, xlim=x_lim, ylim=(0, 5))
    # T
    x_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_x.npy")
    nn_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_kinetic.npy")
    ns_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='cylinder_T_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=x_lim, ylim=(0, 10))

    # ----- kn=1e-3----------
    regime = "3"
    x_lim = (-6, -1)
    # density
    x_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_x.npy")
    nn_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_kinetic.npy")
    ns_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='cylinder_rho_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=x_lim, ylim=(0, 14))
    # U velocity
    x_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_x.npy")
    nn_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_kinetic.npy")
    ns_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="lower left", linetypes=["-", "--", "o"],
              name='cylinder_u_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_1$", black_first=True, xlim=x_lim, ylim=(0, 5))
    # T
    x_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_x.npy")
    nn_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_kinetic.npy")
    ns_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='cylinder_T_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=x_lim, ylim=(0, 10))

    return 0


if __name__ == '__main__':
    main()
