"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 21.12.2021
"""
import numpy as np
import pandas as pd

from utils import plot_1dv2, load_density_function, plot_cylinder_colorbars, beautify_img


def main():
    # ------ 0. Data Generator -------
    print_densities()
    print_macroscopic_var()
    # ------ 1. Sod 1D ---------------
    print_sod_regime()
    print_sod_solution()
    # ------ 2. Shear Layer 2D -------
    # print_shear_layer_regime()
    # print_shear_layer_solution()
    # print_shear_layer_distribution()
    # ------ 3. Cylinder 2D ----------
    print_cylinder_solution()
    # paper_illustrations()
    beautify_old_img()
    return 0


def beautify_old_img():
    # plotting colorbards
    plot_cylinder_colorbars()
    fontsize = 20

    name = "cylinder_rgkngll_kn2"
    img_path = "paper_data/cylinder_2d/img/" + name + ".png"
    beautify_img(load_name=img_path, folder_name="illustration/cylinder_2d", name=name, xlabel=r"$x_1$",
                 ylabel=r"$x_2$", cbar="pred", font_size=fontsize)
    name = "cylinder_rgkngll_kn2_c001"
    img_path = "paper_data/cylinder_2d/img/" + name + ".png"
    beautify_img(load_name=img_path, folder_name="illustration/cylinder_2d", name=name, xlabel=r"$x_1$",
                 ylabel=r"$x_2$", cbar="pred", font_size=fontsize)
    name = "cylinder_rgkngll_kn3"
    img_path = "paper_data/cylinder_2d/img/" + name + ".png"
    beautify_img(load_name=img_path, folder_name="illustration/cylinder_2d", name=name, xlabel=r"$x_1$",
                 ylabel=r"$x_2$", cbar="pred", font_size=fontsize)
    name = "cylinder_rgkngll_kn3_c001"
    img_path = "paper_data/cylinder_2d/img/" + name + ".png"
    beautify_img(load_name=img_path, folder_name="illustration/cylinder_2d", name=name, xlabel=r"$x_1$",
                 ylabel=r"$x_2$", cbar="pred", font_size=fontsize)
    name = "cylinder_rgnn_kn2"
    img_path = "paper_data/cylinder_2d/img/" + name + ".png"
    beautify_img(load_name=img_path, folder_name="illustration/cylinder_2d", name=name, xlabel=r"$x_1$",
                 ylabel=r"$x_2$", cbar="pred", font_size=fontsize)
    name = "cylinder_rgnn_kn3"
    img_path = "paper_data/cylinder_2d/img/" + name + ".png"
    beautify_img(load_name=img_path, folder_name="illustration/cylinder_2d", name=name, xlabel=r"$x_1$",
                 ylabel=r"$x_2$", cbar="pred", font_size=fontsize)
    name = "cylinder_rgref_kn2"
    img_path = "paper_data/cylinder_2d/img/" + name + ".png"
    beautify_img(load_name=img_path, folder_name="illustration/cylinder_2d", name=name, xlabel=r"$x_1$",
                 ylabel=r"$x_2$", cbar="pred", font_size=fontsize)
    name = "cylinder_rgref_kn3"
    img_path = "paper_data/cylinder_2d/img/" + name + ".png"
    beautify_img(load_name=img_path, folder_name="illustration/cylinder_2d", name=name, xlabel=r"$x_1$",
                 ylabel=r"$x_2$", cbar="pred", font_size=fontsize)
    name = "cylinder_t_kn2"
    img_path = "paper_data/cylinder_2d/img/" + name + ".png"
    beautify_img(load_name=img_path, folder_name="illustration/cylinder_2d", name=name, xlabel=r"$x_1$",
                 ylabel=r"$x_2$", cbar="tmp", font_size=fontsize)
    name = "cylinder_t_kn3"
    img_path = "paper_data/cylinder_2d/img/" + name + ".png"
    beautify_img(load_name=img_path, folder_name="illustration/cylinder_2d", name=name, xlabel=r"$x_1$",
                 ylabel=r"$x_2$", cbar="tmp", font_size=fontsize)
    name = "cylinder_u_kn2"
    img_path = "paper_data/cylinder_2d/img/" + name + ".png"
    beautify_img(load_name=img_path, folder_name="illustration/cylinder_2d", name=name, xlabel=r"$x_1$",
                 ylabel=r"$x_2$", cbar="vel", font_size=fontsize)
    name = "cylinder_u_kn3"
    img_path = "paper_data/cylinder_2d/img/" + name + ".png"
    beautify_img(load_name=img_path, folder_name="illustration/cylinder_2d", name=name, xlabel=r"$x_1$",
                 ylabel=r"$x_2$", cbar="vel", font_size=fontsize)

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

    fontsize = 20
    markersize = 4
    plot_1dv2(xs=[v_x],
              ys=[f_kinetic[0, :].reshape(len(f_kinetic[0]), 1), f_kinetic[1, :].reshape(len(f_kinetic[0]), 1),
                  f_kinetic[2, :].reshape(len(f_kinetic[0]), 1), f_ns[:, 0].reshape(len(f_kinetic[0]), 1)],
              labels=['left cell', 'right cell', 'interface', 'BGK reconstruction'], name='maxwell_fusion', log=False,
              marker_size=markersize, folder_name=save_folder_name, linetypes=['-', '--', 'o', '-.'], show_fig=False,
              xlim=(-5, 5),
              font_size=fontsize, ylim=(0, 0.5), xlabel=r"$v$", ylabel=r"$f(v)$", black_first=True)
    plot_1dv2(xs=[v_x],
              ys=[f_kinetic[3, :].reshape(len(f_kinetic[0]), 1), f_kinetic[4, :].reshape(len(f_kinetic[0]), 1),
                  f_kinetic[5, :].reshape(len(f_kinetic[0]), 1), f_ns[:, 1].reshape(len(f_kinetic[0]), 1)],
              labels=['left cell', 'right cell', 'interface', 'BGK reconstruction'], name='bimodal_fusion', log=False,
              folder_name=save_folder_name, linetypes=['-', '--', 'o', '-.'], show_fig=False, xlim=(-5, 5),
              marker_size=markersize, font_size=fontsize, ylim=(0, 0.6), xlabel=r"$v$", ylabel=r"$f(v)$",
              black_first=True)
    plot_1dv2(xs=[v_x],
              ys=[f_kinetic[0, :].reshape(len(f_kinetic[0]), 1), f_kinetic[3, :].reshape(len(f_kinetic[0]), 1),
                  f_kinetic[8, :].reshape(len(f_kinetic[0]), 1)], labels=['Maxwellian', 'bimodal', 'anisotropic'],
              name='Entropy_Sampling', log=False, folder_name=save_folder_name, linetypes=None, show_fig=False,
              marker_size=markersize, font_size=fontsize, xlim=(-5, 5), ylim=(0, 0.75), xlabel=r"$v$", ylabel=r"$f(v)$",
              black_first=True)
    return 0


def print_macroscopic_var():
    save_folder_name = "illustration/generator"

    # ---- illustrate moment dynamics
    w_data_generator = np.load('paper_data/generator1D/generated_macroscopic_variables_normalized.npy')
    w_data_generator_d = np.load('paper_data/generator1D/generated_macroscopic_variables_normalized_gradients.npy')
    sod_data_generator = np.load('paper_data/generator1D/sod_macroscopic_variables_normalized.npy')
    sod_data_generator_d = np.load('paper_data/generator1D/sod_macroscopic_variables_gradients.npy')
    fontsize = 28

    xticks = [-1.5, -0.5, 0.5, 1.5]

    plot_1dv2(xs=[w_data_generator[:, 0]], ys=[w_data_generator[:, 1]], labels=None, name='generated_moments',
              log=False, folder_name=save_folder_name, linetypes=['o'], show_fig=False, xlim=(-1.5, 1.5),
              xticks=xticks, font_size=fontsize, ylim=(0, 1), xlabel=r"$U$", ylabel=r"$T$", black_first=False)
    plot_1dv2(xs=[sod_data_generator[:, 0]], ys=[sod_data_generator[:, 1]], labels=None, name='sod_moments',
              log=False, folder_name=save_folder_name, linetypes=['o'], show_fig=False, xlim=(-1.5, 1.5),
              xticks=xticks, ylim=(0, 1), xlabel=r"$U$", ylabel=r"$T$", black_first=False)
    xticks = [-10, 0, 10]
    yticks = [-10, 0, 10]
    plot_1dv2(xs=[w_data_generator_d[:, 0]], ys=[w_data_generator_d[:, 1]], labels=None, name='generated_moments_grad',
              log=False, folder_name=save_folder_name, linetypes=['o'], show_fig=False, xlim=(-10, 10),
              xticks=xticks, yticks=yticks, font_size=fontsize, ylim=(-10, 10), xlabel=r"$\nabla U$",
              ylabel=r"$\nabla  T$", black_first=False)
    plot_1dv2(xs=[sod_data_generator_d[:, 0]], ys=[sod_data_generator_d[:, 1]], labels=None, name='sod_moments_grad',
              xticks=xticks, yticks=yticks, log=False, folder_name=save_folder_name, linetypes=['o'], show_fig=False,
              xlim=(-10, 10), font_size=fontsize, ylim=(-10, 10), xlabel=r"$\nabla  U$", ylabel=r"$\nabla T$",
              black_first=False)

    return 0


def print_sod_regime():
    folder_name = "paper_data/sod1D/regime/"
    x_data = np.load(folder_name + "sod_x.npy")
    n_jump = 5
    fontsize = 26
    markersize = 8

    # ----- KN=1e-4 ----------
    regime = "1e-4"
    nn_data = np.load(folder_name + "sod_Kn_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "sod_Kn_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "sod_Kn_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data, x_data[::n_jump], x_data[::n_jump]],
              ys=[ground_truth_data, nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left", font_size=fontsize,
              marker_size=markersize, name='Regime_labels' + regime, log=False, folder_name="illustration/Sod1d/",
              linetypes=["-", "o", "^"], show_fig=False, xlim=(0, 1), ylim=(-0.05, 1.05), xlabel=r"$x$",
              ylabel=r"regime", black_first=True)

    # ----- KN=1e-3 ----------
    regime = "1e-3"
    nn_data = np.load(folder_name + "sod_Kn_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "sod_Kn_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "sod_Kn_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data, x_data[::n_jump], x_data[::n_jump]],
              ys=[ground_truth_data, nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left", font_size=fontsize,
              marker_size=markersize, name='Regime_labels' + regime, log=False, folder_name="illustration/Sod1d/",
              linetypes=["-", "o", "^"], show_fig=False, xlim=(0, 1), ylim=(-0.05, 1.05), xlabel=r"$x$",
              ylabel=r"regime", black_first=True)

    # ----- KN=1e-2 ----------
    regime = "1e-2"
    nn_data = np.load(folder_name + "sod_Kn_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "sod_Kn_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "sod_Kn_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data, x_data[::n_jump], x_data[::n_jump]],
              ys=[ground_truth_data, nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="lower right", font_size=fontsize,
              marker_size=markersize, name='Regime_labels' + regime, log=False, folder_name="illustration/Sod1d/",
              linetypes=["-", "o", "^"], show_fig=False, xlim=(0, 1), ylim=(-0.05, 1.05), xlabel=r"$x$",
              ylabel=r"regime", black_first=True)

    return 0


def print_sod_solution():
    folder_name = "paper_data/sod1D/solution/"
    x_data = np.load(folder_name + "sod_sol_x_1.npy")
    n_jump = 1
    fontsize = 26
    markersize = 8

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
              font_size=fontsize, xticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.59, 0.69), ylim=(.2, .5),
              font_size=fontsize, xticks=[0.59, 0.61, 0.63, 0.65, 0.67, 0.69], yticks=[0.2, 0.3, 0.4, 0.5])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.4, 0.5), ylim=(.47, .7),
              font_size=fontsize, xticks=[0.4, 0.42, 0.44, 0.46, 0.48, 0.5])

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
              font_size=fontsize)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="lower right",
              name='solution_U_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.4, 0.6),
              xticks=[0.4, 0.45, 0.5, 0.55, 0.6], ylim=(0.5, 0.85), yticks=[0.5, 0.6, 0.7, 0.8],
              font_size=fontsize)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_U_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.25, 0.35), ylim=(-0.001, 0.25),
              font_size=fontsize)

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
              font_size=fontsize, yticks=[1.2, 1.6, 2.0, 2.4])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_T_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.44, 0.55), ylim=(1.22, 1.4),
              yticks=[1.24, 1.28, 1.32, 1.36, 1.4], font_size=fontsize)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_T_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.58, 0.68),
              yticks=[1.3, 1.6, 1.9, 2.2, 2.5], ylim=(1.22, 2.65), font_size=fontsize)

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
              font_size=fontsize)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.55, 0.68), ylim=(.2, .5),
              font_size=fontsize, xticks=[0.56, 0.59, 0.62, 0.65, 0.68], yticks=[0.2, 0.26, 0.32, 0.38, 0.44, 0.5])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.7, 0.85), ylim=(.122, .24),
              font_size=fontsize, xticks=[0.7, 0.74, 0.78, 0.82, 0.85], yticks=[0.12, 0.16, 0.2, 0.24])

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
              font_size=fontsize)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="lower left",
              name='solution_U_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.56, 0.76), ylim=(0.8, 0.88),
              font_size=fontsize, xticks=[0.57, 0.62, 0.67, 0.72, 0.77], yticks=[0.8, 0.82, 0.84, 0.86, 0.88])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_U_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.27, 0.33), ylim=(-0.001, 0.15),
              font_size=fontsize, xticks=[0.28, 0.3, 0.32], yticks=[0.0, 0.05, 0.1, 0.15])

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
              font_size=fontsize, xlim=(0, 1), yticks=[1.2, 1.5, 1.8, 2.1, 2.4])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_T_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.44, 0.55), ylim=(1.22, 1.4),
              font_size=fontsize, xticks=[0.44, 0.47, 0.5, 0.53, 0.56], yticks=[1.22, 1.27, 1.32, 1.37, 1.42])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_T_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.58, 0.68), ylim=(1.22, 2.65),
              font_size=fontsize, yticks=[1.2, 1.7, 2.2, 2.7])

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
              font_size=fontsize)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="lower left",
              name='solution_rho_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.2, 0.35), ylim=(.85, 1.01),
              font_size=fontsize, yticks=[0.85, 0.9, 0.95, 1.0], xticks=[0.2, 0.25, 0.3, 0.35])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_rho_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(0.7, 0.9), ylim=(.122, .24),
              font_size=fontsize, xticks=[0.70, 0.75, 0.80, 0.85, 0.90], yticks=[0.12, 0.16, 0.2, 0.24])

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
              font_size=fontsize)
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="lower left",
              name='solution_U_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.56, 0.76), ylim=(0.8, 0.88),
              font_size=fontsize, xticks=[0.57, 0.62, 0.67, 0.72, 0.77], yticks=[0.8, 0.82, 0.84, 0.86, 0.88])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='solution_U_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U$", black_first=True, xlim=(0.27, 0.33), ylim=(-0.001, 0.15),
              font_size=fontsize, xticks=[0.27, 0.29, 0.31, 0.33], yticks=[0.0, 0.05, 0.1, 0.15])

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
              font_size=fontsize, xlim=(0, 1), ylim=(1.2, 2.6), yticks=[1.3, 1.7, 2.1, 2.5])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="lower left",
              name='solution_T_zoom1_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.2, 0.35), ylim=(1.75, 2.01),
              font_size=fontsize, xticks=[0.2, 0.25, 0.3, 0.35])
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper right",
              name='solution_T_zoom2_' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(0.7, 0.9),
              xticks=[0.7, 0.75, 0.8, 0.85, 0.9], ylim=(1.6, 2.65),
              font_size=fontsize)
    return 0


def print_shear_layer_regime():
    folder_name = "paper_data/shear_layer_2d/regime/"
    save_folder_name = "illustration/shear_layer_2d"
    x_data = np.load(folder_name + "layer_f_regime_x_1.npy")
    n_jump = 2
    # ----- KN=1tau ----------
    regime = "1"
    nn_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='Regime_labels_shear_layer' + regime, log=False, folder_name=save_folder_name,
              linetypes=["-", "o", "^"],
              show_fig=False, xlim=(-0.1, 0.1), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime", black_first=True,
              yticks=[0, 1], xticks=[-.1, -.05, 0.0, .05, .1], font_size=20)

    # ----- KN=1tau ----------
    regime = "2"
    n_jump = 3
    nn_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='Regime_labels_shear_layer' + regime, log=False, folder_name=save_folder_name,
              linetypes=["-", "o", "^"],
              show_fig=False, xlim=(-0.15, 0.15), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime", black_first=True,
              yticks=[0, 1], xticks=[-0.15, -.1, -.05, 0.0, .05, .1, 0.15], font_size=20)

    # ----- KN=50tau ----------
    regime = "3"
    n_jump = 8
    nn_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "layer_f_regime_tau_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data[::n_jump]], ys=[ground_truth_data[::n_jump], nn_data[::n_jump], kn_gll_data[::n_jump]],
              labels=['true', r'$\rm{N}_\theta$', r'$\rm{Kn}_{GLL}$'], legend_pos="upper left",
              name='Regime_labels_shear_layer' + regime, log=False, folder_name=save_folder_name,
              linetypes=["-", "o", "^"],
              show_fig=False, xlim=(-0.5, 0.5), ylim=(-0.05, 1.05), xlabel=r"$x$", ylabel=r"regime",
              black_first=True,
              yticks=[0, 1], font_size=20)

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
    n_jump = 1
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='shear_rho_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=(-0.04, 0.04), ylim=(.8, 1.2),
              font_size=26, symbol_size=2, marker_size=6, xticks=[-.04, -0.02, 0.0, 0.02, 0.04],
              yticks=[0.8, 0.9, 1.0, 1.1, 1.2])
    # U_1
    nn_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_u1_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_1$", black_first=True, xlim=(-0.04, 0.04),
              font_size=26, symbol_size=2, marker_size=6, xticks=[-.04, -0.02, 0.0, 0.02, 0.04],
              yticks=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25], ylim=(-0.015, .25))
    # U_2
    nn_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_u2_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_2$", black_first=True, xlim=(-0.04, 0.04),
              font_size=26, symbol_size=2, marker_size=6, xticks=[-.04, -0.02, 0.0, 0.02, 0.04],
              yticks=[-1.0, -0.5, 0., 0.5, 1], ylim=(-1.05, 1.05))
    # T
    nn_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_T_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=(-0.04, 0.04),
              font_size=26, symbol_size=2, marker_size=6, xticks=[-.04, -0.02, 0.0, 0.02, 0.04],
              yticks=[0.5, 0.75, 1.0, 1.25], ylim=(.48, 1.37))

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
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=x_lim,
              xticks=[-0.1, -0.05, 0, 0.05, 0.1],
              font_size=26, symbol_size=2, marker_size=6,
              ylim=(.65, 1.35), yticks=[0.6, 0.8, 1, 1.2])
    # U_1
    nn_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='shear_u1_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_1$", black_first=True, xlim=x_lim,
              xticks=[-0.1, -0.05, 0, 0.05, 0.1],
              font_size=26, symbol_size=2, marker_size=6, ylim=(-0.01, .21), yticks=[0, 0.05, .1, 0.15, 0.2])
    # U_2
    nn_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_u2_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_2$", black_first=True, xlim=x_lim,
              xticks=[-0.1, -0.05, 0, 0.05, 0.1],
              font_size=26, symbol_size=2, marker_size=6, ylim=(-1.05, 1.05), yticks=[-1.0, -0.5, 0, 0.5, 1])
    # T
    nn_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_T_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=x_lim,
              xticks=[-0.1, -0.05, 0, 0.05, 0.1],
              font_size=26, symbol_size=2, marker_size=6, ylim=(.48, 1.27), yticks=[0.5, 0.75, 1.0, 1.25])

    # ----- t=50\tau----------
    n_jump = 7
    regime = "3"
    x_lim = (-0.4, .4)
    x_ticks = [-0.4, -0.2, 0, 0.2, 0.4]
    # rho
    x_data = np.load(folder_name + "layer_n_t_x_" + regime + ".npy")
    nn_data = np.load(folder_name + "layer_n_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_n_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_n_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='shear_rho_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$\rho$", black_first=True, xlim=x_lim, xticks=x_ticks,
              font_size=26, symbol_size=2, marker_size=6, ylim=(.65, 1.37), yticks=[0.7, 0.9, 1.1, 1.3])
    # U_1
    nn_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_u_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='shear_u1_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_1$", black_first=True, xlim=x_lim, xticks=x_ticks,
              font_size=26, symbol_size=2, marker_size=6, ylim=(-0.01, .22), yticks=[0., 0.05, 0.1, 0.15, 0.2])
    # U_2
    nn_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_v_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_u2_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$U_2$", black_first=True, xlim=x_lim, xticks=x_ticks,
              font_size=26, symbol_size=2, marker_size=6, ylim=(-1.05, 1.05), yticks=[-1., -0.5, 0, 0.5, 1.0])
    # T
    nn_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_Kinetic.npy")
    ns_data = np.load(folder_name + "layer_t_t_tau_" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper right", linetypes=["-", "--", "o"],
              name='shear_T_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x$", ylabel=r"$T$", black_first=True, xlim=x_lim, xticks=x_ticks,
              font_size=26, symbol_size=2, marker_size=6, ylim=(.48, 1.19), yticks=[0.5, 0.7, 0.9, 1.1])

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
    fontsize = 26
    # ----- kn=1e-2----------
    regime = "2"
    x_lim = (-5, -1)
    x_ticks = [-5, -4, -3, -2, -1]
    # density
    x_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_x.npy")
    nn_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_kinetic.npy")
    ns_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='cylinder_rho_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x_1$", ylabel=r"$\rho$", black_first=True, ylim=(0, 13), xlim=x_lim,
              xticks=x_ticks, font_size=fontsize)
    # U velocity
    x_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_x.npy")
    nn_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_kinetic.npy")
    ns_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="lower left", linetypes=["-", "--", "o"],
              name='cylinder_u_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x_1$", ylabel=r"$U_1$", black_first=True, ylim=(0, 5), xlim=x_lim,
              xticks=x_ticks, font_size=fontsize)
    # T
    x_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_x.npy")
    nn_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_kinetic.npy")
    ns_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='cylinder_T_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x_1$", ylabel=r"$T$", black_first=True, ylim=(0, 10), xlim=x_lim,
              xticks=x_ticks, font_size=fontsize)

    # ----- kn=1e-3----------
    regime = "3"
    x_lim = (-5, -1)
    x_ticks = [-5, -4, -3, -2, -1]
    # density
    x_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_x.npy")
    nn_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_kinetic.npy")
    ns_data = np.load(folder_name + "0_cylinder_f_n_kn" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='cylinder_rho_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x_1$", ylabel=r"$\rho$", black_first=True, ylim=(0, 13), xlim=x_lim,
              xticks=x_ticks, font_size=fontsize)
    # U velocity
    x_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_x.npy")
    nn_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_kinetic.npy")
    ns_data = np.load(folder_name + "0_cylinder_f_u_kn" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="lower left", linetypes=["-", "--", "o"],
              name='cylinder_u_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x_1$", ylabel=r"$U_1$", black_first=True, ylim=(0, 5), xlim=x_lim,
              xticks=x_ticks, font_size=fontsize)
    # T
    x_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_x.npy")
    nn_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_Adaptive.npy")
    kinetic_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_kinetic.npy")
    ns_data = np.load(folder_name + "0_cylinder_f_t_kn" + regime + "_NS.npy")
    plot_1dv2(xs=[x_data[::n_jump]],
              ys=[kinetic_data[::n_jump], ns_data[::n_jump], nn_data[::n_jump]],
              labels=['Kinetic', 'Navier-Stokes', 'Adaptive'], legend_pos="upper left", linetypes=["-", "--", "o"],
              name='cylinder_T_' + regime, log=False, folder_name=save_folder_name,
              show_fig=False, xlabel=r"$x_1$", ylabel=r"$T$", black_first=True, ylim=(0, 10), xlim=x_lim,
              xticks=x_ticks, font_size=fontsize)

    return 0


if __name__ == '__main__':
    main()
