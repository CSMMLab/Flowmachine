"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 21.12.2021
"""

import numpy as np
from utils import plot_1dv2


def main():
    print_sod_regime()

    # paper_illustrations()
    return 0


def print_sod_regime():
    folder_name = "paper_data/sod1D/regime/"
    x_data = np.load(folder_name + "sod_x.npy")
    # ----- KN=1e-4 ----------
    regime = "1e-4"
    nn_data = np.load(folder_name + "sod_Kn_" + regime + "_NN.npy")
    kn_gll_data = np.load(folder_name + "sod_Kn_" + regime + "_KnGLL.npy")
    ground_truth_data = np.load(folder_name + "sod_Kn_" + regime + "_True.npy")

    plot_1dv2(xs=[x_data], ys=[ground_truth_data, nn_data, kn_gll_data], labels=['ground truth', 'NN', 'KnGLL'],
              name='Regime_labels' + regime, log=False, folder_name="illustration/Sod1d/", linetypes=None,
              show_fig=False, xlim=(0, 1), ylim=(0, 1), xlabel=r"$x$", ylabel=r"regime", black_first=True)

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
