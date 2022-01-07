"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 21.12.2021
"""

import numpy as np
import pandas as pd
from utils import load_density_function, scatter_plot_2d_N2, scatter_plot_3d, scatter_plot_2d


def paper_illustrations():
    # ---- illustrate moment dynamics
    x_dat = np.load('data/sod1D/X.npy')
    y_dat = np.load('data/sod1D/Y.npy')
    z_dat = np.load('data/sod1D/Z.npy')
    iter_dat = np.load('data/sod1D/I.npy')

    # 1) moment - Sod test case
    moment_u = x_dat[:3, :]
    # normalize everything
    u_normal = moment_u / moment_u[0, :]
    scatter_plot_2d_N2(x_in=u_normal[1:, :].reshape((u_normal.shape[1], 2)), z_in=iter_dat[0], show_fig=False,
                       log=False,
                       folder_name='illustrations', color_map=1, name='moment_dynamics', label_x='velocity',
                       label_y='temperature', title='macroscopic variables over time')

    # 2) gradient - sod test case
    grad_u = x_dat[3:6, :]
    scatter_plot_3d(xyz_in=grad_u.reshape((u_normal.shape[1], 3)), color_in=iter_dat[0], show_fig=False, log=False,
                    folder_name='illustrations', color_map=1, name='grad_moment_dynamics',
                    title='gradient macroscopic variables over time', lim_x=(-1, 10),
                    lim_y=(0, 10), lim_z=(0, 10))
    scatter_plot_2d(x_in=grad_u[1:, :].reshape((u_normal.shape[1], 2)), z_in=iter_dat[0], show_fig=False, log=False,
                    folder_name='illustrations', color_map=1, name='grad_moment_dynamics2D', label_x=r"$\nabla_x U$",
                    label_y=r"$\nabla_x T$", title='gradient macroscopic variables over time')

    # II ----- illustrate generated data
    x_dat = np.load('data/generator1D/X.npy')
    y_dat = np.load('data/generator1D/Y.npy')
    # 1) moment - generated
    moment_u = x_dat[:3, :]
    # normalize everything
    u_normal = moment_u / moment_u[0, :]
    scatter_plot_2d_N2(x_in=u_normal[1:, :].reshape((u_normal.shape[1], 2)), z_in=np.zeros(shape=(u_normal.shape[1])),
                       show_fig=False, log=False,
                       folder_name='illustrations', color_map=1, name='moment_dynamics_genData', label_x='velocity',
                       label_y='temperature', title='macroscopic variables (generated)')

    # 2) gradient - generated
    grad_u = x_dat[3:6, :]
    scatter_plot_3d(xyz_in=grad_u.reshape((u_normal.shape[1], 3)),
                    color_in=np.zeros(shape=(u_normal.shape[1])), show_fig=False, log=False,
                    folder_name='illustrations', color_map=1, name='grad_moment_dynamics_genData',
                    title='gradient macroscopic variables (generated)', lim_x=(-1, 10),
                    lim_y=(0, 10), lim_z=(0, 10))
    scatter_plot_2d(x_in=grad_u[1:, :].reshape((u_normal.shape[1], 2)),
                    z_in=np.zeros(shape=(u_normal.shape[1])), show_fig=False, log=False,
                    folder_name='illustrations', color_map=1, name='grad_moment_dynamics2D_genData',
                    label_x=r"$\nabla_x U$", label_y=r"$\nabla_x T$",
                    title='gradient macroscopic variables (generated)')
    return 0


if __name__ == '__main__':
    paper_illustrations()
