"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 21.12.2021
"""

import numpy as np
import pandas as pd
from utils import load_density_function2D, load_density_function, plot_density_fusion_1d, plot_densities, plot_1d
from entropytools import EntropyTools, qGaussLegendre1D


def paper_illustrations():
    x_dat = np.load('data/sod1D/X.npy')
    y_dat = np.load('data/sod1D/Y.npy')
    z_dat = np.load('data/sod1D/Z.npy')  #

    moment_u = x_dat[:3, :]
    # normalize everything
    u_normal = moment_u / moment_u[0, :]
    
    """ Skript that performs all illustrations for the paper done by Steffen"""
    [v_x, weights, f_kinetic] = load_density_function("data/pdfs.csv")
    f_ns = pd.read_csv("data/fns.csv").to_numpy()
    grads = pd.read_csv("data/gradient_w.csv").to_numpy()
    params = pd.read_csv("data/paras.csv").to_numpy()
    conservative_variables = pd.read_csv("data/w.csv").to_numpy()

    # ----- Ilustrate Upwing Merging of two pdf -----
    v_x = np.reshape(v_x, newshape=(v_x.shape[1]))

    # plot_density_fusion_1d(v_x=v_x, f_l=f_kinetic[0], f_r=f_kinetic[1], f_fuse=f_kinetic[2], f_ns=f_ns[:, 0],
    #                       show_fig=False, save_name='maxwellians_fusion')
    # plot_density_fusion_1d(v_x=v_x, f_l=f_kinetic[3], f_r=f_kinetic[4], f_fuse=f_kinetic[5], f_ns=f_ns[:, 1],
    #                       show_fig=False, save_name='bimodal_fusion')

    plot_1d(xs=[v_x], ys=[f_kinetic[0, :].reshape(len(f_kinetic[0]), 1), f_kinetic[1, :].reshape(len(f_kinetic[0]), 1),
                          f_kinetic[2, :].reshape(len(f_kinetic[0]), 1), f_ns[:, 0].reshape(len(f_kinetic[0]), 1)],
            labels=['left cell', 'right cell', 'interface', 'BGK reconstruction'], name='maxwell_fusion', log=False,
            folder_name="illustrations", linetypes=['-', '--', 'o', '-.'], show_fig=False, xlim=(-5, 5), ylim=(0, 0.75),
            xlabel="velocity", ylabel="density", title=" ")
    plot_1d(xs=[v_x], ys=[f_kinetic[3, :].reshape(len(f_kinetic[0]), 1), f_kinetic[4, :].reshape(len(f_kinetic[0]), 1),
                          f_kinetic[5, :].reshape(len(f_kinetic[0]), 1), f_ns[:, 1].reshape(len(f_kinetic[0]), 1)],
            labels=['left cell', 'right cell', 'interface', 'BGK reconstruction'], name='bimodal_fusion', log=False,
            folder_name="illustrations", linetypes=['-', '--', 'o', '-.'], show_fig=False, xlim=(-5, 5), ylim=(0, 0.75),
            xlabel="velocity", ylabel="density", title=" ")

    plot_1d(xs=[v_x], ys=[f_kinetic[0, :].reshape(len(f_kinetic[0]), 1), f_kinetic[3, :].reshape(len(f_kinetic[0]), 1),
                          f_kinetic[8, :].reshape(len(f_kinetic[0]), 1)],
            labels=['Maxwellian', 'Bimodal', 'Highly anisotropic'],
            name='Entropy_Sampling', log=False,
            folder_name="illustrations", linetypes=None,
            show_fig=False, xlim=(-5, 5), ylim=(0, 0.75), xlabel="velocity", ylabel="density",
            title=" ")

    # plot_densities(v_x=v_x, f_maxwell=f_kinetic[0], f_entropy=f_kinetic[4], f_fourier=f_kinetic[6],
    #               f_random=f_kinetic[7], f_unlikely=f_kinetic[8], show_fig=False, save_name='example_densities')

    return 0


if __name__ == '__main__':
    paper_illustrations()
