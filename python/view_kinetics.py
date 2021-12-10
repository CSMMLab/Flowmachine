"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 22.10.2021
"""

import numpy as np
from random import random
import matplotlib.pyplot as plt
import pandas as pd

from utils import load_density_function2D, load_density_function, plot_density_fusion_1d, plot_densities, plot_1d
from entropytools import EntropyTools, qGaussLegendre1D


def maxwellian_2d(v_x, v_y, rho, T):
    return rho / (2 * np.pi * T) * np.exp(-rho / (2 * T) * (v_x ** 2 + v_y ** 2))


def maxwellian_1d(v_x, rho, U, T):
    return (rho / (2 * np.pi * T)) ** 0.5 * np.exp(-rho / (2 * T) * ((v_x - U) ** 2))


def upwind_reconstruction(v_x, f_l: np.ndarray, f_r: np.ndarray) -> np.ndarray:
    f_res = np.ndarray(shape=f_r.shape)
    for idx in range(f_r.shape[0]):
        if v_x[idx] > 0:
            f_res[idx] = f_l[idx]
        else:
            f_res[idx] = f_r[idx]
    return f_res


def generate_maxwellian_1d(v_x, rho, U, T):
    f_res = np.ndarray(shape=(len(v_x)))
    for idx in range(len(v_x)):
        f_res[idx] = maxwellian_1d(v_x[idx], rho, U, T)
    return f_res


def generate_kinetic_density(v_x, alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, offset=0.0):
    f_res = np.ndarray(shape=(len(v_x)))
    for idx in range(len(v_x)):
        v = v_x[idx]
        f_res[idx] = np.exp(
            alpha_0 + alpha_1 * v + alpha_2 * ((v - offset) ** 2) + alpha_3 * ((v - offset) ** 3) + alpha_4 * (
                    (v - offset) ** 4))
    return f_res


def generate_random_density(v_x):
    f_res = np.ndarray(shape=(len(v_x)))
    for idx in range(len(v_x)):
        v = v_x[idx]
        f_res[idx] = random()
    return f_res


def generate_sin_density(v_x):
    f_res = np.ndarray(shape=(len(v_x)))
    for idx in range(len(v_x)):
        v = v_x[idx]
        f_res[idx] = np.sin(v) ** 2 * 0.35
    return f_res


def compute_condition_number(alpha: list):
    # initialize entropy tools
    poly_degree = len(alpha)
    et = EntropyTools(polynomial_degree=poly_degree, spatial_dimension=1)
    alpha = np.asarray(alpha).reshape((1, poly_degree))
    alpha_tf = et.convert_to_tensor_float(alpha)
    alpha_complete = et.reconstruct_alpha(alpha_tf).numpy().reshape((poly_degree + 1, 1))
    hessian = et.entropy_hessian(alpha_complete, et.momentBasis.numpy(), et.quadWeights.numpy())
    return np.linalg.cond(hessian)


def compute_normalized_kinetic_density(alpha: list):
    # initialize entropy tools
    poly_degree = len(alpha)
    et = EntropyTools(polynomial_degree=poly_degree, spatial_dimension=1)
    alpha = np.asarray(alpha).reshape((1, poly_degree))
    alpha_tf = et.convert_to_tensor_float(alpha)
    alpha_complete = et.reconstruct_alpha(alpha_tf)
    f_res = et.compute_kinetic_density(alpha_complete).numpy()
    v_x = et.quadPts.numpy()
    return v_x.reshape((f_res.shape[1])), f_res.reshape((f_res.shape[1])), alpha_complete.numpy()


def create_illustration_data():
    """ Skript that performs all illustrations for the paper done by Steffen"""
    # [v_x, weights, f_kinetic] = load_density_function("../data/1d/a2_ev10.csv")
    # v_x = np.reshape(v_x, newshape=(v_x.shape[1]))
    # v_x = v_x[1:]
    # weights = np.reshape(weights, newshape=(weights.shape[1]))
    # weights = weights[1:]
    # nq = 200
    # v_x = np.linspace(-5, 5, nq)
    # weights = np.asarray([5.0 / float(nq)] * nq)
    v_x, weights = qGaussLegendre1D(200)
    v_x = v_x * 5.0
    weights = weights * 5.0
    print(sum(weights))
    kinetics = [v_x, weights]

    # ----- Ilustrate Upwing Merging of two pdf -----
    # 1) Mix two Maxwellians
    f_maxwelll_L = generate_maxwellian_1d(v_x, 1.0, 0.0, 1.0)
    f_maxwell_R = generate_maxwellian_1d(v_x, 1.0, 0.1, .98)
    f_maxwell_fusion = upwind_reconstruction(v_x, f_maxwelll_L, f_maxwell_R)
    kinetics.append(f_maxwelll_L)
    kinetics.append(f_maxwell_R)
    kinetics.append(f_maxwell_fusion)

    # 2) Generate bimodal Distributions
    alpha_0 = -0.9
    alpha_1 = -0.5
    alpha_2 = -0.5
    alpha_3 = 0.5
    alpha_4 = -0.1000
    t1, t2, alpha_complete = compute_normalized_kinetic_density([alpha_1, alpha_2, alpha_3, alpha_4])
    f_L_bimod = generate_kinetic_density(v_x, alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, offset=-0.4)

    kinetics.append(f_L_bimod)
    alpha_0 = -0.5
    alpha_1 = 0.5
    alpha_2 = -0.3
    alpha_3 = 0.2
    alpha_4 = -0.1000
    f_R_bimod = generate_kinetic_density(v_x, alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, offset=-1.4)
    f_bimod_fusion = upwind_reconstruction(v_x, f_L_bimod, f_R_bimod)
    kinetics.append(f_R_bimod)
    kinetics.append(f_bimod_fusion)

    # 3) Generate random Distributions
    f_random = generate_random_density(v_x)
    kinetics.append(f_random)

    # 4) Generate sinusoidal Distributions
    f_L_sin = generate_sin_density(v_x)
    kinetics.append(f_L_sin)

    # 5) Generate unlikely entropy distributions
    alpha_0 = -1.3
    alpha_1 = -0.2
    alpha_2 = -2
    alpha_3 = -0.38  # -0.01
    alpha_4 = -0.0
    f_unlikely = generate_kinetic_density(v_x, alpha_0, alpha_1, alpha_2, alpha_3, alpha_4, offset=0.4)
    kinetics.append(f_unlikely)

    # Save to file
    kinetics_np = np.asarray(kinetics)
    np.savetxt('data/pdfs.csv', kinetics_np, delimiter=',')
    print("Save constructed densities to file pfs.csv")

    """
    plot_1d(xs=[v_x],
            ys=[kinetics_np[2, :].reshape(len(kinetics_np[0]), 1), kinetics_np[3, :].reshape(len(kinetics_np[0]), 1),
                kinetics_np[4, :].reshape(len(kinetics_np[0]), 1)],
            labels=['left', 'right', 'Interface'],
            name='Entropy_Sampling2', log=False,
            folder_name="illustrations", linetypes=None,
            show_fig=False, xlim=(-5, 5), ylim=(0, 0.75), xlabel="velocity", ylabel="density",
            title=" ")
    plot_1d(xs=[v_x],
            ys=[kinetics_np[5, :].reshape(len(kinetics_np[0]), 1), kinetics_np[6, :].reshape(len(kinetics_np[0]), 1),
                kinetics_np[7, :].reshape(len(kinetics_np[0]), 1)],
            labels=['left', 'right', 'Interface'],
            name='Entropy_Sampling3', log=False,
            folder_name="illustrations", linetypes=None,
            show_fig=False, xlim=(-5, 5), ylim=(0, 0.75), xlabel="velocity", ylabel="density",
            title=" ")
    plot_1d(xs=[v_x],
            ys=[kinetics_np[2, :].reshape(len(kinetics_np[0]), 1), kinetics_np[5, :].reshape(len(kinetics_np[0]), 1),
                kinetics_np[10, :].reshape(len(kinetics_np[0]), 1)],
            labels=['Maxwellian', 'Bimodal', 'Highly anisotropic'],
            name='Entropy_Sampling', log=False,
            folder_name="illustrations", linetypes=None,
            show_fig=False, xlim=(-5, 5), ylim=(0, 0.75), xlabel="velocity", ylabel="density",
            title=" ")
    """
    # plot_density_fusion_1d(v_x=v_x, f_l=f_L_unlikely, f_r=f_R, f_fuse=f_res, show_fig=True,
    #                       save_name='maxwellians_fusion')

    # ------ with condition number

    #  some tests for condition number
    # cond_maxwellian = compute_condition_number([0.0, -0.5])
    # pts, f_test = compute_normalized_kinetic_density([0.0, -0.5])
    # plt.plot(pts, f_test)
    # plt.show()
    # cond_maxwellian2 = compute_condition_number([2.0, -0.5])
    # cond_maxwellian3 = compute_condition_number([0.0, -0.05])
    return 0


def paper_illustrations():
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
    create_illustration_data()
    paper_illustrations()
