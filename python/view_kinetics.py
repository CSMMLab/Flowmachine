"""
Script to call different plots and illustrative methods - specifically tailored for the paper
Author: Steffen Schotthoefer
Version: 0.1
Date 22.10.2021
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_density_function2D, load_density_function, plot_density_fusion_1d, scatter_plot_2d


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


def paper_illustrations():
    """ Skript that performs all illustrations for the paper done by Steffen"""
    [v_x, weights, f_kinetic] = load_density_function("data/1d/a2_ev10.csv")
    v_x = np.reshape(v_x, newshape=(v_x.shape[1]))
    # ----- Ilustrate Upwing Merging of two pdf -----
    f_L = generate_maxwellian_1d(v_x, 1.0, 0.0, 1.0)
    f_R = generate_maxwellian_1d(v_x, 1.0, 0.5, 1.0)
    f_res = upwind_reconstruction(v_x, f_L, f_R)
    # Save to file
    kinetics = np.asarray([f_L, f_R, f_res])
    np.savetxt('maxwellians.csv', kinetics, delimiter=',')

    plot_density_fusion_1d(v_x=v_x, f_l=f_L, f_r=f_R, f_fuse=f_res, show_fig=True, save_name='maxwellians_fusion')
    return 0


def create_illustration_data():
    """ Skript that performs all illustrations for the paper done by Steffen"""
    [v_x, weights, f_kinetic] = load_density_function("../data/1d/a2_ev10.csv")
    v_x = np.reshape(v_x, newshape=(v_x.shape[1]))
    weights = np.reshape(weights, newshape=(weights.shape[1]))
    # ----- Ilustrate Upwing Merging of two pdf -----
    f_L = generate_maxwellian_1d(v_x, 1.0, 0.0, 1.0)
    f_R = generate_maxwellian_1d(v_x, 1.0, 0.5, 1.0)
    f_res = upwind_reconstruction(v_x, f_L, f_R)
    # Save to file
    kinetics = np.asarray([v_x, weights, f_L, f_R, f_res])
    np.savetxt('maxwellians.csv', kinetics, delimiter=',')

    # plot_density_fusion_1d(v_x=v_x, f_l=f_L, f_r=f_R, f_fuse=f_res, show_fig=True, save_name='maxwellians_fusion')
    return 0


"""
def main():
    print("---------- Start Result Illustration Suite ------------")
    [x, y, w, kinetic_f] = load_density_function2D("xxx.csv")
    fig = plt.figure(figsize=(5.8, 4.7), dpi=400)
    ax = fig.add_subplot(111)  # , projection='3d')
    out = plt.scatter(x, y, c=kinetic_f)
    cbar = fig.colorbar(out, ax=ax, extend='both')
    # plt.show()
    sum = 0
    sum2 = 0
    rec_U = 0.0
    rec_T = 0
    kin = []
    nq = kinetic_f.shape[0]
    w2 = 25.0 / nq
    for i in range(kinetic_f.shape[0]):
        kin.append(maxwellian2D(x[0, i], y[0, i], 1, 1))
        sum += kinetic_f[i] * w[0, i]
        sum2 += (x[0, i] ** 2 + y[0, i] ** 2) / 2 * kinetic_f[i] * w[0, i]
        rec_U += maxwellian2D(x[0, i], y[0, i], 1, 2) * w[0, i]
        rec_T += (x[0, i] ** 2 + y[0, i] ** 2) / 2 * maxwellian2D(x[0, i], y[0, i], 1, 2) * w[0, i]  # * w[0, i]
    print(rec_U)
    print(rec_T)
    print("...")
    print(sum)
    print(sum2)
    print("_---")
    kin = np.asarray(kin)

    fig = plt.figure(figsize=(5.8, 4.7), dpi=400)
    ax = fig.add_subplot(111)  # , projection='3d')
    out = plt.scatter(x, y, c=kin)
    cbar = fig.colorbar(out, ax=ax, extend='both')
    # plt.show()

    # plt.ylim(0, 1)
    # plt.xlim(-5, 5)
    # plt.show()
    #  plt.savefig("test_a10_ev5")
    # for i in range(int(kinetic_f.shape[0] / 5)):
    #    kinetic_list = [kinetic_f[i + 0], kinetic_f[i + 1], kinetic_f[i + 2], kinetic_f[i + 3], kinetic_f[i + 4]]
    #    plot_1d(x, kinetic_list, show_fig=False, log=False, name='kinetics_kond3_' + str(i).zfill(3), ylim=[0, 3],
    #            xlim=[x[0, 0], x[0, -1]])

    return True
"""

if __name__ == '__main__':
    create_illustration_data()
    # paper_illustrations()
