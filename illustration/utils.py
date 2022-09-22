'''
Accumulation of utility functions
Date: 15.03.2021
Author: Steffen Schotthöfer
'''

import numpy as np
import time
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
import seaborn as sns
import os
from pathlib import Path


# plt.style.use("kitish")


def finiteDiff(x, y):
    '''
    :param x: Function Argument
    :param y: Function value = f(x)
    :return: df/dx at all points x
    '''

    grad = np.zeros(x.shape)

    grad[0] = (y[1] - y[0]) / (x[1] - x[0])

    for i in range(0, x.shape[0] - 1):
        grad[i + 1] = (y[i] - y[i - 1]) / (x[i] - x[i - 1])

    return grad


def integrate(x, y):
    '''
    :param x: function argument
    :param y: = f(x)
    :return: integrate y over span of x
    '''

    integral = np.zeros(x.shape)

    for i in range(0, x.shape[0] - 1):
        integral[i + 1] = integral[i] + (x[i + 1] - x[i]) * y[i + 1]

    return integral


def load_density_function(filename: str) -> list:
    '''
    Load training Data from csv file <filename>
    u, alpha have length <inputDim>
    returns: training_data = [u,alpha,h]
    '''
    print("Loading Data from location: " + filename)
    start = time.time()
    df = pd.read_csv(filename, header=None)
    # df = df.drop(df.columns[0], axis=1)
    data = df.to_numpy()
    x = data[0, :].reshape((1, len(data[0, :])))
    weights = data[1, :].reshape((1, len(data[0, :])))
    f_kinetic = data[2:, :]
    end = time.time()
    print("Data loaded. Elapsed time: " + str(end - start))
    return [x, weights, f_kinetic]


def load_density_function2D(filename: str) -> list:
    '''
    Load training Data from csv file <filename>
    u, alpha have length <inputDim>
    returns: training_data = [u,alpha,h]
    '''
    print("Loading Data from location: " + filename)
    start = time.time()
    df = pd.read_csv(filename, header=None)
    df = df.drop(df.columns[0], axis=1)
    data = df.to_numpy()
    x = data[0, :].reshape((1, len(data[0, :])))
    y = data[1, :].reshape((1, len(data[0, :])))

    weights = data[3, :].reshape((1, len(data[0, :])))
    f_kinetic = data[4, :]
    end = time.time()
    print("Data loaded. Elapsed time: " + str(end - start))
    return [x, y, weights, f_kinetic]


def load_solution(filename: str) -> list:
    '''
    Load training Data from csv file <filename>
    u, alpha have length <inputDim>
    returns: training_data = [u,alpha,h]
    '''
    print("Loading Data from location: " + filename)
    start = time.time()
    df = pd.read_csv(filename)
    data = df.to_numpy()
    t = data.shape[1] / 2
    u_neural = data[:, :int(data.shape[1] / 2)]
    u_ref = data[:, int(data.shape[1] / 2):]
    end = time.time()
    print("Data loaded. Elapsed time: " + str(end - start))
    return [u_neural, u_ref]


def plot_density_fusion_1d(v_x: np.ndarray, f_l: np.ndarray, f_r: np.ndarray, f_fuse: np.ndarray, f_ns: np.ndarray,
                           show_fig=True, save_name: str = 'fig1'):
    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    # fig, axs = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')
    plt.plot(v_x, f_l, 'r-')
    plt.plot(v_x, f_r, 'b-.')
    plt.plot(v_x, f_fuse, 'ko')
    plt.plot(v_x, f_ns, 'g--')
    plt.legend(["left cell", "right cell", "cell interface", "BGK reconstruction"])
    plt.xlabel("velocity")
    plt.ylabel("kinetic density")
    plt.xlim(-5.0, 5.0)
    if show_fig:
        plt.show()
    plt.savefig('illustrations/' + save_name + ".png", dpi=400)
    return 0


def plot_densities(v_x: np.ndarray, f_maxwell: np.ndarray, f_entropy: np.ndarray, f_fourier: np.ndarray,
                   f_random: np.ndarray, f_unlikely: np.ndarray, show_fig=True, save_name: str = 'fig1'):
    plt.clf()
    sns.set_theme()
    sns.set_style("white")
    # fig, axs = plt.subplots(2)
    # fig.suptitle('Vertically stacked subplots')
    plt.plot(v_x, f_maxwell, 'r-')
    plt.plot(v_x, f_entropy, 'b--')
    plt.plot(v_x, f_unlikely, 'c-.-')
    plt.plot(v_x, f_fourier, 'k.-')
    plt.plot(v_x, f_random, 'g--')

    plt.legend(["Maxwellian", "entropy - low condition", "entropy - high condition ", "fourier series generated",
                "random sampling at gridpoints"])
    plt.xlabel("velocity")
    plt.ylabel("kinetic density")
    plt.xlim(-5.0, 5.0)
    if show_fig:
        plt.show()
    plt.savefig('illustrations/' + save_name + ".png", dpi=400)
    return 0


def plot_1d(xs, ys, labels=None, name='defaultName', log=True, folder_name="figures", linetypes=None, show_fig=False,
            xlim=None, ylim=None, xlabel=None, ylabel=None, title: str = r"$h^n$ over ${\mathcal{R}^r}$"):
    plt.clf()
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h',
                     'H',
                     '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    sns.set_theme()
    sns.set_style("white")
    colors = ['k', 'r', 'g', 'b']
    symbol_size = 0.7
    if len(xs) == 1:
        x = xs[0]
        for y, lineType in zip(ys, linetypes):
            for i in range(y.shape[1]):
                if colors[i] == 'k' and lineType in ['.', ',', 'o', 'v', '^', '<', '>']:
                    colors[i] = 'w'
                plt.plot(x, y[:, i], colors[i] + lineType, linewidth=symbol_size, markersize=2.5,
                         markeredgewidth=0.5, markeredgecolor='k')
        if labels != None:
            plt.legend(labels)
    elif len(xs) is not len(ys):
        print("Error: List of x entries must be of same length as y entries")
        exit(1)
    else:
        for x, y, lineType in zip(xs, ys, linetypes):
            plt.plot(x, y, lineType, linewidth=symbol_size)
        plt.legend(labels)  # , prop={'size': 6})
    if log:
        plt.yscale('log')

    if show_fig:
        plt.show()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=14)
        # plt.xticks(fontsize=6)
        # plt.yticks(fontsize=6)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=14)
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".png", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".png"))
    return 0


def plot_1dv2(xs, ys, labels=None, name='defaultName', log=True, loglog=False, folder_name="figures", linetypes=None,
              show_fig=False, xlim=None, ylim=None, xlabel=None, ylabel=None, legend_pos="upper right",
              black_first=False):
    """
    Expected shape for x in xs : (nx,)
                       y in ys : (1,nx)
    """
    plt.clf()
    plt.figure(figsize=(5.8, 4.7), dpi=400)
    if not linetypes:
        linetypes = ['-', '--', '-.', ':', ':', '.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*',
                     'h',
                     'H',
                     '+', 'x', 'D', 'd', '|']
        if labels is not None:
            linetypes = linetypes[0:len(labels)]

    sns.set_theme()
    sns.set_style("white")
    colors = ['r', 'g', 'b', 'k']
    if black_first:
        colors = ['k', 'r', 'g', 'b']
    symbol_size = 2
    marker_size = 4
    marker_width = 0.5
    if len(xs) == 1:
        x = xs[0]
        i = 0
        for y, lineType in zip(ys, linetypes):
            if lineType in ['.', ',', 'o', 'v', '^', '<', '>']:
                if colors[i] == 'k':
                    plt.plot(x, y, 'w' + lineType, linewidth=symbol_size, markersize=marker_size,
                             markeredgewidth=marker_width, markeredgecolor='k')
                else:
                    plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size, markersize=marker_size,
                             markeredgewidth=marker_width, markeredgecolor='k')
            else:
                plt.plot(x, y, colors[i] + lineType, linewidth=symbol_size)
            i += 1
        if labels != None:
            plt.legend(labels, loc=legend_pos)
    elif len(xs) is not len(ys):
        print("Error: List of x entries must be of same length as y entries")
        exit(1)
    else:
        for x, y, lineType, color in zip(xs, ys, linetypes, colors):
            plt.plot(x, y, color + lineType, linewidth=symbol_size)
        plt.legend(labels)  # , prop={'size': 6})
    if log:
        plt.yscale('log')
    if loglog:
        plt.yscale('log')
        plt.xscale('log')
    if show_fig:
        plt.show()
    if ylim is not None:
        plt.ylim(ylim[0], ylim[1])
    if xlim is not None:
        plt.xlim(xlim[0], xlim[1])
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=14)
        # plt.xticks(fontsize=6)
        # plt.yticks(fontsize=6)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=14)
    # plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(folder_name + "/" + name + ".png", dpi=500)
    print("Figure successfully saved to file: " + str(folder_name + "/" + name + ".png"))
    plt.close()
    return 0


def scatter_plot_2d(x_in: np.ndarray, z_in: np.ndarray, lim_x: tuple = (-1, 1), lim_y: tuple = (0, 1),
                    label_x: str = r"$u_1^r$", label_y: str = r"$u_2^r$",
                    title: str = r"$h^n$ over ${\mathcal{R}^r}$", name: str = 'defaultName', log: bool = True,
                    folder_name: str = "figures", show_fig: bool = False, color_map: int = 0):
    '''
    brief: Compute a scatter plot
    input: x_in = [x1,x2] function arguments
           y_in = function values
    return: True if exit successfully
    '''
    # choose colormap
    if color_map == 1:
        c_map = cm.summer
    else:
        c_map = cm.hot

    fig = plt.figure(figsize=(5.8, 4.7), dpi=400)
    ax = fig.add_subplot(111)  # , projection='3d')
    x = x_in[:, 0]
    y = x_in[:, 1]
    z = z_in
    if log:
        out = ax.scatter(x, y, s=6, c=z, cmap=c_map, norm=colors.LogNorm())
    else:
        out = ax.scatter(x, y, s=6, c=z, cmap=c_map)
    # plt.xlim(lim_x[0], lim_x[1])
    # plt.ylim(lim_y[0], lim_y[1])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_aspect('auto')
    cbar = fig.colorbar(out, ax=ax, extend='both')
    if show_fig:
        plt.show()
    plt.savefig(folder_name + "/" + name + ".png", dpi=400)
    return 0


def scatter_plot_2d_N2(x_in: np.ndarray, z_in: np.ndarray, lim_x: tuple = (-1, 1), lim_y: tuple = (0, 1),
                       lim_z: tuple = (0, 1), label_x: str = r"$u_1^r$", label_y: str = r"$u_2^r$",
                       title: str = r"$h^n$ over ${\mathcal{R}^r}$", name: str = 'defaultName', log: bool = True,
                       folder_name: str = "figures", show_fig: bool = False, color_map: int = 0):
    '''
    brief: Compute a scatter plot
    input: x_in = [x1,x2] function arguments
           y_in = function values
    return: True if exit successfully
    '''
    # choose colormap
    if color_map == 1:
        c_map = cm.summer
    else:
        c_map = cm.hot

    plt.plot()
    fig = plt.figure(figsize=(5.8, 4.7), dpi=400)
    ax = fig.add_subplot(111)  # , projection='3d')

    u1 = np.linspace(-1, 1, 100)
    u2 = u1 * u1
    u2_top = np.ones(100)
    ax.plot(u1, u2, 'k--')
    ax.plot(u1, u2_top, 'k--')

    x = x_in[:, 0]
    y = x_in[:, 1]
    z = z_in
    if log:
        out = ax.scatter(x, y, s=6, c=z, cmap=c_map, norm=colors.LogNorm())
    else:
        out = ax.scatter(x, y, s=6, c=z, cmap=c_map)
    plt.xlim(lim_x[0], lim_x[1])
    plt.ylim(lim_y[0], lim_y[1])
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_aspect('auto')
    cbar = fig.colorbar(out, ax=ax, extend='both')
    if show_fig:
        plt.show()
    plt.savefig(folder_name + "/" + name + ".png", dpi=400)
    return 0


def scatter_plot_3d(xyz_in: np.ndarray, color_in: np.ndarray, lim_x: tuple = (-1, 1), lim_y: tuple = (0, 1),
                    lim_z: tuple = (0, 1), label_x: str = r"$\nabla_x\rho$", label_y: str = r"$\nabla_x U$",
                    label_z: str = r"$\nabla_x T$",
                    title: str = r"$h^n$ over ${\mathcal{R}^r}$", name: str = 'defaultName', log: bool = True,
                    folder_name: str = "figures", show_fig: bool = False, color_map: int = 0):
    '''
    brief: Compute a scatter plot
    input: x_in = [x1,x2] function arguments
           y_in = function values
    return: True if exit successfully
    '''
    # choose colormap
    if color_map == 1:
        c_map = cm.summer
    else:
        c_map = cm.hot

    fig = plt.figure(figsize=(5.8, 4.7), dpi=600)
    ax = fig.add_subplot(projection='3d')

    x = xyz_in[:, 0]
    y = xyz_in[:, 1]
    z = xyz_in[:, 2]
    if log:
        out = ax.scatter(x, y, z, c=color_in, cmap=c_map, norm=colors.LogNorm())
    else:
        out = ax.scatter(x, y, z, c=color_in, cmap=c_map)
    # plt.xlim(lim_x[0], lim_x[1])
    # plt.ylim(lim_y[0], lim_y[1])
    # plt.zlim(lim_z[0], lim_z[1])

    ax.set_title(title, fontsize=14)
    ax.set_xlabel(label_x)
    ax.set_ylabel(label_y)
    ax.set_zlabel(label_z)

    ax.set_aspect('auto')
    cbar = fig.colorbar(out, ax=ax, extend='both')
    if show_fig:
        plt.show()
    plt.savefig(folder_name + "/" + name + ".png", dpi=600)
    return 0


def make_directory(path_to_directory):
    if not os.path.exists(path_to_directory):
        p = Path(path_to_directory)
        p.mkdir(parents=True)
    return 0
