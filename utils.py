#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np  # numpy==1.21.4
from sklearn.manifold import MDS  # scipy==1.7.2
import cv2
import matplotlib.pyplot as plt
import ot
from ot.gromov import gwggrad, gwloss, init_matrix
# from ot.gromov import cg
from tqdm.notebook import trange, tqdm
from PIL import Image
from importlib import reload
import networkx as nx
import time
from utils import *
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.colorbar
import sklearn
from scipy.spatial import cKDTree


# functions
def lgw_procedure(M_ref, height_ref, posns, Ms, heights, max_iter=1000, mode="euclidean"):
    assert mode in ["euclidean", "graph"]
    N = len(Ms)

    Ps = []  # GW Plans
    Ts = []  # barycentric projections
    st = time.time()
    for i in range(0, N):
        # GW computation
        P = ot.gromov.gromov_wasserstein(M_ref,
                                         Ms[i],
                                         height_ref,
                                         heights[i],
                                         "square_loss",
                                         log=True,
                                         max_iter=max_iter,
                                         tol_rel=1e-20,
                                         tol_abs=1e-20,
                                         armijo=False)[0]
        Ps.append(P)

        # euclidean barycentric projection
        if mode == "euclidean":
            T = (np.divide(P.T, height_ref).T).dot(posns[i])
        # generalized barycentric projection
        else:
            T = []
            k = len(Ms[i])
            k_ref = len(M_ref)
            for v in range(k_ref):
                barycentricity = []
                weights = P[v] / height_ref[v]
                for w in range(k):
                    breakBool = False
                    if weights[w] == 1:
                        bary = w
                        breakBool = True
                        break
                    barycentricity.append(weights.dot(Ms[i][w] ** 2))
                if breakBool:
                    T.append(bary)
                else:
                    bary = np.argmin(barycentricity)
                    T.append(bary)
        Ts.append(T)

    # LGW computation
    lgw = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            if mode == "euclidean":
                lgw[i, j] = LGW_eucl(Ts[i], Ts[j], height_ref)
            else:
                lgw[i, j] = LGW_graph(Ts[i], Ts[j], Ms[i], Ms[j], height_ref)
    lgw += lgw.T
    et = time.time()
    return lgw, et - st


def LGW_graph(T1, T2, D1, D2, sigma):
    return np.sqrt(np.sum(np.multiply((D1[T1].T[T1].T - D2[T2].T[T2].T) ** 2, np.outer(sigma, sigma))))


def LGW_eucl(T1, T2, sigma, normalized=False):
    M1 = ot.dist(T1, T1, metric="euclidean")
    M2 = ot.dist(T2, T2, metric="euclidean")
    if normalized:
        M1 = M1 / np.max(M1)
        M2 = M2 / np.max(M2)
    return np.sqrt(np.sum(np.multiply((M1 - M2) ** 2, np.outer(sigma, sigma))))


def posns_n_nis_d(posns, heights):
    '''
    Given a positions and heights array, figure out (and make security checks for) the number of measures, number
    of support points array and dimension. Also modify posns to array, if given only one array for all measures.
    '''
    # if given a list, we assume that we are given multiple measures and the length of the list is
    # the number of measures
    if isinstance(posns, list):
        n = len(posns)
        assert len(heights) == n, "heights needs have same length as pos (equal number of measures)"
    # if given a 2d-array and a list of of height arrays, we assume that we are given a number of measures,
    # which are all supported on the same posns-array, so the number of measures n is len(heights)
    elif isinstance(posns, np.ndarray) and posns.ndim == 2 and (isinstance(heights, list) \
                                                                or (isinstance(heights,
                                                                               np.ndarray) and heights.ndim == 2)):
        n = len(heights)
        posns = [posns] * n
    # if given a 2d-array and a 1d-array of heights, assume that we are given only one measure
    elif isinstance(posns, np.ndarray) and posns.ndim == 2 and isinstance(heights, np.ndarray) and heights.ndim == 1:
        n = 1
        posns = [posns]
        heights = heights[None, :]
    else:
        raise ValueError("cannot see what the number of measures is")
    assert n >= 1, "at least one measure needs to be given"
    assert all([pos.ndim == 2 for pos in posns]), "position arrays need to be two-dimenional"
    nis = np.array([pos.shape[0] for pos in posns])  # number of support points for all measures
    d = posns[0].shape[1]
    return posns, heights, n, nis, d


def scatter_atomic(posns, heights, n_plots_per_col=2, scale=5, invert=False, disk_size=6000 / 5, xmarkers=True,
                   xmarker_posns=None, axis_off=False, margin_fac=0.2, savepath=None, cmaps=None):
    '''
    Produce scatter plots for given

    posns: list of length n of position-arrays of shape (n_i, d), where n_i is the number of atoms in measure i
    heights: array of shape (n, n_i), where n is the total number of measures
    '''
    posns, heights, n, nis, d = posns_n_nis_d(posns, heights)
    n_plots = n
    n_rows = np.ceil(n_plots / n_plots_per_col).astype(int)
    n_cols = min(n_plots_per_col, n_plots)
    figsize = (scale * n_cols, scale * n_rows)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    xmin, xmax, ymin, ymax = min([pos[:, 0].min() for pos in posns]), max([pos[:, 0].max() for pos in posns]), min(
        [pos[:, 1].min() for pos in posns]), max([pos[:, 1].max() for pos in posns])

    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_plots_per_col + j
            if idx >= n_plots or axis_off:
                ax[i, j].axis('off')
                if idx >= n_plots:
                    continue
            pos = posns[idx]
            height = heights[idx]

            # set plot dimensions
            xmargin = margin_fac * (xmax - xmin)
            ymargin = margin_fac * (ymax - ymin)
            ax[i, j].set_xlim([xmin - xmargin, xmax + xmargin])
            ax[i, j].set_ylim([ymin - ymargin, ymax + ymargin])
            ax[i, j].set_aspect('equal')

            # plot
            if xmarkers:
                ax[i, j].scatter(pos[:, 0], pos[:, 1], marker='x', c='red')
            if xmarker_posns is not None:
                ax[i, j].scatter(xmarker_posns[:, 0], xmarker_posns[:, 1], marker='x', c='red')

            if cmaps == None:
                cmap = ['gray'] * len(pos)
            else:
                cmap = cmaps[idx]
            ax[i, j].scatter(pos[:, 0], pos[:, 1], marker='o', s=height * disk_size * scale, c=cmap, alpha=0.5)
            if invert:
                ax[i, j].set_ylim(ax[i, j].get_ylim()[::-1])

    if savepath is not None:
        plt.savefig(savepath, dpi=300, pad_inches=0, bbox_inches='tight')


def img2atomic(img):
    # Creates a discrete measure from an image.
    assert img.ndim == 2, "img needs to be 2d array"
    x, y = img.shape
    pts = np.stack([grid.flatten() for grid in np.meshgrid(np.arange(x), y - np.arange(y))], axis=1)
    return pts[img.flatten() > 0], img.flatten()[img.flatten() > 0]


def plot_2d_shape_embedding(data, embedding, min_dist, figsize, cutoff=5, font_size=16, labels=None, save_path=None,
                            col=None, show_numbers=False, padwidth=2, return_img=False, axex=None):
    # Cut outliers
    n_pts = data.shape[0]
    n_dims = data.shape[1]
    low = [np.percentile(embedding[:, 0], q=cutoff), np.percentile(embedding[:, 1], q=cutoff)]
    high = [np.percentile(embedding[:, 0], q=100 - cutoff), np.percentile(embedding[:, 1], q=100 - cutoff)]
    cut_inds = np.arange(n_pts)[(embedding[:, 0] >= low[0]) * (embedding[:, 0] <= high[0])
                                * (embedding[:, 1] >= low[1]) * (embedding[:, 1] <= high[1])]

    data = data[cut_inds, :]
    embedding = embedding[cut_inds, :]

    # Visualize
    fig_x, fig_y = figsize
    fig_ratio = fig_x / fig_y
    # fig = plt.figure(figsize=(fig_x, fig_y))
    # ax = fig.add_subplot(111)

    # Plot images
    img_scale = 0.03
    pixels_per_dimension = int(np.sqrt(n_dims))

    x_size = (max(embedding[:, 0]) - min(embedding[:, 0])) * img_scale
    y_size = (max(embedding[:, 1]) - min(embedding[:, 1])) * img_scale * fig_ratio
    shown_images = np.array([[100., 100.]])

    if labels is not None:
        NUM_COLORS = len(np.unique(labels))
        cm = plt.get_cmap('gist_rainbow')
        unique_labels = np.unique(labels)

    for i in range(n_pts):
        #         dist = np.sqrt(np.sum((embedding[i] - shown_images) ** 2, axis=1))
        # don't show points that are too close
        #         if np.min(dist) < min_dist:
        #             continue
        #         shown_images = np.r_[shown_images, [embedding[i]]]
        x0 = embedding[i, 0] - (x_size / 2.)
        y0 = embedding[i, 1] - (y_size / 2.)
        x1 = embedding[i, 0] + (x_size / 2.)
        y1 = embedding[i, 1] + (y_size / 2.)
        if col is None:
            img = data[i, :].reshape(pixels_per_dimension, pixels_per_dimension)
        else:
            img = data[i, :].reshape(pixels_per_dimension, pixels_per_dimension, 3)
        # print(np.shape(data[i,:]))
        if labels is not None:
            j = list(unique_labels).index(labels[i])
            col_lab = cm(1. * j / NUM_COLORS)[:3]
            img = np.pad(img.astype(float), (padwidth, padwidth), "constant", constant_values=-1)
            img = np.array(
                [np.array([[x / 255, x / 255, x / 255] if x != -1 else col_lab for x in tmp]) for tmp in img])
            axex.imshow(img, aspect='auto', interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1),
                        cmap="viridiris")
        else:
            axex.imshow(img, aspect='auto', interpolation='nearest', zorder=100000, extent=(x0, x1, y0, y1))
        if show_numbers:
            plt.text(x1, y1, str(i), color="black",
                     fontdict={"fontsize": 10, "fontweight": 'bold', "ha": "left", "va": "baseline"})

    # scatter plot points
    axex.scatter(embedding[:, 0], embedding[:, 1], marker='.', s=150, alpha=0.5)
    axex.tick_params(axis='both', which='major', labelsize=font_size - 4)

    if save_path is not None:
        print("test")
        plt.savefig(save_path + ".png", transparent=True)
        plt.savefig(save_path + ".pdf", transparent=True)
    # plt.show()


def LGW(T1, T2, sigma, metric="sqsq_loss", normalized=False):
    if metric == "sqsq_loss":
        M1 = ot.dist(T1, T1)
        M2 = ot.dist(T2, T2)
        if normalized:
            M1 = M1 / np.max(M1)
            M2 = M2 / np.max(M2)
    elif metric == "sq_loss":
        M1 = ot.dist(T1, T1, metric="euclidean")
        M2 = ot.dist(T2, T2, metric="euclidean")
        if normalized:
            M1 = M1 / np.max(M1)
            M2 = M2 / np.max(M2)
    else:
        raise Exception("metric not known")
    return np.sqrt(np.sum(np.multiply((M1 - M2) ** 2, np.outer(sigma, sigma))))
    # return np.sum(np.multiply(np.linalg.norm(M1-M2)**2,np.outer(sigma,sigma)))


def vis_gw_plan(img1, pos1, pos2, height2, P, return_img=False, plot_img=True, sub_box=False):
    # Colour img1
    nx, ny = np.shape(img1)
    if sub_box:
        x_min = int(np.min(pos1[:, 0]))
        x_max = int(np.max(pos1[:, 0]))
        y_min = int(np.min(pos1[:, 1]))
        y_max = int(np.max(pos1[:, 1]))
    else:
        x_min = 0
        x_max = nx
        y_min = 0
        y_max = ny
    nx, ny = np.shape(img1)
    img1_col = np.zeros((nx, ny, 3))
    for i in range(nx):
        for j in range(ny):
            if img1[i, j] != 0:
                col = [1, (i - x_min) / (x_max - x_min), (j - y_min) / (y_max - y_min)]
                img1_col[i, j] = col

    # Calculate colours in img2 according to P
    n_pts1 = len(pos1)
    n_pts2 = len(pos2)
    counts = np.zeros(n_pts2)
    cols = np.zeros((n_pts2, 3))
    for j in range(n_pts2):
        for i in range(n_pts1):
            if P[i, j] != 0:
                counts[j] += 1
                x, y = pos1[i]
                # print(x,y)
                cols[j] += P[i, j] * img1_col[x, y]
                # print(cols[j])
        cols[j] = cols[j] / height2[j]

    # Colour img2
    img2_col = np.zeros((nx, ny, 3))
    for i in range(n_pts2):
        y, x = pos2[i]
        img2_col[x, y] = cols[i]

    # Plot
    if plot_img:
        fig, ax = plt.subplots(1, 2, figsize=(10, 20))
        ax[0].imshow(img1_col)
        ax[1].imshow(img2_col)
        plt.show()
    if return_img:
        return img1_col.clip(min=0, max=1), img2_col.clip(min=0, max=1)


def mm_space_from_img(img, metric="euclidean", normalize_meas=True):
    assert img.ndim == 2, "img needs to be 2d array"
    supp = np.dstack(np.where(img > 0))[0]
    height = img[supp[:, 0], supp[:, 1]]
    if normalize_meas:
        height /= np.sum(height)
    M = ot.dist(supp, supp, metric=metric)
    return supp, M, height