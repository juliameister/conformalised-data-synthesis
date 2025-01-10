import cv2
import numpy as np
np.random.seed(42)
import math
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import tables
import tensorflow as tf
from tqdm.notebook import tqdm
from umap import UMAP
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def cp_split(x_train, y_train, prop_size=0.33):
    x_prop, x_calib, y_prop, y_calib = train_test_split(x_train, y_train, test_size=prop_size, shuffle=True, stratify=y_train, random_state=42)
    return x_prop, y_prop, x_calib, y_calib


def get_label_conditional_knns(x_prop, y_prop):
    dists_knns = dict()
    labels_unique = np.unique(y_prop)
    for l in labels_unique:
        dists_knns[l] = {
            "same_class": KNeighborsClassifier().fit(x_prop[y_prop==l], np.ones((x_prop[y_prop==l].shape[0])))
        }
    return dists_knns


def neighbor_distance(xs_test, knns, k=5):
    dists_same_class = knns["same_class"].kneighbors(xs_test, n_neighbors=k, return_distance=True)[0].sum(axis=1)
    return dists_same_class


def get_ncm_scores(xs, knns):
    xs_scores = dict()
    for i in tqdm(list(knns.keys()), desc="NCMs for labels"):
        xs_scores[i] = neighbor_distance(xs, knns[i])
    return xs_scores


def calc_pvalues(calib_ncms, test_ncms):
    calib_ncms.sort()
    js = np.searchsorted(calib_ncms, test_ncms, side='left')
    rank = len(calib_ncms) - js + 1
    ps = rank / (len(calib_ncms) + 1)
    return ps


def get_pvalues(calib_ys, calib_ncms, test_ncms):
    pvalues = dict()
    for label in tqdm(list(calib_ncms.keys()), desc="p-values for labels"):
        pvalues[label] = calc_pvalues(calib_ncms[label][calib_ys==label], test_ncms[label])  # MICP
    return pvalues


def show_confidence_feature_space(epsilon, grid_xs, grid_ys, ps_grid_target, x_train_source=None, y_train_source=None, x_train_target=None, y_train_target=None, title=None, target_name="", source_name=""):
    cols = 3
    labels = list(ps_grid_target.keys())
    rows = math.ceil(len(labels)/cols)
    figsize = (20, 5*rows)
    fig, axs = plt.subplots(rows, cols, sharey=True, sharex=True, figsize=figsize)
    fig.set_tight_layout(True)
    try:
        sns.scatterplot(x=x_train_target[:, 0], y=x_train_target[:, 1], s=5, hue=y_train_target, palette="hls", ax=axs[0][0])
    except:  # if rows==1
        sns.scatterplot(x=x_train_target[:, 0], y=x_train_target[:, 1], s=5, hue=y_train_target, palette="hls", ax=axs[0])

    for i, l in enumerate(labels):
        ax_x = math.floor((i+1)/cols)
        ax_y = (i+1)%cols

        try:
            axs[ax_x, ax_y].set_title("Class {}".format(i))
            axs[ax_x, ax_y].contourf(grid_xs, grid_ys, (ps_grid_target[i]>epsilon).reshape(len(grid_ys), len(grid_xs), order='F'))
            if x_train_target.any():
                sns.scatterplot(x=x_train_target[:, 0][y_train_target==l], y=x_train_target[:, 1][y_train_target==l], s=10, color='green', ax=axs[ax_x][ax_y], alpha=0.6, label=target_name)
            if x_train_source.any():
                sns.scatterplot(x=x_train_source[:, 0][y_train_source==l], y=x_train_source[:, 1][y_train_source==l], s=10, color='red', ax=axs[ax_x][ax_y], alpha=0.6, label=source_name)
        except:  # if rows==1
            axs[ax_y].set_title("Class {}".format(i))
            axs[ax_y].contourf(grid_xs, grid_ys, (ps_grid_target[i]>epsilon).reshape(len(grid_ys), len(grid_xs), order='F'))
            if x_train_target.any():
                sns.scatterplot(x=x_train_target[:, 0][y_train_target==l], y=x_train_target[:, 1][y_train_target==l], s=10, color='green', ax=axs[ax_y], alpha=0.6, label=target_name)
            if x_train_source.any():
                sns.scatterplot(x=x_train_source[:, 0][y_train_source==l], y=x_train_source[:, 1][y_train_source==l], s=10, color='red', ax=axs[ax_y], alpha=0.6, label=source_name)

    if title:  plt.suptitle(title)
    if target_name!="" or source_name!="":  plt.legend()
    plt.show()


def cartesian(arrays, out=None):
    # https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)
    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out
