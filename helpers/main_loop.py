from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
np.random.seed(42)
import os
import pickle

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import resample, shuffle

from helpers import (cp_split, cartesian, get_label_conditional_knns, \
                     get_ncm_scores, get_pvalues, show_confidence_feature_space)

from nn_models import FFNN
import tensorflow as tf


def set_seed(tf_seed):
    np.random.seed(42)
    tf.random.set_seed(tf_seed)
    tf.keras.utils.set_random_seed(tf_seed)


def reset_session(tf_seed):
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    set_seed(tf_seed)


def run_experiment(setup_dict, is_original=False, is_o_s=False, per_class=[], is_synthetic=False, epsilon=None, tf_seed=1, epochs=10, batch_size=64):
    set_seed(tf_seed)

    if is_original:
        model = FFNN(setup_dict['X_train'].shape[1:], len(np.unique(setup_dict['y_train'])), is_graph=False, \
             epochs=epochs, batch_size=batch_size, validation_split=0.3)
        ml_classification_report(setup_dict['X_train'], setup_dict['y_train'], setup_dict['X_test'], setup_dict['y_test'], "\n\nBaseline results: ORIGINAL", model=model)
        del model
        reset_session(tf_seed)

    if (is_o_s or is_synthetic):
        show_confidence_feature_space(epsilon, setup_dict['grid_arrays'][0], setup_dict['grid_arrays'][1], setup_dict['ps_grid'], x_train_target=setup_dict['X_prop'], y_train_target=setup_dict['y_prop'], x_train_source=setup_dict['X_calib'], y_train_source=setup_dict['y_calib'], title="$\epsilon$ = {}".format(epsilon), target_name="Prop", source_name="Calib")

        synthetic_samples = dict()
        print("Synthetic samples per class")
        for i in np.sort(np.unique(setup_dict['y_train'])):
            synthetic_samples[i] = setup_dict['grid_points'][setup_dict['ps_grid'][i]>epsilon]
            print("Label {}:".format(i), synthetic_samples[i].shape)

        X_train_syn = None
        y_train_syn = None
        for i in np.sort(np.unique(setup_dict['y_train'])):
            if not np.any(X_train_syn):
                X_train_syn = synthetic_samples[i]
                y_train_syn = np.full((synthetic_samples[i].shape[0],), i)
            else:
                X_train_syn = np.vstack((X_train_syn, synthetic_samples[i]))
                y_train_syn = np.hstack((y_train_syn, np.full((synthetic_samples[i].shape[0],), i)))

    if is_o_s:
        index = shuffle(range(len(setup_dict['y_train'])+len(y_train_syn)), random_state=42)
        X_train_all = np.vstack((setup_dict['X_train'], X_train_syn))[index]
        y_train_all = np.hstack((setup_dict['y_train'], y_train_syn))[index]
        print("\n\nTotal synthetic samples:", X_train_syn.shape)

        model = FFNN(setup_dict['X_train'].shape[1:], len(np.unique(setup_dict['y_train'])), is_graph=False, \
             epochs=10, batch_size=64, validation_split=0.3)
        ml_classification_report(X_train_all, y_train_all, setup_dict['X_test'], setup_dict['y_test'], "ORIGINAL + SYNTHETIC", model=model)
        del model
        reset_session(tf_seed)

    for i in per_class:
        index = shuffle(range(len(setup_dict['y_train'])+len(synthetic_samples[i])), random_state=42)
        X_train_ex = np.vstack((setup_dict['X_train'], synthetic_samples[i]))[index]
        y_train_ex = np.hstack((setup_dict['y_train'], np.full((synthetic_samples[i].shape[0],), i)))[index]
        print("\n\nCLASS", i, "| Total:", X_train_ex.shape, "(synthetic", synthetic_samples[i].shape, ")")

        model = FFNN(setup_dict['X_train'].shape[1:], len(np.unique(setup_dict['y_train'])), is_graph=False, \
             epochs=10, batch_size=batch_size, validation_split=0.3)
        ml_classification_report(X_train_ex, y_train_ex, setup_dict['X_test'], setup_dict['y_test'], "ORIGINAL+SYNTHETIC (CLASS {})".format(i), model=model)
        del model
        reset_session(tf_seed)

    if is_synthetic:
        index = shuffle(range(len(y_train_syn)), random_state=42)
        print("\n\nTotal synthetic samples:", X_train_syn.shape)

        model = FFNN(setup_dict['X_train'].shape[1:], len(np.unique(setup_dict['y_train'])), is_graph=False, \
             epochs=10, batch_size=64, validation_split=0.3)
        ml_classification_report(X_train_syn[index], y_train_syn[index], setup_dict['X_test'], setup_dict['y_test'], "SYNTHETIC ONLY", model=model)
        del model
        reset_session(tf_seed)


def setup_experiment(data_name, grid_step=0.1, nr_train_samples='all', prop_size=0.33):
    X_train, y_train, X_test, y_test = load_data_pickle(data_name)
    grid_points, grid_arrays = prepare_grid(X_train, grid_step)
    print("Done preparing grid:", grid_points.shape)

    if nr_train_samples!="all":
        if type(nr_train_samples)==dict:  # resample individual classes
            class_key = list(nr_train_samples.keys())[0]
            map0 = y_train!=class_key
            map1 = y_train==class_key
            X_train_resampled, y_train_resampled = resample(X_train[map1, :], y_train[map1], replace=False, n_samples=nr_train_samples[class_key], random_state=42)
            X_train = np.vstack((X_train[map0, :], X_train_resampled))
            y_train = np.hstack((y_train[map0], y_train_resampled))
            X_train, y_train = shuffle(X_train, y_train, random_state=42)
        else:  # resample the entire dataset, maintaining the class balance
            X_train, y_train = resample(X_train, y_train, replace=False, n_samples=nr_train_samples, stratify=y_train, random_state=42)

    # Identify confidence regions
    X_prop, y_prop, X_calib, y_calib = cp_split(X_train, y_train, prop_size=prop_size)
    dists_knns = get_label_conditional_knns(X_prop, y_prop)

    ncm_calib = get_ncm_scores(X_calib, dists_knns)
    ncm_grid = get_ncm_scores(grid_points, dists_knns)
    ps_grid = get_pvalues(y_calib, ncm_calib, ncm_grid)

    # visualise confidence regions
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    temp0 = ax0.contourf(grid_arrays[0], grid_arrays[1], (ps_grid[0]).reshape(len(grid_arrays[1]), len(grid_arrays[0]), order='F'), levels)
    fig.colorbar(temp0, label="$p$-values", format=FormatStrFormatter('%.2f'))
    ax0.set_xlabel("Feature 1")
    ax0.set_ylabel("Feature 2")
    ax0.set_title("Class 0")

    levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    temp1 = ax1.contourf(grid_arrays[0], grid_arrays[1], (ps_grid[1]).reshape(len(grid_arrays[1]), len(grid_arrays[0]), order='F'), levels)
    fig.colorbar(temp1, label="$p$-values", format=FormatStrFormatter('%.2f'))
    ax1.set_xlabel("Feature 1")
    ax1.set_ylabel("Feature 2")
    ax1.set_title("Class 1")
    plt.plot()

    setup_dict = {
        'grid_arrays': grid_arrays,
        'grid_points': grid_points, 'ps_grid': ps_grid,
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test, 'y_test': y_test,
        'X_prop': X_prop, 'y_prop': y_prop,
        'X_calib': X_calib, 'y_calib': y_calib,
    }

    return setup_dict


def prepare_grid(X_train_target, gridstep):
    dims = X_train_target.shape[1]
    grid_mins = np.floor(np.min(X_train_target, axis=0))-1
    grid_maxs = np.ceil(np.max(X_train_target, axis=0))+1

    grid_arrays = dict()
    for i in range(dims):
        grid_arrays[i] = np.arange(grid_mins[i], grid_maxs[i], step=gridstep)
    grid_points = cartesian(tuple(v for (k, v) in grid_arrays.items()))

    return grid_points, grid_arrays


def load_data_pickle(data_name):
    filename = "./../data/{}.pickle".format(data_name)
    data = pickle.load(open(filename, 'rb'))
    print("Data loaded: {}".format(filename))
    return data['X_train'], data['y_train'], data['X_test'], data['y_test']


def ml_classification_report(X_train, y_train, X_test, y_test, name=None, model='KNN'):
    if name:  print("{}\n==========================================".format(name))

    if type(model)==str:
        if model=='KNN':
            model = KNeighborsClassifier().fit(X_train, y_train)
        elif model=='LR':
            model = LogisticRegression().fit(X_train, y_train)
        elif model=='SVM':
            model = SVC().fit(X_train, y_train)
        elif model=='RF':
            model = RandomForestClassifier().fit(X_train, y_train)
        else:
            print("ERROR: Model name not recognised")
    else:  # FFNN
        model.fit(X_train, y_train)

    print(classification_report(y_test, model.predict(X_test), zero_division=0))
