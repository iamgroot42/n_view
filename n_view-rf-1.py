
# coding: utf-8

# In[ ]:


import matplotlib
import matplotlib.pyplot as plt

from datasets import get_dataset

import numpy as np
np.random.seed(123)

import random
random.seed(123)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
set_session(tf.Session(config=config))

from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape
from keras.optimizers import Adadelta, SGD
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.cluster import KMeans,DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import label_binarize

import cv2
import progressbar
import os
from scipy import stats
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# In[147]:


n_views = 3
datasets = ['australian', 'bupa', 'colic', 'diabetes', 'german', 'ionosphere', 'kr-vs-kp', 'tic-tac-toe', 'vote', 'wdbc']
dataset = datasets[0]


# In[148]:


def find_optimal_n(L_x, noise_threshold=10, lower_cap=3, upper_cap=8):
    min_counts = []
    clusters = []
    for find_n in range(lower_cap, upper_cap+1):
        kmeans = AgglomerativeClustering(n_clusters=find_n).fit_predict(L_x)
        temp = np.array([])
        new_km = np.array([])
        _, counts = np.unique(kmeans, return_counts=True)
        for i in range(len(counts)):
            if counts[i] > noise_threshold:
                if temp.shape[0]==0:
                    temp = counts[i]
                    new_km = np.where(kmeans==i)
                else:
                    temp  = np.append(temo, counts[i], axis=0)
                    new_km = np.append(new_km, np.where(kmeans==i), axis=0)
        print min(temp)
        print new_km
        min_counts.append(min(temp))
        clusters.append(new_km)
    return lower_cap + np.argmax(min_counts), clusters[np.argmax(min_counts)]


# In[149]:


# 1. Partition L into (v_1, v_2, ..., v_n)
def get_dset(dataset):
    global n_views
    ds = get_dataset(dataset, 0.7, 0.25)
    [L_x, L_y], U, [test_x, test_y] = ds.get_data()
    L_y = np.argmax(L_y, axis=1)
    test_y = np.argmax(test_y, axis=1)
    n_views, kmeans = find_optimal_n(L_x)
    V = []
    for ind in range(n_views):
        left = int(ind * L_x.shape[0] / n_views)
        right = int((ind+1) * L_x.shape[0] / n_views)
        indices = np.where(kmeans == ind)
#         V.append([L_x[left:right], L_y[left:right]])
        V.append([L_x[indices], L_y[indices]])
    return ds, [L_x, L_y], U, [test_x, test_y], V


# In[150]:


# 2. Learn h_i on v_i using Learn
def get_classifiers(ds, L_x, L_y, U, test_x, test_y, V):
    H = []
    n_attr = V[0][0].shape[1]

    classifiers = [RandomForestClassifier(max_depth=2, random_state=0)
                   ,AdaBoostClassifier(),
                   GradientBoostingClassifier(),
                   KNeighborsClassifier()
                   ,MLPClassifier()]
    
    for ind in range(n_views):
        h = classifiers[ind]
        H.append(h)

    for ind in range(n_views):
        H[ind].fit(V[ind][0], V[ind][1])
        print accuracy_score(test_y, H[ind].predict(test_x))
    
    return H, n_attr


# In[151]:


# 3
def cotrain(ds, L_x, L_y, U, test_x, test_y, V, H, n_attr):
    changed = True
    to_plot = []
    num_runs = 5
    while (changed and num_runs <= 10):
        print num_runs
        preds_L = []
        for ind in range(n_views):
            preds_L.append(H[ind].predict(L_x))
        preds_L = np.array(preds_L)
        # preds_L = np.argmax(preds_L, axis=2)

        preds_U = []
        for ind in range(n_views):
            preds_U.append(H[ind].predict(U))
        preds_U = np.array(preds_U)
        # preds_U = np.argmax(preds_U, axis=2)

        perfs = []
        test_preds = []
        for ind in range(n_views):
            test_preds.append(H[ind].predict(test_x))
        test_preds = np.array(test_preds)

        for ind in range(n_views):
            perf = accuracy_score(test_y, test_preds[ind])
            perfs.append(perf)
        perfs.append(accuracy_score(test_y,
                                    np.reshape(stats.mode(test_preds, axis=0)[0], (-1,))))
        to_plot.append(perfs)

        Q = []
        update = [False for _ in range(n_views)]
        for cur in range(n_views):
            elems_take = [view_ind for view_ind in range(n_views) if view_ind != cur]
            preds_others_L = preds_L[elems_take]
            preds_others_U = preds_U[elems_take]
            # pdb.set_trace()
            acc_others_L = accuracy_score(L_y, 
                                          np.reshape(stats.mode(preds_others_L, axis=0)[0], (-1,)))
            acc_cur_L = accuracy_score(L_y, preds_L[cur])
            q_cur = [[], []]
            if acc_others_L > acc_cur_L:
                update[cur] = True
                for u_ind in range(preds_U.shape[1]):
                    mode_prediction = stats.mode(preds_others_U[:, u_ind])[0][0]
                    if np.sum(preds_others_U[:, u_ind] == mode_prediction) >= 0.5 * (n_views - 1):
                        q_cur[0].append(U[u_ind])
                        label_temp = mode_prediction
                        q_cur[1].append(label_temp)
            Q.append([np.array(q_cur[0]), np.array(q_cur[1])])
        for cur in range(n_views):
            if update[cur]:
                comb_x = np.concatenate([L_x, Q[cur][0]], axis=0)
                comb_y = np.concatenate([L_y, Q[cur][1]], axis=0)
                H[cur].fit(comb_x, comb_y)

        preds_L_new = []
        for ind in range(n_views):
            preds_L_new.append(H[ind].predict(L_x))

        preds_U_new = []
        for ind in range(n_views):
            preds_U_new.append(H[ind].predict(U))

        same = np.array_equal(preds_L, np.array(preds_L_new)) and np.array_equal(preds_U, np.array(preds_U_new))
        changed = not same
        num_runs += 1
    return to_plot


# In[152]:


def plot(to_plot, n_views, dset):
    plt.figure(figsize=(7, 7))
    plt.clf()
    handles = []
    labels = []
    for ind in range(n_views):
        ys = [x[ind] for x in to_plot]
        handle, = plt.plot(range(len(to_plot)), ys, marker='o', label = str(ind))
        handles.append(handle)
        labels.append('Classifier %d' % ind)
    ys = [x[n_views] for x in to_plot]
    handle, = plt.plot(range(len(to_plot)), ys, marker='o', label = 'Overall')
    handles.append(handle)
    labels.append('Max Voting')

    plt.legend(handles, labels, bbox_to_anchor=(1.04,1), loc="upper left")
    # plt.legend(handles, labels)
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Iterations vs accuracy for Dataset: %s' % dset)
    plt.show()


# In[153]:


for dset in ['australian', 'bupa', 'colic', 'diabetes', 'german', 'ionosphere', 'kr-vs-kp', 'tic-tac-toe', 'vote', 'wdbc']:
    ds, [L_x, L_y], U, [test_x, test_y], V = get_dset(dset)
    print L_x.shape, L_y.shape
    H, n_attr = get_classifiers(ds, L_x, L_y, U, test_x, test_y, V)
    to_plot = cotrain(ds, L_x, L_y, U, test_x, test_y, V, H, n_attr)
    plot(to_plot, n_views, dset)

