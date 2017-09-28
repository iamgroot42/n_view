import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
np.random.seed(123)

import random
random.seed(123)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape
from keras.optimizers import Adadelta
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import label_binarize
import cv2
import pdb
import progressbar
import os
from scipy import stats
from datasets import get_dataset

ds = get_dataset("bupa", 0.7, 0.25)
[L_x, L_y], U, [test_x, test_y] = ds.get_data()
U = np.array(U)
n_views = 5
views = []
for ind in range(n_views):
    views.append([np.array(L_x[ind * n_views:(ind + 1) * n_views]), np.array(L_y[ind * n_views:(ind + 1) * n_views])])
for ind in range(n_views):
    print views[ind][0].shape, views[ind][1].shape

# Define Models
models = []
n_attr = views[ind][0].shape[1]
for ind in range(n_views):
    model = Sequential()
    model.add(Dense(input_shape=(n_attr,), units=n_attr / 2))
    model.add(Activation('relu'))
    model.add(Dense(units=n_attr/5))
    model.add(Activation('relu'))
    model.add(Dense(units=views[ind][1].shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    models.append(model)
print models[0].summary()

# Train models on Labelled Data
for ind in range(n_views):
    models[ind].fit(views[ind][0], views[ind][1], epochs=100, batch_size = 64, validation_split = 0.2, 
              callbacks=[EarlyStopping(monitor='val_acc', patience=5)])

# Run Co-Training Algorithm 1
# Simple majority voting over all the classifiers for the unlabelled example
L = views[0]
for ind in range(1, n_views):
    L[0] = np.concatenate([L[0], views[ind][0]], axis = 0)
    L[1] = np.concatenate([L[1], views[ind][1]], axis = 0)
changed = False
pred_modes = None
preds = np.zeros((U.shape[0], n_views))
for ind in range(n_views):
    preds[:, ind] = np.argmax(models[ind].predict(U), axis = 1)
while (not changed):
    pred_modes_new, counts = stats.mode(preds, axis=1)
    if pred_modes_new != pred_modes:
        changed = True
        pred_modes = pred_modes_new
        sel = np.array(np.argmax(pred_modes, axis=0), dtype=int)
        sel_one_hot = label_binarize([pred_modes[sel].squeeze()], classes=range(len(L_y[0]) + 1))[:, :-1]
        print sel_one_hot
        L[0] = np.concatenate([L[0], U[sel]], axis = 0)
        L[1] = np.concatenate([L[1], sel_one_hot], axis = 0)
    for ind in range(n_views):
        models[ind].fit(L[0], L[1], epochs=100, batch_size = 64, validation_split = 0.2, 
              callbacks=[EarlyStopping(monitor='val_acc', patience=5)])
perf = [None for _ in range(n_views)]
for ind in range(n_views):
    perf[ind] = model.evaluate(np.array(test_x), np.array(test_y))
    print perf[ind]
