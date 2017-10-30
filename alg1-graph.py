import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import get_dataset

import numpy as np
# np.random.seed(123)

import random
# random.seed(123)

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))

from keras.utils import plot_model
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, BatchNormalization, Dropout, Reshape
from keras.optimizers import Adadelta, SGD
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import label_binarize
import cv2
import pdb
import progressbar
import os
from scipy import stats

import argparse

parser = argparse.ArgumentParser(description='n-view co-learning')
parser.add_argument('--n_views', type=int)
parser.add_argument('--dataset', type=str)
parser.add_argument('--verbose', type=int)
args = parser.parse_args()

ds = get_dataset(args.dataset, 0.7, 0.25)
[L_x, L_y], U, [test_x, test_y] = ds.get_data()
n_views = args.n_views
views = []

for ind in range(n_views):
    left = int(ind * L_x.shape[0] / n_views)
    right = int((ind+1) * L_x.shape[0] / n_views)
    views.append([L_x[left:right], L_y[left:right]])

#for ind in range(n_views):
#    print views[ind][0].shape, views[ind][1].shape

# Define Models
models = []
n_attr = views[ind][0].shape[1]

for ind in range(n_views):
    model = Sequential()
    model.add(Dense(input_shape=(n_attr,), units=n_attr / 2))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=n_attr/5))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dense(units=views[ind][1].shape[1]))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])
    models.append(model)
#print models[0].summary()

# Train models on Labelled Data
for ind in range(n_views):
    models[ind].fit(views[ind][0], views[ind][1], epochs=50, batch_size = 4, validation_split = 0.2, verbose=args.verbose)
              # ,callbacks=[EarlyStopping(monitor='val_acc', patience=5)])
    print model.evaluate(test_x,test_y)

# Run Co-Training Algorithm 1
# Simple majority voting over all the classifiers for the unlabelled example
L = views[0]
for ind in range(1, n_views):
    L[0] = np.concatenate([L[0], views[ind][0]], axis = 0)
    L[1] = np.concatenate([L[1], views[ind][1]], axis = 0)
changed = True
pred_modes = []
preds = np.zeros((U.shape[0], n_views))

for ind in range(n_views):
    preds[:, ind] = np.argmax(models[ind].predict(U, verbose=args.verbose), axis = 1)

to_plot = []
num_runs = 0
while (changed):
    if num_runs == 5:
        break
    num_runs += 1
    pred_modes = stats.mode(preds, axis=1)[0]
    changed=False

    for ind in range(n_views):
        models[ind].fit(L[0], L[1], epochs=20, batch_size = 4, validation_split = 0.2, verbose=args.verbose)

    for ind in range(n_views):
        preds[:, ind] = np.argmax(models[ind].predict(U, verbose=args.verbose), axis = 1)

    pred_modes_new = stats.mode(preds, axis=1)[0]
    #print pred_modes_new, "okk"
    #print pred_modes, "kko"

    if not np.array_equal(pred_modes_new, pred_modes):
        changed = True
        counts = pred_modes_new
        sel = np.array(np.argmax(pred_modes, axis=0), dtype=int)
        sel_one_hot = label_binarize([pred_modes[sel].squeeze()], classes=range(len(L_y[0]) + 1))[:, :-1]
        L[0] = np.concatenate([L[0], U[sel]], axis = 0)
        L[1] = np.concatenate([L[1], sel_one_hot], axis = 0)
	mask = np.ones(U.shape[0])
	mask[sel[0]] = 0
	U = U[mask==1]
	preds = preds[mask==1]
	print "Unlabelled pool left:",U.shape[0]
    
    perfs = []
    for ind in range(n_views):
        perf = models[ind].evaluate(test_x,test_y)
        perfs.append(perf[0])
    to_plot.append(perfs)

handles = []
labels = []
# to_plot = [[1, 2, 3, 5, 6], [2, 2, 3, 3, 4], [3, 3, 3, 0, 1]]
for ind in range(args.n_views):
    ys = [x[ind] for x in to_plot]
    handle, = plt.plot(range(len(to_plot)), ys, marker='o', label = str(ind))
    handles.append(handle)
    labels.append('Classifier %d' % ind)
plt.legend(handles, labels)
plt.savefig('fig/%s_%d.png' % (args.dataset, args.n_views))
# plt.show()