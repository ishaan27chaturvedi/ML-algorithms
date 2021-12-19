import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
X_full[:,0] = data['f1']
X_full[:,1] = data['f2']

########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 3

#########################################
# Write your code here

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

X_phonemes_1_2 = X_full[np.where(np.isin(phoneme_id, [1,2]))]  
true_labels = phoneme_id[np.where(np.isin(phoneme_id, [1,2]))]  

########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"

X = X_phonemes_1_2.copy()


k = 3
p_id = 1
npy_filename = 'data/GMM_params_phoneme_{:02}_k_{:02}.npy'.format(p_id, k)
model_3_1 = np.load(npy_filename, allow_pickle=True)
model = model_3_1.item()
Z_3_1 = get_predictions(model['mu'], model['s'], model['p'], X)
preds_3_1 = Z_3_1.sum(axis=1).reshape(304, 1)

p_id = 2
npy_filename = 'data/GMM_params_phoneme_{:02}_k_{:02}.npy'.format(p_id, k)
model_3_2 = np.load(npy_filename, allow_pickle=True)
model = model_3_2.item()
Z_3_2 = get_predictions(model['mu'], model['s'], model['p'], X)
preds_3_2 = Z_3_2.sum(axis=1).reshape(304, 1)


preds = np.append(preds_3_1, preds_3_2, axis = 1)

label_dict = {0:1, 1:2}
pred_labels = preds.argmax(axis=1)
pred_labels = np.vectorize(label_dict.get)(pred_labels)

import pandas as pd
net_data = pd.DataFrame({'pred':pred_labels, 'true':true_labels}, columns=['pred', 'true'])

net_data['accuracy'] = net_data['pred'] == net_data['true']
net_data.accuracy.value_counts().plot.pie(autopct='%1.1f%%')

net_data['mode_3_1'] = preds_3_1
net_data['mode_3_2'] = preds_3_2
net_data[['mode_3_1', 'mode_3_2']].plot()

########################################/

accuracy = net_data.accuracy.sum()*100/net_data.shape[0]
print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################


k = 6
p_id = 1
npy_filename = 'data/GMM_params_phoneme_{:02}_k_{:02}.npy'.format(p_id, k)
model_6_1 = np.load(npy_filename, allow_pickle=True)
model = model_6_1.item()
Z_6_1 = get_predictions(model['mu'], model['s'], model['p'], X)
preds_6_1 = Z_6_1.sum(axis=1).reshape(304, 1)

p_id = 2
npy_filename = 'data/GMM_params_phoneme_{:02}_k_{:02}.npy'.format(p_id, k)
model_6_2 = np.load(npy_filename, allow_pickle=True)
model = model_6_2.item()
Z_6_2 = get_predictions(model['mu'], model['s'], model['p'], X)
preds_6_2 = Z_6_2.sum(axis=1).reshape(304, 1)


preds6 = np.append(preds_6_1, preds_6_2, axis = 1)

label_dict = {0:1, 1:2}
pred_labels6 = preds6.argmax(axis=1)
pred_labels6 = np.vectorize(label_dict.get)(pred_labels6)

import pandas as pd
net_data6 = pd.DataFrame({'pred':pred_labels6, 'true':true_labels}, columns=['pred', 'true'])

net_data6['accuracy'] = net_data6['pred'] == net_data6['true']
net_data6.accuracy.value_counts().plot.pie(autopct='%1.1f%%')

net_data6['mode_6_1'] = preds_6_1
net_data6['mode_6_2'] = preds_6_2
net_data6[['mode_6_1', 'mode_6_2']].plot()

########################################/

accuracy = net_data6.accuracy.sum()*100/net_data6.shape[0]
print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################

net = pd.concat([net_data, net_data6], axis =1)

net[['mode_3_1', 'mode_6_1']].plot()
net[['mode_3_2', 'mode_6_2']].plot()

########################################/

#################################
net_data['type'] = net_data.accuracy.astype(str) + '_' + net_data.pred.astype(str)

net_data.type.value_counts().plot.pie(autopct='%1.1f%%')

colors = net_data.type.map({'True_1':'tab:blue', 'True_2': 'tab:orange', 'False_1': 'tab:red', 'False_2': 'tab:green'}).values


fig, ax1 = plt.subplots()
X=X_phonemes_1_2
title_string=title_string
ax=ax1

# clear subplot from previous (if any) drawn stuff
ax.clear()
# set label of horizontal axis
ax.set_xlabel('f1')
# set label of vertical axis
ax.set_ylabel('f2')
# scatter the points, with red color
ax.scatter(X[:,0], X[:,1], c=colors, marker='.', label=title_string)
# add legend to the subplot

#############################################

net_data6['type'] = net_data6.accuracy.astype(str) + '_' + net_data6.pred.astype(str)
net_data6.type.value_counts().plot.pie(autopct='%1.1f%%')

colors6 = net_data6.type.map({'True_1':'tab:blue', 'True_2': 'tab:orange', 'False_1': 'tab:red', 'False_2': 'tab:green'}).values


fig, ax1 = plt.subplots()
X=X_phonemes_1_2
title_string=title_string
ax=ax1

# clear subplot from previous (if any) drawn stuff
ax.clear()
# set label of horizontal axis
ax.set_xlabel('f1')
# set label of vertical axis
ax.set_ylabel('f2')
# scatter the points, with red color
ax.scatter(X[:,0], X[:,1], c=colors6, marker='.', label=title_string)
# add legend to the subplot




# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()