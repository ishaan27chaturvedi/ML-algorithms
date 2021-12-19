#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 15:11:36 2021

@author: ishaanchaturvedi
"""



import matplotlib.pyplot as plt

fig, ax1 = plt.subplots()
# clear subplot from previous (if any) drawn stuff
ax1.clear()
# set label of horizontal axis
ax1.set_xlabel('x1')
# set label of vertical axis
ax1.set_ylabel('y=f(x1)')
# scatter the points representing the groundtruth prices of the training samples, with red color
ax1.scatter(X_normalized[:,1], y, c='red', marker='x', label='groundtruth')
# scatter the points representing the predicted prices, with blue color
ax1.scatter(X_normalized[:,1], hypothesis_to_vector(X_normalized, theta_final), c='blue', marker='+', label='prediction')

ax1.scatter(x1_normalized[1], pred1, c='black', marker='o', s = 90)
ax1.scatter(x1_normalized[1], pred1, c='yellow', marker='o', label='1650_prediction', s = 60)
ax1.scatter(x2_normalized[1], pred2, c='black', marker='o', label='3000_prediction', s = 80)

# add legend to the subplot
ax1.legend()





########################################/


# ROUGH

import pandas as pd

dff = pd.DataFrame(X, columns = ['foot', 'rooms'])

df.loc[df.foot<1700]

# [1, 22, 40]
[329900, 242900, 368500]




new_X = np.append(X, [x1,x2], axis =0)

new_X_normalized, new_mean_vec, new_std_vec = normalize_features(new_X)
# After normalizing, we append a column of ones to X, as the bias term
new_column_of_ones = np.ones((new_X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
new_X_normalized = np.append(new_column_of_ones, new_X_normalized, axis=1)


print("Prediction of sample: ", x1, " --> ", calculate_hypothesis(new_X_normalized, theta_final, len(new_X_normalized)-2))
print("Prediction of sample: ", x2, " --> ", calculate_hypothesis(new_X_normalized, theta_final, len(new_X_normalized)-1))

#############################################



import pandas as pd
df = pd.DataFrame(dic).transpose()

df.plot()


dic2 = {}
for i in range(150, 1250):
    alpha = i/1000
    print("ALPHA:", alpha)
    theta_final, dic, min_cos, step = gradient_descent(X_normalized, y, theta, alpha, iterations, do_plot)
    dic2[i] = {}
    dic2[i]['alpha'] = alpha
    dic2[i]['min_cost'] = min_cost
    dic2[i]['step'] = step

df4 = pd.DataFrame(dic2).transpose()

