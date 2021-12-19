#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:55:08 2021

@author: ishaanchaturvedi
"""

alpha_dic = {}
for i in range(0,100):
    alpha = i/1000
    #alpha = 10**i
    print("ALPHA:", alpha)
    theta_final, cost_vector = gradient_descent(X_normalized, y, theta, alpha, iterations)
    print(theta_final, cost_vector)
    print("Lowest cost: ", cost_vector.min())
    alpha_dic[i] = {}
    alpha_dic[i]['alpha'] = alpha
    alpha_dic[i]['min_cost'] = cost_vector.min()

import pandas as pd
alpha_df = pd.DataFrame(alpha_dic).transpose()

alpha_df.min_cost.min()

alpha_df.min_cost.idxmin()



############################################################################

alpha_dic = {}
for i in range(0,100):
    alpha = i/1000
    #alpha = 10**i
    print("ALPHA:", alpha)
    theta_final, cost_vector_train, cost_vector_test = gradient_descent_training(X_train_normalized, y_train, X_test_normalized, y_test, theta, alpha, iterations)
    min_train_cost = np.min(cost_vector_train)
    argmin_train_cost = np.argmin(cost_vector_train)
    min_test_cost = np.min(cost_vector_test)
    argmin_test_cost = np.argmin(cost_vector_test)
    print('Final train cost: {:.5f}'.format(cost_vector_train[-1]))
    print('Minimum train cost: {:.5f}, on iteration #{}'.format(min_train_cost, argmin_train_cost+1))
    print('Final test cost: {:.5f}'.format(cost_vector_test[-1]))
    print('Minimum test cost: {:.5f}, on iteration #{}'.format(min_test_cost, argmin_test_cost+1))

    alpha_dic[i] = {}
    alpha_dic[i]['alpha'] = alpha
    alpha_dic[i]['min_cost_train'] = cost_vector_train.min()
    alpha_dic[i]['min_cost_test'] = cost_vector_test.min()

import pandas as pd
alpha_df = pd.DataFrame(alpha_dic).transpose()

alpha_df.min_cost_train.plot()

alpha_df.min_cost_train.min()

alpha_df.min_cost_train.idxmin()

alpha_df.min_cost_test.min()

alpha_df.min_cost_test.idxmin()





