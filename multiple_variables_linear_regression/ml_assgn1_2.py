from load_data_ex2 import *
from normalize_features import *
from gradient_descent import *
from calculate_hypothesis import *
import os


figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# This loads our data
X, y = load_data_ex2()

# Normalize
X_normalized, mean_vec, std_vec, repeated_mean, repeated_std = normalize_features(X)

# After normalizing, we append a column of ones to X, as the bias term
column_of_ones = np.ones((X_normalized.shape[0], 1))
# append column to the dimension of columns (i.e., 1)
X_normalized = np.append(column_of_ones, X_normalized, axis=1)

# initialise trainable parameters theta, set learning rate alpha and number of iterations
theta = np.zeros((3))
alpha = 0.938  #0.938 #0.035
iterations = 100

# plot predictions for every iteration?
do_plot = True

# call the gradient descent function to obtain the trained parameters theta_final
theta_final, dic, min_cost, step = gradient_descent(X_normalized, y, theta, alpha, iterations, do_plot)
print(theta_final)


x1 = [1650, 3]
x2 = [3000, 4]

x1_normalized = (x1 - repeated_mean[0]) / repeated_std[0]
x1_normalized = np.append([1], x1_normalized)

x2_normalized = (x2 - repeated_mean[0]) / repeated_std[0]
x2_normalized = np.append([1], x2_normalized)

pred1 = np.dot(x1_normalized, theta_final)
print("Prediction of sample: ", x1, " --> ", pred1)

pred2 = np.dot(x2_normalized, theta_final)
print("Prediction of sample: ", x2, " --> ", pred2)



