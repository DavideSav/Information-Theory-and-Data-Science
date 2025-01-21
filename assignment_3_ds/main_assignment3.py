#! /usr/bin/env python3
# Author: Davide Savini (0310795)

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import assignment_1_ds.myfunctions as myfunc1
import assignment_2_ds.myfunc_assign2 as myfunc2
import myfunc_assign3 as myfunc3

# Import of Iris dataset
iris = datasets.load_iris()
# data_matrix is array[x,y] (2D) with len 150 and size 600 (150x4) 150 instances (rows) and 4 features (col)
# class_vector is array[x] (1D) with len and size 150 (150x1) 150 instances (rows) linked with the prev. one.
data_matrix, class_vector = iris.data, iris.target

# Information about Iris dataset (test_pdf_multivariate_iris.py)
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
classes = ['Setosa', 'Versicolour', 'Virginica']
# select a triplet of features from the overall set of 4 features
first_feature = 0  # Sepal Length
second_feature = 1  # Sepal Width
third_feature = 2  # Petal Length
fourth_feature = 3  # Petal Width

# Sepal Length - Sepal Width - Petal Length
fig1 = plt.figure(1)
ax = plt.axes(projection='3d')
colors_of_classes = ListedColormap(['r', 'b', 'y'])
sc = ax.scatter(data_matrix[:, first_feature], data_matrix[:, second_feature], data_matrix[:, third_feature],
                c=class_vector, cmap=colors_of_classes, edgecolor='k', s=40)
ax.set_xlabel(features[first_feature])
ax.set_ylabel(features[second_feature])
ax.set_zlabel(features[third_feature])
plt.title("Distribution of Iris dataset respect associated classes\n"
          "Sepal Length - Sepal Width - Petal Length")

# Sepal Length - Sepal Width - Petal Width
fig11 = plt.figure(31)
ax = plt.axes(projection='3d')
colors_of_classes = ListedColormap(['r', 'b', 'g'])
sc1 = ax.scatter(data_matrix[:, first_feature], data_matrix[:, second_feature], data_matrix[:, fourth_feature],
                 c=class_vector, cmap=colors_of_classes, edgecolor='k', s=40)
ax.set_xlabel(features[first_feature])
ax.set_ylabel(features[second_feature])
ax.set_zlabel(features[fourth_feature])
plt.title("Distribution of Iris dataset respect associated classes\n"
          "Sepal Length - Sepal Width - Petal Width")


# ################### Assignment #3 ###################
print("\n################### Assignment #3 ###################")
# (1) Compute the probability mass function of the features of the Iris dataset
#     after having discretized the samples as integers.
# (*) Note: Start from the example in pmf estimation multivariate.py and test pdf multivariate iris.py

# Discretization of samples as integers.
data_discr = (data_matrix*10).astype(int)  # discrete to integer values
# P.M.F. computation - Loop to print the all results
samp_list_all = []
pmf_data_all = []
for i in range(0, 4):  # i=3 is the last index of loop
    samp_list_feat, pmf_data_feat = myfunc2.pmf_univariate(data_discr[:, i])
    print("# The estimated p.m.f. of", features[i],
          "feature with respective samples", samp_list_feat, 'is: \n', pmf_data_feat)
    # Plot of estimated p.m.f. in 2D
    fig2 = plt.figure(2)
    plt.subplot(2, 2, i+1)  # (nrows, ncol, index that means: "1, this is the first plot; 2, this is the 2nd etc..")
    plt.stem(samp_list_feat, pmf_data_feat)
    # plt.xlabel("x")
    plt.ylabel("p.m.f.")
    plt.title(features[i]+" p.m.f.")
    plt.grid(color='0.80')
    # Storage of all p.m.f. in a list
    samp_list_all.append([samp_list_feat])
    pmf_data_all.append([pmf_data_feat])
print("\n--------------------------------------------------------------")

# (2) Compute the entropy of the features of the Iris dataset (discrete entropy).
# I call my discrete entropy function by assignment_1_ds.myfunctions.py
entr_all = []
for i in range(0, 4):
    entr_single_col = myfunc1.discrete_entropy(pmf_data_all[i][0])
    print("# The Discrete Entropy of", features[i], "feature is: \n", entr_single_col, "[bit]")
    # Storage of all entropies in a list
    entr_all.append([entr_single_col])
print("\n--------------------------------------------------------------")

# (3) Compute the mutual information between any pair of features of the
#     Iris dataset (discrete mutual information).

# I did loops to make all possible cases of joint features including same cases like:
# X as Sepal Length and Y as Sepal Length (so X=Y);
data_2d_list = []
# I print 16 mutual information considering also I(X,X), I(Y,Y) and same I(Xi,Yj) = I(Yj,Xi)
mu_i_list = []
counter = 1
for i in range(0, 4):
    for j in range(0, 4):
        print('\033[1m' + "\n[ " + features[i] + " - " + features[j], "]" + '\033[0m')
        data_2d_matrix_tmp = np.append(data_discr[:, i], data_discr[:, j])
        data_2d_matrix = np.reshape(data_2d_matrix_tmp, (2, np.size(data_2d_matrix_tmp) // 2)).T  # 1D to 2D array
        data_2d_list.append(data_2d_matrix)  # Storage of all sets
        # Estimation of Joint p.m.f.
        array2d, pmf_join = myfunc3.pmf_multivariate(data_2d_matrix)  # Joint p.m.f.estimation
        # print("# Joint p.m.f. between", features[i], "feature and", features[j], "feature is: \n", pmf_join)

        # Plot of estimated Joint p.m.f. in 3D - some time is needed to plot
        # fig3 = plt.figure(5)
        # bx = fig3.add_subplot(4, 4, counter, projection='3d')  # for subplot - plots to small
        fig3 = plt.figure(counter+4)
        bx = plt.axes(projection='3d')  # for single plot
        bx.stem(array2d[:, 0], array2d[:, 1], pmf_join)
        # bx.scatter3D(array2d[:, 0], array2d[:, 1], pmf_join, c='red')
        bx.set_xlabel('x')
        bx.set_ylabel('y')
        bx.set_zlabel('Joint p.m.f.')
        plt.title("Multivariate probability mass function estimation \n"
                  "of " + features[i] + "-" + features[j])
        # Mutual Information
        mu_i = myfunc1.mutual_information(pmf_join, pmf_data_all[i][0], pmf_data_all[j][0])  # Mutual information
        # Storage of mu_i with string inside features
        mu_i_list.append(features[i] + "-" + features[j])  # add string info inside list
        mu_i_list.append(mu_i)  # add mu_i results inside list
        print("# Mutual Information between", features[i], "and", features[j], "is: ", mu_i, "[bit]")
        counter += 1

# Extra - continuous case of all pairs joint probabilities
# Some seconds to rendering are needed
# kde_est_list, x_plot_list, y_plot_list, bandwidth = myfunc3.plot_3d_joint_probs(data_matrix)

# Rendering of plot
plt.show()
