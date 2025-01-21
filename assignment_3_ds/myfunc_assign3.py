#! /usr/bin/env python3
# Author: Davide Savini (0310795)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.neighbors import KernelDensity


# data_matrix has 'rows' row vector samples and 'columns' features
def pmf_multivariate(data_matrix):
    rows, columns = data_matrix.shape  # returns the number of rows and columns of the data_matrix
    # the parameter axis=0 allows to count the unique rows
    unique_rows_array, pmf_vector = np.unique(data_matrix, axis=0, return_counts=True)
    # To obtain the probability, the count must be normalized to the total count of samples
    return unique_rows_array, pmf_vector/rows


# Function to plot joint probability in 3D - continuous case
def plot_3d_joint_probs(data_matrix):
    # Information about Iris dataset (test_pdf_multivariate_iris.py)
    features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    first_feature = 0  # respect of data set of 2 col taken from original data set
    second_feature = 1
    # List of n features-dimensional data points. Each row corresponds to a single data point
    # We have selected two columns from the data matrix
    # We may also select rows belonging to the same class
    # The following process take count of all possible cases of joint probability
    bandwidth = 0.4
    kde_est_list = []
    x_plot_list = []
    y_plot_list = []
    counter = 1
    for i in range(0, len(data_matrix[0, :])):
        for j in range(0, len(data_matrix[0, :])):
            data_samples = np.transpose(np.vstack((data_matrix[:, i], data_matrix[:, j])))  # All classes for all combination
            feature_1_min, feature_1_max, feature_1_std, feature_1_mean = data_samples[:, 0].min(), data_samples[:, 0].max(), \
                                                                          data_samples[:, 0].std(), data_samples[:, 0].mean()
            feature_2_min, feature_2_max, feature_2_std, feature_2_mean = data_samples[:, 1].min(), data_samples[:, 1].max(), \
                                                                          data_samples[:, 1].std(), data_samples[:, 1].mean()

            # #######################################################################################à
            n_samples = 10
            start_sample_1 = feature_1_mean - 2 * feature_1_std
            start_sample_2 = feature_2_mean - 2 * feature_2_std
            stop_sample_1 = feature_1_mean + 2 * feature_1_std
            stop_sample_2 = feature_2_mean + 2 * feature_2_std

            x_plot = np.linspace(start_sample_1, stop_sample_1, n_samples, endpoint=True)  # row vector
            y_plot = np.linspace(start_sample_2, stop_sample_2, n_samples, endpoint=True)  # row vector

            data_plot_x, data_plot_y = np.meshgrid(x_plot, y_plot)
            data_plot_x_vectorized = data_plot_x.flatten()  # Vectorize the grid matrix data_plot_x
            data_plot_y_vectorized = data_plot_y.flatten()  # Vectorize the grid matrix data_plot_y
            data_plot = np.transpose(np.vstack((data_plot_x_vectorized, data_plot_y_vectorized)))

            # ############### Kernel density estimation ################
            # As the bandwidth increases, the estimated pdf goes from being too rough to too smooth
            bandwidth_kde = bandwidth
            # Valid kernel functions are: ‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’
            kernel_function = 'gaussian'
            kde_object = KernelDensity(kernel=kernel_function, bandwidth=bandwidth_kde).fit(data_samples)

            kde_log_density_estimate = kde_object.score_samples(data_plot)
            kde_estimate = np.exp(kde_log_density_estimate)

            # ####### 3D plot of KDE p.d.f. ##############################
            # For single plot -> 16 figures (1 figure per each plot)
            fig = plt.figure()  # for 1 figure per each plot
            ax = plt.axes(projection='3d')  # for 1 figure per each plot
            trisurf = ax.plot_trisurf(data_plot[:, first_feature], data_plot[:, second_feature], kde_estimate,
                                      cmap=cm.coolwarm, linewidth=0, antialiased=False)
            ax.set_xlabel(features[i])
            ax.set_ylabel(features[j])
            ax.set_zlabel("p.d.f.")
            plt.title("Multivariate probability density function estimation \n(kernel method) "
                      "of " + features[i] + "-" + features[j])
            # A StrMethodFormatter is used automatically
            ax.zaxis.set_major_formatter('{x:.02f}')  # color on z-axis
            # Add a color bar which maps values to colors.
            fig.colorbar(trisurf, shrink=0.5, aspect=5)  # bar color on the right of plot
            ##########################################################

            # ###### For subplot -> 16 plot in a figure ##############
            fig = plt.figure(18)  # for subplot
            ax = fig.add_subplot(4, 4, counter, projection='3d')  # for subplot
            trisurf = ax.plot_trisurf(data_plot[:, first_feature], data_plot[:, second_feature], kde_estimate,
                                      cmap=cm.coolwarm, linewidth=0, antialiased=False)
            # A StrMethodFormatter is used automatically
            ax.zaxis.set_major_formatter('{x:.02f}')  # color on z-axis
            # Add a color bar which maps values to colors.
            fig.colorbar(trisurf, shrink=0.5, aspect=5)  # bar color on the right of plot

            ax.title.set_text(features[i][0] + features[i][6] + "-" + features[j][0] + features[j][6])
            # ax.set_xlabel(features[i][0] + features[i][6])
            # ax.set_ylabel(features[j][0] + features[j][6])
            ax.xaxis.set_ticklabels([])  # I hide numbers on x-axis
            ax.yaxis.set_ticklabels([])  # I hide numbers on y-axis
            ax.zaxis.set_ticklabels([])  # I hide numbers on z-axis
            # ############################################################

            # Storage of arrays that I need
            kde_est_list.append(kde_estimate)
            x_plot_list.append(x_plot)
            y_plot_list.append(y_plot)
            counter += 1  # used for subplots
    return kde_est_list, x_plot_list, y_plot_list, bandwidth
