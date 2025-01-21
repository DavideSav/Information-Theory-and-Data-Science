#! /usr/bin/env python3
# Author: Davide Savini (0310795)

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KernelDensity
import assignment_2_ds.myfunc_assign2 as myfunc2
from sklearn.metrics import accuracy_score


# (1) Build a Bayes classifier function which takes a training dataset, a class
#     label vector for the training dataset and a test dataset and returns a
#     class label vector for the test dataset. Assume that the features are
#     continuous random variables and estimate the multivariate probability
#     density function by using a multivariate kernel density estimator.

# Kernel estimator function multivariate case
def kernel_estimator_multivariate(sub_data_train, test_data, bandwidth_input):
    # ############### Kernel density estimation ################
    # As the bandwidth increases, the estimated pdf goes from being too rough to too smooth
    bandwidth_kde = bandwidth_input
    # Valid kernel functions are: ‘gaussian’|’tophat’|’epanechnikov’|’exponential’|’linear’|’cosine’
    kernel_function = 'gaussian'
    # Fit the Kernel Density model on the data. KernelDensity returns the instance itself.
    kde_object = KernelDensity(kernel=kernel_function, bandwidth=bandwidth_kde).fit(sub_data_train)  # 25x4 each class
    # Compute the log-likelihood of each sample under the model.
    kde_log_density_estimate = kde_object.score_samples(test_data)  # 75x4 test dataset
    kde_estimate = np.exp(kde_log_density_estimate)
    return kde_estimate


# Function to subset division with respect number of classes
def subset_division_func(sub_data, classes):
    walk = np.size(sub_data[:, 0])//np.size(classes)  # walk is a scalar that represents length of each sub-dataset
    training_data_list = [sub_data[0:walk], sub_data[walk:2*walk], sub_data[2*walk:3*walk]]  # List of subsets
    return training_data_list


# function to create class_label_vector
def class_label_classification(arg_bayes_test_list):
    class_label_vector = np.array([])
    for i in range(0, np.size(arg_bayes_test_list[0])):
        # Setosa (0)
        if max(arg_bayes_test_list[0][i], arg_bayes_test_list[1][i], arg_bayes_test_list[2][i]) == arg_bayes_test_list[0][i]:
            class_label_vector = np.append(class_label_vector, 0)
        # Versicolour (1)
        elif max(arg_bayes_test_list[0][i], arg_bayes_test_list[1][i], arg_bayes_test_list[2][i]) == arg_bayes_test_list[1][i]:
            class_label_vector = np.append(class_label_vector, 1)
        # Virginica (2)
        elif max(arg_bayes_test_list[0][i], arg_bayes_test_list[1][i], arg_bayes_test_list[2][i]) == arg_bayes_test_list[2][i]:
            class_label_vector = np.append(class_label_vector, 2)
        else:
            print("The function class_label_classification must be improved!\n")  # Just a check message
    return class_label_vector


# Bayes classifier function
def bayes_classifier(training_data, class_lab, test_data, classes, bandwidth_input):
    # Therefore, in order to apply the Bayes classifier we need to estimate the prior probabilities
    # P(cj) and the likelihood P(xj|cj) using the training set Dtr.

    # ##################   Estimation of class probabilities   #################################
    samp_list_class, prob_class = myfunc2.pmf_univariate(class_lab)  # class is a discrete r.v.(Prior Probability)

    # ################# Subset division of training set used to FIT kde estimator
    training_data_list = subset_division_func(training_data, classes)

    # ########  Bayes steps  #####
    joint_test_prob_list = []
    prob_likelihood_test_list = []
    arg_bayes_test_list = []
    for i in range(0, len(training_data_list)):  # loop is used to extrapolate arrays from list
        sub_data = training_data_list[i]  # 25x4 inside training...list[0], 25x4 in tr...list[1] and tr...list[2]
        joint_test_prob = kernel_estimator_multivariate(sub_data, test_data, bandwidth_input)
        joint_test_prob_list.append(joint_test_prob)  # at the end of for, joint...list contains 3 arrays 75x1
        # ##### Bayes computation argument ######
        prob_likelihood_test_list.append(joint_test_prob_list[i]/prob_class[i])  # (Likelihood prob.)
        arg_prod_test_bayes = (prob_likelihood_test_list[i]*prob_class[i])  # bayes classifier
        arg_bayes_test_list.append(arg_prod_test_bayes)
    class_label_vector_test = class_label_classification(arg_bayes_test_list)  # 75x1 vector
    return class_label_vector_test, prob_class

# ################################################################################
# (2) Build a naive Bayes classifier function by assuming that the features
#     are continuous and independent random variables. Use a univariate
#     kernel density estimator to estimate the probability density function of
#     each feature.


# define function - Probability Density Function with Kernel estimator univariate
def kernel_univariate_naive(training_subdata, test_data, bandwdith_input):
    # # ##################   Kernel Method   ############################
    # As the bandwidth increases, the estimated pdf goes from being too rough to too smooth
    bandwidth_kde = bandwdith_input
    # bandwidth_kde = 0.1  # bad shape but estimation is acceptable in terms of diff. entropy
    # bandwidth_kde = 10  # the shape is totally smoothed
    # Valid kernel functions are: 'gaussian'|'tophat'|'epanechnikov'|’exponential'|'linear'|'cosine'
    kernel_function = 'gaussian'
    kde_estimate_list = []
    for i in range(0, len(training_subdata[0, :])):
        # Generation of kernel object using the training subset -> MODEL FIT
        # training_subdata.reshape(-1, 1) array 25x4 becames an array 100x1
        kde_object = KernelDensity(kernel=kernel_function, bandwidth=bandwidth_kde).fit(training_subdata[:, i].reshape(-1, 1))
        # TEST of the generated kde object
        x_plot = test_data[:, i].reshape(-1, 1)  # 75x4 to 300x1
        kde_log_density_estimate = kde_object.score_samples(x_plot)  # log results
        kde_estimate = np.exp(kde_log_density_estimate)  # single kde_estimate is the set of pdf features
        kde_estimate_list.append(kde_estimate)
    prod_pdfs = kde_estimate_list[0]*kde_estimate_list[1]*kde_estimate_list[2]*kde_estimate_list[3]  # of a single class
    return prod_pdfs


# Naive assumption: independence between features - in order to reduce the computational cost of computing
def naive_bayes_classifier(training_data, class_lab, test_data, classes, bandwidth_input):
    # ##################   Estimation of class probabilities   #################################
    samp_list_class, prob_class = myfunc2.pmf_univariate(class_lab)  # class is a discrete r.v.(Prior Probability)

    # ################# Subset division of training set used to FIT kde estimator
    training_data_list = subset_division_func(training_data, classes)

    # ############ Naive Bayes steps ##############
    prod_pdf_list = []
    for i in range(0, len(training_data_list)):
        sub_data = training_data_list[i]
        prod_pdf_of_feat = kernel_univariate_naive(sub_data, test_data, bandwidth_input)  # vector 300x1
        prod_pdf_list.append(prod_pdf_of_feat)  # three vectors inside list
    class_label_vector_test = class_label_classification(prod_pdf_list)
    return class_label_vector_test, prob_class

# ################################################################################
# (3) Build a naive Bayes classifier function by assuming that the features are
#     continuous, independent and Gaussian distributed random variables
#     with mean muj and variance sigmaj^2.


def gauss_estimation_univariate(sub_data, test_data):
    gauss_pdf_list = []
    for i in range(0, len(sub_data[0, :])):  # select of all features (col)
        mean_feat = np.mean(sub_data[:, i])  # mean of a single feature
        std_feat = np.std(sub_data[:, i])  # standard deviation of single feature
        gauss_feat_pdf = myfunc2.gaussian_pdf(test_data[:, i], mean_feat, std_feat)  # gaussian pdf
        gauss_pdf_list.append(gauss_feat_pdf)
    gauss_prod = gauss_pdf_list[0]*gauss_pdf_list[1]*gauss_pdf_list[2]*gauss_pdf_list[3]  # ind. gaussian prod result
    return gauss_prod


def gauss_naive_bayes_classifier(training_data, class_lab, test_data, classes):
    # ##################   Estimation of class probabilities   #################################
    samp_list_class, prob_class = myfunc2.pmf_univariate(class_lab)  # class is a discrete r.v.(Prior Probability)

    # ################# Subset division of training set used to FIT kde estimator
    training_data_list = subset_division_func(training_data, classes)

    # ############ Gauss Naive Bayes steps ##############
    gauss_prod_list = []
    for i in range(0, len(training_data_list)):
        sub_data = training_data_list[i]  # training subset
        gauss_prod_pdf = gauss_estimation_univariate(sub_data, test_data)  # result of product of gaussian pdfs
        gauss_prod_list.append(gauss_prod_pdf)  # argument of naive gaussian bayes classifier

    class_label_vector_test = class_label_classification(gauss_prod_list)  # classification
    return class_label_vector_test, prob_class


# Accuracy plot function respect to bandwidth
def accuracy_plot(class_label_true, training_data, class_vector, test_data, classes):
    walk = 0.01
    bandwidth = np.arange(0.01, 3.01, walk)
    accuracy_array_bayes = np.array([])
    accuracy_array_naive = np.array([])
    for i in range(0, len(bandwidth)):
        # Accuracy array for Bayes classifier respect to bandwidth
        class_label_vector_bayes, prob_class = \
            bayes_classifier(training_data, class_vector, test_data, classes, bandwidth[i])
        accuracy_bayes = accuracy_score(class_label_true, class_label_vector_bayes)
        accuracy_array_bayes = np.append(accuracy_array_bayes, accuracy_bayes)

        # Accuracy array for Naive Bayes classifier respect to bandwidth
        class_label_vector_naive, prob_class = \
            naive_bayes_classifier(training_data, class_vector, test_data, classes, bandwidth[i])
        accuracy_naive = accuracy_score(class_label_true, class_label_vector_naive)
        accuracy_array_naive = np.append(accuracy_array_naive, accuracy_naive)

    # Accuracy plot for Bayes classifier respect to bandwidth
    plt.figure(4)
    plt.plot(bandwidth, accuracy_array_bayes, "k")
    plt.title("Accuracy Bayes classifier respect to different bandwidth\n"
              "with bandwidth array walk=%f" % walk)
    plt.xlabel("Bandwidth (linear)")
    plt.ylabel("Accuracy")
    plt.grid(color='0.80')

    # Accuracy plot for Naive Bayes classifier respect to bandwidth
    plt.figure(5)
    plt.plot(bandwidth, accuracy_array_naive, "r")
    plt.title("Accuracy Naive Bayes classifier respect to different bandwidth\n"
              "with bandwidth array walk=%f" % walk)
    plt.xlabel("Bandwidth (linear)")
    plt.ylabel("Accuracy")
    plt.grid(color='0.80')

    # Bayes acc. and Naive Bayes acc. together
    plt.figure(6)
    plt.plot(bandwidth, accuracy_array_bayes, "k", label="Bayes Accuracy")
    plt.plot(bandwidth, accuracy_array_naive, "r--", label="Naive Accuracy")
    plt.title("Accuracy Classifiers respect to different bandwidth\n"
              "with bandwidth array walk=%f" % walk)
    plt.xlabel("Bandwidth (linear)")
    plt.ylabel("Accuracy")
    plt.legend(loc='lower right', prop={'size': 10})
    plt.grid(color='0.80')
    return bandwidth  # array of bandwidths
