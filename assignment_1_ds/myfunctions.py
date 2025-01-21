#! /usr/bin/env python3
# Author: Davide Savini (0310795)

import numpy as np

# ################### Assignment #1 ###################
# The project should be carried out using Python. Good programming dis-
# cipline should be applied. This means that the variable names should be
# logical, the code must be commented and it should be written in such a way
# that it is easy to follow and understand. Figures should have appropriate
# titles and axis labels. The software implementation should find a solution
# which minimizes the processing time over a set of equivalent methods.

# (1) Write a function called "entropy" which computes the entropy of a dis-
#     crete random variable given its probability mass function [p1; p2; :::; pN].


# define function - Discrete Entropy
# pk is the array of probability mass functions
def discrete_entropy(pk):
    # pk = pk[pk != 0]  # In this way I removed zeros that give me math errors
    return -np.sum(pk * np.log2(pk))


# (2) write a script called \test entropy2" which computes the entropy for a
#     generic binary random variable as a function of p0 and plots the entropy
#     function.


# define function - Binary Entropy
def binary_entropy(p0):
    psum = -p0*np.log2(p0)
    qsum = -(1-p0)*np.log2(1-p0)
    return (psum + qsum), (1-p0)

# (3) write a function called \joint entropy" which computes the joint en-
#     tropy of two generic discrete random variables given their joint p.m.f.


# define function - Joint Entropy
def joint_entropy(joint_prob):
    return -np.sum(joint_prob*np.log2(joint_prob))

# (4) write a function called \conditional entropy" which computes the con-
#     ditional entropy of two generic discrete random variables given their
#     joint and marginal p.m.f.


# define function - Conditional Entropy
def conditional_entropy(joint_prob, prob_y):
    # Probability of x given y
    p_x_given_y = joint_prob / prob_y
    return np.sum(joint_prob * (np.log2(1/p_x_given_y))), p_x_given_y

# (5) write a function called "mutual information" which computes the mu-
#     tual information of two generic discrete random variables given their
#     joint and marginal p.m.f.


# define function - Mutual Information
def mutual_information(joint_prob, prob_x, prob_y):
    # I have to manage with different sizes of probabilities
    # To automatize the problem, I can do a kind of zeropadding because the extra zeros don't change my final result
    # In this function, the computations are done in separated steps because I have to manage size of arrays!
    if np.size(prob_x) > np.size(prob_y):
        # Zeropadding on prob_y
        prob_y = np.append(prob_y, np.zeros(np.size(prob_x)-np.size(prob_y)))
        arg_prod = prob_x * prob_y
    elif np.size(prob_x) < np.size(prob_y):
        # Zeropadding on prob_x
        prob_x = np.append(prob_x, np.zeros(np.size(prob_y)-np.size(prob_x)))
        arg_prod = prob_x * prob_y
    else:
        # Zeropadding is not necessary
        arg_prod = prob_x * prob_y
    # I remove redondancy
    arg_prod = arg_prod[arg_prod != 0]
    # Now I have to manage the joint_prob size respect to prob_x*prob_y
    joint_prob = np.reshape(joint_prob, np.size(joint_prob))  # 2D array to 1D array
    if np.size(joint_prob) > np.size(arg_prod):
        # Zeropadding on arg_prod
        arg_prod = np.append(arg_prod, np.zeros(np.size(joint_prob)-np.size(arg_prod)))
    elif np.size(joint_prob) < np.size(arg_prod):
        # Zeropadding on joint_prob
        joint_prob = np.append(joint_prob, np.zeros(np.size(arg_prod)-np.size(joint_prob)))
    else:
        # Zeropadding is not necessary
        joint_prob = joint_prob
        arg_prod = arg_prod
    # Computation of sum argument
    arg_of_sum = joint_prob*np.log2(joint_prob/arg_prod)
    # I remove redondancy
    arg_of_sum = arg_of_sum[arg_of_sum != np.inf]
    return np.sum(arg_of_sum)


# (6) write the functions for normalized versions of conditional entropy, joint
#     entropy and mutual information for the discrete case.


# Normalized conditional entropy
# Bounded in the interval [0; 1].
def normalized_cond_entropy(cond_x_given_y, entr_of_x):
    return cond_x_given_y/entr_of_x


# Normalized joint entropy
# Bounded in the interval [1/2; 1].
def normalized_joint_entropy(mutualinf, entr_of_x, entr_of_y):
    return 1 - (mutualinf/(entr_of_x + entr_of_y))


# Normalized mutual information
# Type 1 - Bounded in the interval [0; 1].
def norm_mutual_inf_type1(norm_joint):
    return (1/norm_joint) - 1


# Type 2 - Bounded in the interval [1; 2].
def norm_mutual_inf_type2(norm_mu_inf_1):
    return 1 + norm_mu_inf_1


# Type 3 - Bounded in the interval [0; 1].
def norm_mutual_inf_type3(mutualinf, entr_of_x, entr_of_y):
    return mutualinf/np.sqrt(entr_of_x * entr_of_y)
