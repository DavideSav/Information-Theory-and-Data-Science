#! /usr/bin/env python3
# Author: Davide Savini (0310795)

import numpy as np
import math
from math import pi, sqrt
import scipy.integrate as integrate
from sklearn.neighbors import KernelDensity


# define function - Function suggested by pmf_estimation_univariate.py
def pmf_univariate(samples):
    n = len(samples)  # length of samples
    # np.unique returns the sorted unique elements of an array.
    # Return_counts = True, also return the number of times each unique item appears in ar.
    samples_list, pmf_vector = np.unique(samples, return_counts=True)
    return samples_list, pmf_vector/n  # list of samples and pmf vector normalized


# define function - I create a function to generate a Rayleigh p.d.f.
def rayleigh_pdf(sup_set_x, standard_deviation):
    x = sup_set_x  # set of positive real numbers
    std1 = standard_deviation  # standard deviation of Rayleigh r.v.
    var = math.pow(std1, 2)  # variance of Rayleigh r.v.
    if np.any(x):  # Domain of Rayleigh r.v.
        return (x/var)*np.exp(-(np.power(x, 2))/(2*var))  # Rayleigh p.d.f.
    else:
        return print("\nA p.d.f. can't be negative!\n")


# define function - Integrand with Rayleigh p.d.f. for differential entropy with quad()
def integrand_ray(x):
    std1 = 1  # standard deviation of Rayleigh r.v.
    var = math.pow(std1, 2)  # variance of Rayleigh r.v.
    pdf = (x / var) * np.exp(-(np.power(x, 2)) / (2 * var))  # Rayleigh p.d.f.
    return -pdf * np.log2(pdf)


# define function - Differential entropy with Trapezoid and Simpson method for integration
def differential_entropy(prob_density_func, support_set):
    pdf = prob_density_func[prob_density_func != 0]  # I delete zeros
    x = support_set[support_set != 0]  # I delete zeros
    # Method #1 - Trapezoid method that is defined in numpy. This method estimate assuming linear interp. between points
    diff_ent_1 = np.trapz(pdf*np.log2(1/pdf), x)
    # Method #2 - Simpson method, better than the prev. one, instead of linear interpolation uses quadratic interp.
    diff_ent_2 = integrate.simps(pdf*np.log2(1/pdf), x)
    return diff_ent_1, diff_ent_2


# define function - Differential entropy with quad() method - Rayleigh p.d.f.
def differ_ent_quad_ray(x):
    # Method #3 - Function quad() with Rayleigh p.d.f.
    # From SciPy documentation: Methods for Integrating Functions given function object.
    # The function quad is provided to integrate a function of one variable between two points. The points can be
    # (inf) to indicate infinite limits. (To write inf -> np.inf)
    a = x[0]  # first value of support set for quad() method
    b = x[len(x)-1]  # last value of support set for quad() method
    return integrate.quad(integrand_ray, a, b)[0]  # With [0] I don't need to print the abserror of integration


# define function - Function to generate a Gaussian p.d.f. of r.v. N(mean,std)
# Support set of real numbers, given mean and given standard_deviation (std)
def gaussian_pdf(support_set, mean, std):
    x = support_set
    return (1/(std*sqrt(2*pi)))*np.exp(-0.5*np.power((x-mean)/std, 2))  # Gauss p.d.f.


# define function - Probability Density Function with Kernel estimator
def pdf_estimation_kernel(vector_to_est):
    # ##################   Continuous r.v   ############################
    # continuous_samples = np.hstack((a, b, c))      # Multi-modal pdf if mean_1 is not equal to mean_2, etc.
    continuous_samples = vector_to_est
    # Computation of upper bound and lower bound of samples vector
    cont_samp_min = min(continuous_samples)  # upper bound
    cont_samp_max = max(continuous_samples)  # lower bound
    # Computation of statistical parameters
    cont_samp_std = np.std(continuous_samples)
    cont_samp_mean = np.mean(continuous_samples)
    # cont_samp_iqr = np.percentile(continuous_samples, 75, interpolation='midpoint') \
    #     - np.percentile(continuous_samples, 25, interpolation='midpoint')  # intequartile range
    margin = cont_samp_std * 2
    cont_samp_len = len(continuous_samples)
    # ##################   Kernel Method   ############################
    # Computation of optimal bandwidth, with vector_length = 1000, is around 0.26
    optimal_bandwidth = 1.06 * cont_samp_std * np.power(cont_samp_len, -1/5)
    # As the bandwidth increases, the estimated pdf goes from being too rough to too smooth
    bandwidth_kde = optimal_bandwidth
    # bandwidth_kde = 0.1  # bad shape but estimation is acceptable in terms of diff. entropy
    # bandwidth_kde = 2
    # bandwidth_kde = 10  # the shape is totally smoothed
    # Valid kernel functions are: 'gaussian'|'tophat'|'epanechnikov'|’exponential'|'linear'|'cosine'
    kernel_function = 'gaussian'
    # Generation of kernel object
    kde_object = KernelDensity(kernel=kernel_function, bandwidth=bandwidth_kde).fit(continuous_samples.reshape(-1, 1))
    # Domain X
    x_plot = np.linspace(cont_samp_min - margin, cont_samp_max + margin, 1000)[:, np.newaxis]
    kde_log_density_estimate = kde_object.score_samples(x_plot)  # log results
    kde_estimate = np.exp(kde_log_density_estimate)  # exponential to remapping the log results in normal results
    return kde_estimate, x_plot, cont_samp_mean, cont_samp_std, bandwidth_kde


# define function - Integrand to compute the integral with quad function
def integrand(x, pdf):
    return -pdf * np.log2(pdf)


# define function - Estimation of diff. entropy using quad integration
# Method #3 - Function quad() with generic p.d.f.
# quad_vec used instead of quad because I need to pass a np.array as args; quad accept only scalar in input of args!
def differ_ent_quad(x, pdf):
    # integ = integrate.quad_vec(integrand, x[0, 0], x[len(x)-1, 0], args=(pdf,))[0]
    integ = integrate.quad_vec(integrand, x[0], x[len(x)-1], args=(pdf,))[0]
    return np.sum(integ / (np.size(integ)))
