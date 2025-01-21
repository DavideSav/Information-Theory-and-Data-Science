#! /usr/bin/env python3
# Author: Davide Savini (0310795)

import numpy as np
import matplotlib.pyplot as plt
import math
import assignment_1_ds.myfunctions as myfunc
import assignment_2_ds.myfunc_assign2 as myfunc2

# ################### Assignment #2 ###################
# General Recommendations
# The project should be carried out using Python. Good programming disci-
# pline should be followed when writing the Python code. This means that the
# variable names should be logical, the code must be commented and it should
# be written in such a way that it is easy to follow and understand.

# (1) Compute the difference between the entropy of a discrete random vari-
#     able given its p.m.f. vector (choose some p.m.f. vectors to test) and the
#     entropy of an estimated p.m.f. from a set of samples generated through
#     the same p.m.f. vector. You can choose freely one p.m.f. vector.

# (*) Note: consider the impact of bandwidth, kernel function and number of
#     samples on the difference between such entropies. Start from the example in
#     pmf_estimation.py and test_pmf_pdf_py

print("\n################### Assignment #2 ###################")
# First step -  entropy of a discrete random variable given its p.m.f. vector
samples_list = np.arange(1, 11, 1)  # list of samples
pmf = np.array([[0.12, 0.11, 0.01, 0.3, 0.02, 0.1, 0.14, 0.08, 0.07, 0.05]])  # Set of p.m.f. (1)
print("\nSet of used p.m.f. is: \n", pmf)
# samples_list = np.arange(1, 101, 1)
# pmf = np.random.random(100)
entr_of_disc = myfunc.discrete_entropy(pmf)  # computation of discrete entropy
print("\nEntropy of a discrete random variable given its p.m.f. vector is: \n", entr_of_disc, "[bit]")
# Plot of p.m.f.
fig1 = plt.figure(1)
plt.stem(samples_list, pmf[0], 'k')
plt.xlabel("Samples")
plt.ylabel("Set of p.m.f.", rotation=85)
plt.title("Probability mass function")
plt.grid(color='0.80')
# plt.legend(title='p.m.f.', loc='upper right', prop={'size': 8})

# Second step - entropy of an estimated p.m.f. from a set of samples generated through the same p.m.f. vector
# I took some information by test_pmf_pdf_py

# vector_length = 10
# vector_length = 100         # Acceptable estimation with vector_length = 100
vector_length = 1000        # The larger the vector length, the more accurate is the probability estimation
np.random.seed(1)           # Useful to generate same data over different runs

samp_for_est = np.random.choice(samples_list, p=pmf[0], size=vector_length)
samp_list, estimated_pmf = myfunc2.pmf_univariate(samp_for_est)  # estimation of p.m.f.
print("\nSet of estimated p.m.f. is: \n", estimated_pmf)
# Plot of estimated p.m.f.
fig2 = plt.figure(2)
plt.stem(samp_list, estimated_pmf)
plt.xlabel("Samples")
plt.ylabel("Est. p.m.f.", rotation=85)
plt.title("Estimated Probability mass function \n with a vector length = %i" % vector_length)
plt.grid(color='0.80')

# call function to compute the entropy of estimated p.m.f
entr_of_est_pmf = myfunc.discrete_entropy(estimated_pmf)  # discrete entropy of estimated p.m.f.
print("\nEntropy of estimated p.m.f. is: \n", entr_of_est_pmf, "[bit]")
difference = abs(entr_of_disc-entr_of_est_pmf)
print("\nThe difference between the two entropies in absolute value is: \n", difference, "[bit]")
print("\n--------------------------------------------------------------")

# (2) Write a function called "differential entropy" which computes the dif-
#     ferential entropy of a generic continuous random variable given its p.d.f.
# Differential Entropy h(X) is measured in nat while if we replace ln with log2 the unit of measure is bit.

# I choose a Rayleigh r.v. with the follow p.d.f.:
std1 = 1  # standard deviation (o scale parameter)
mean_ray = std1*math.sqrt(math.pi/2)  # mean formulation of Rayleigh distribution
x = np.arange(0, 10, 0.01)  # support set of positive real numbers
ray_pdf = myfunc2.rayleigh_pdf(x, std1)  # Rayleigh p.d.f. generation
# Plot of Rayleigh p.d.f.
fig3 = plt.figure(3)
plt.plot(x, ray_pdf, 'k', label="stand_dev = %i" % std1)
plt.xlabel("[x]")
plt.ylabel("[p.d.f.]", rotation=85)
plt.title("Rayleigh probability density function with standard deviation = %i" % std1)
plt.legend(loc='upper right', prop={'size': 10})
plt.grid(color='0.80')

# call function - Differential entropy
diff_entropy1, diff_entropy2 = myfunc2.differential_entropy(ray_pdf, x)  # differential entropy with two methods
diff_entropy3 = myfunc2.differ_ent_quad_ray(x)  # differential entropy with quad() method
print("\nThe differential entropy with Trapezoid method is: \n", diff_entropy1, "[bit]")  # Trapezoid
print("\nThe differential entropy with Simpson method is: \n", diff_entropy2, "[bit]")  # Simpson
print("\nThe differential entropy using Quad function is: \n", diff_entropy3, "[bit]")  # quad
print("\n--------------------------------------------------------------")

# (3) Compute the difference between the differential entropy of a Gaussian
#     continuous random variable given its p.d.f. vector and the differential
#     entropy of an estimated p.d.f. from a set of samples generated through
#     the same p.d.f. vector. You can choose freely the mean and variance.

mean = 0  # gaussian mean
standard_dev = 1  # gaussian standard deviation
x_vect = np.arange(-5, 5, 0.001)  # support set of all real numbers
gauss_pdf = myfunc2.gaussian_pdf(x_vect, mean, standard_dev)  # Gaussian p.d.f. generation
# Plot of Gaussian p.d.f.
fig4 = plt.figure(4)
plt.plot(x_vect, gauss_pdf, 'k', label='mean = %i' % mean + "\nstand_dev = %i" % standard_dev)
plt.xlabel("[x]")
plt.ylabel("[p.d.f.]", rotation=85)
plt.title("Gaussian p.d.f. with mean = %i" % mean + " and standard deviation = %i" % standard_dev)
plt.legend(loc='upper right', prop={'size': 10})
plt.grid(color='0.80')

# Differential entropy of Gaussian r.v.
diff_ent_gauss1, diff_ent_gauss2 = myfunc2.differential_entropy(gauss_pdf, x_vect)  # Trapezoid and Simpson
# diff_ent_gauss3 = myfunc2.differ_ent_quad_gau(x_vect)  # quad method
diff_ent_gauss3 = myfunc2.differ_ent_quad(x_vect, gauss_pdf)  # quad method
print("\nGaussian differential entropy with Trapezoid method is: \n", diff_ent_gauss1, "[bit]")  # Trapezoid
print("\nGaussian differential entropy with Simpson method is: \n", diff_ent_gauss2, "[bit]")  # Simpson
print("\nGaussian differential entropy with Quad function is: \n", diff_ent_gauss3, "[bit]")  # quad
print("\n--------------------------------------------------------------")

# Estimation of p.d.f. from x_vector using Kernel estimator explained in test_pmf_pdf.py
# ##################   Continuous r.v   ############################
vector_to_est = np.random.normal(loc=mean, scale=standard_dev, size=vector_length)
zero_vect = np.zeros(np.size(vector_to_est))  # array to plot samples
est_gauss_pdf, x_domain, est_mean, est_std, bandwidth = myfunc2.pdf_estimation_kernel(vector_to_est)
# Plot of estimated Gaussian p.d.f.
fig5 = plt.figure(5)
plt.plot(x_domain, est_gauss_pdf, 'r', label='mean = %i' % est_mean +
                                             "\nstand_dev = %i" % est_std + "\nbandwidth = %f" % bandwidth)
plt.scatter(vector_to_est, zero_vect, color='r', s=30, alpha=0.5, label="samples=%i" % vector_length)
plt.xlabel("[x]")
plt.ylabel("[Est. p.d.f.]", rotation=85)
plt.title("Estimated Gaussian p.d.f. (Kernel method) \n with mean = %i" % est_mean +
          " and standard deviation = %i" % est_std)
plt.legend(loc='upper right', prop={'size': 10})
plt.grid(color='0.80')

# Plot between Gaussian p.d.f. and estimated Gaussian p.d.f.
fig6 = plt.figure(6)
plt.plot(x_vect, gauss_pdf, 'k', label='mean = %i' % mean + "\nstand_dev = %i" % standard_dev)
plt.plot(x_domain, est_gauss_pdf, 'r--', label='mean = %i' % est_mean +
                                               "\nstand_dev = %i" % est_std + "\nbandwidth = %f" % bandwidth)
plt.scatter(vector_to_est, zero_vect, color='r', s=30, alpha=0.5, label="samples=%i" % vector_length)
plt.xlabel("[x]")
plt.ylabel("[p.d.f.]", rotation=85)
plt.title("Gaussian p.d.f. and Estimated Gaussian p.d.f. \n"
          "with mean = %i" % mean + " and standard deviation = %i" % standard_dev)
plt.legend(loc='upper right', prop={'size': 10})
plt.grid(color='0.80')

# Differential entropy of estimated Gaussian r.v.
diff_ent_estgauss1, diff_ent_estgauss2 = myfunc2.differential_entropy(est_gauss_pdf, x_domain)  # Trapz and Simpson
x_domain = x_domain[:, 0]  # x_domain is array of array, so I change in a single array
# diff_ent_estgauss3 = integrate.quad_vec(myfunc2.integrand, x_domain[0, 0], x_domain[len(x_domain)-1, 0], args=(est_gauss_pdf, ))[0]
# diff_ent_estgauss3 = np.sum(diff_ent_estgauss3 / (np.size(diff_ent_estgauss3)))  # Quad
diff_ent_estgauss3 = myfunc2.differ_ent_quad(x_domain, est_gauss_pdf)  # Quad
print("\nDifferential entropy of estimated Gaussian p.d.f. "
      "with Trapezoid method is: \n", diff_ent_estgauss1, "[bit]")  # Trapz
print("\nDifferential entropy of estimated Gaussian p.d.f. "
      "with Simpson method is: \n", diff_ent_estgauss2, "[bit]")  # Simpson
print("\nDifferential entropy of estimated Gaussian p.d.f. "
      "with Quad function is: \n", diff_ent_estgauss3, "[bit]")  # quad
print("\n--------------------------------------------------------------")

# Difference between the two differential entropies with respective methods
diff_1 = abs(diff_ent_gauss1 - diff_ent_estgauss1)
print("\n(Trapezoid method) The difference between the two diff. entropies in absolute value is: \n", diff_1, "[bit]")
diff_2 = abs(diff_ent_gauss2 - diff_ent_estgauss2)
print("\n(Simpson method) The difference between the two diff. entropies in absolute value is: \n", diff_2, "[bit]")
diff_3 = abs(diff_ent_gauss3 - diff_ent_estgauss3)
print("\n(Quad method) The difference between the two diff. entropies in absolute value is: \n", diff_3, "[bit]")

# rendering of all plots
plt.show()
