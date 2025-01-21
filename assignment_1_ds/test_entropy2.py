#! /usr/bin/env python3
# Author: Davide Savini (0310795)

import numpy as np
import matplotlib.pyplot as plt
import assignment_1_ds.myfunctions as myfunctions

# (2) write a script called \test entropy2" which computes the entropy for a
#     generic binary random variable as a function of p0 and plots the entropy
#     function.

# The function binary_entropy is defined inside myfunctions.py
# call function
print("\n################### Assignment #1 - test_entropy2.py ###################")
prob_0 = np.arange(0.0001, 1, 0.0001)  # Set of single probabilities
# p0 = 0.2  # One single probability
binary_entropy_trend, q0 = myfunctions.binary_entropy(prob_0)  # binary entropy
print("\nThe set of Binary Entropies is: \n", binary_entropy_trend, "[bit]")  # Print of the binary entropy values
if isinstance(prob_0, np.ndarray):  # Plot of all binary entropies
    plt.plot(prob_0, binary_entropy_trend, 'k-', label='Entropy respect to p0', linewidth=2, markersize=9)
    plt.scatter(0.5, 1, s=80, facecolors='none', edgecolors='r', label='Max Binary Entropy Value with p0 = 0.5')
    plt.xlabel('Probability Mass Function [p0]')
    plt.ylabel('Discrete Entropy H(p0) [bit]')
    plt.grid(color='0.80')
    plt.legend(title='Entropies', loc='upper right', prop={'size': 8})
    plt.title('Trend of Binary Entropy computed respect to all possible p0')
    # rendering plot
    plt.show()
# Extra
elif isinstance(prob_0, float):  # Plot of a single binary entropy
    plt.plot(prob_0, binary_entropy_trend, 'bo-', label='Entropy rispect to p0', linewidth=2, markersize=3)
    plt.xlabel('Probability Mass Function [p0]')
    plt.ylabel('Discrete Entropy [H(p0)]')
    plt.grid(color='0.80')
    plt.legend(title='Entropies', loc='upper right', prop={'size': 8})
    plt.title('Entropy of generic binary random variable')
    # rendering plot
    plt.show()
