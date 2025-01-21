#! /usr/bin/env python3
# Author: Davide Savini (0310795)

import numpy as np
import assignment_1_ds.myfunctions as myfunctions

# ################### Assignment #1 ###################
print("\n################### Assignment #1 ###################")
print("\n######### X and Y are discrete random variables #########")
# (1) First point
# call function - discrete entropy of a given set probabilities
prob_x = np.array([[0.381, 0.259, 0.36]])  # Set of probabilities (1)
entropy = myfunctions.discrete_entropy(prob_x)  # discrete entropy
print("\nRandom variable X with probability set: \n", prob_x)
print('\nThe Discrete Entropy is: \n', entropy, "[bit]")
print("\n--------------------------------------------------------------")

# (2) Second point
# ##################### See script called: test_entropy2.py #####################

# (3) Third point
# call function - Joint Entropy
couple_prob = np.array([[0.21, 0.19], [0.19, 0.05], [0.11, 0.25]])  # Set of joint probabilities (1)
jointentr = myfunctions.joint_entropy(couple_prob)  # joint entropy
print("\nRandom variables X and Y with joint probability set: \n", couple_prob)
print("\nJoint Entropy is: \n", jointentr, "[bit]")
print("\n--------------------------------------------------------------")

# (4) Fourth point
# call function
prob_y = np.array([[0.451, 0.549]])  # Set of probabilities of Y r.v. (1)
cond_entr, px_given_y = myfunctions.conditional_entropy(couple_prob, prob_y)  # conditional entropy
# print("\nThe set of conditional probabilities X given Y is: \n", px_given_y)
print("\nRandom variable Y with probability set: \n", prob_y,
      "\n\nRandom variables X and Y with joint probability set: \n", couple_prob)
print("\nThe conditional entropy of X given Y is: \n", cond_entr, "[bit]")
print("\n--------------------------------------------------------------")

# (5) Fifth point
# call function
mu_i = myfunctions.mutual_information(couple_prob, prob_x, prob_y)  # mutual information
print("\nRandom variable Y with probability set: \n", prob_y,
      "\n\nRandom variables X and Y with joint probability set: \n", couple_prob)
print("\nThe mutual information is: \n", mu_i, "[bit]")
print("\n--------------------------------------------------------------")

# (6) Sixth point
# call functions
# Normalized conditional entropy
# Bounded in the interval [0; 1].
entr_of_x = myfunctions.discrete_entropy(prob_x)
cond_entr, prob_x_given_y = myfunctions.conditional_entropy(couple_prob, prob_y)
eta_nce = myfunctions.normalized_cond_entropy(cond_entr, entr_of_x)
print("\nThe normalized conditional entropy is: \n(bounded in [0, 1])\n", eta_nce, "[bit]")
# Normalized joint entropy
# Bounded in the interval [1/2; 1].
entr_of_y = myfunctions.discrete_entropy(prob_y)
eta_nje = myfunctions.normalized_joint_entropy(mu_i, entr_of_x, entr_of_y)
print("\nThe normalized joint entropy is: \n(bounded in [1/2, 1])\n", eta_nje, "[bit]")
# Normalized mutual information
# Type 1 - Bounded in the interval [0; 1].
eta_nmui1 = myfunctions.norm_mutual_inf_type1(eta_nje)
print("\nThe normalized mutual information type 1 is: \n(bounded in [0, 1])\n", eta_nmui1, "[bit]")
# Type 2 - Bounded in the interval [1; 2].
eta_nmui2 = myfunctions.norm_mutual_inf_type2(eta_nmui1)
print("\nThe normalized mutual information type 2 is: \n(bounded in [1, 2])\n", eta_nmui2, "[bit]")
# Type 3 - Bounded in the interval [0; 1].
eta_nmui3 = myfunctions.norm_mutual_inf_type3(mu_i, entr_of_x, entr_of_y)
print("\nThe normalized mutual information type 3 is: \n(bounded in [0, 1])\n", eta_nmui3, "[bit]")
print("\n--------------------------------------------------------------")

# Extras - Chain rule Theorem
print("\n######### X and Y are discrete random variables #########")
jointentr_chain = cond_entr + entr_of_y

print("\nChain rule (Theorem): H(X,Y)=H(X|Y)+H(Y) = ", jointentr_chain,
      "\nThe discrete joint entropy computed before is: H(X,Y) = ", jointentr)

# Extra - Variation of Information V(X,Y) also called shared information distance
var_inf = jointentr - mu_i
print("\nVariation of Information: V(X,Y)=H(X|Y)+H(Y|X)=H(X,Y)-I(X,Y) = ", var_inf)
eta_vi = var_inf/jointentr
print("\nNormalized Variation of Information: eta_vi(X,Y)=V(X,Y)/H(X,Y) = ", eta_vi)
