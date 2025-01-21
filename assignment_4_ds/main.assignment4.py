#! /usr/bin/env python3
# Author: Davide Savini (0310795)

from sklearn.metrics import accuracy_score
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import assignment_4_ds.myfunc_assign4 as myfunc4

# Import of Iris dataset
iris = datasets.load_iris()
# data_matrix is array[x,y] (2D) with len 150 and size 600 (150x4) 150 instances (rows) and 4 features (col)
# class_vector is array[x] (1D) with len and size 150 (150x1) 150 instances (rows) linked with the prev. one.
data_matrix, class_vector = iris.data, iris.target

# Information about Iris dataset (test_pdf_multivariate_iris.py)
features = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
classes = ['Setosa', 'Versicolour', 'Virginica']
class_value = [0, 1, 2]  # 0 for Setosa, 1 for Versicolour, 2 for Virginica
# colors_of_classes = ListedColormap(['r', 'b', 'g'])
# select a triplet of features from the overall set of 4 features
first_feature = 0  # Sepal Length
second_feature = 1  # Sepal Width
third_feature = 2  # Petal Length
fourth_feature = 3  # Petal Width


# ################### Assignment #4 ###################
print("\n################### Assignment #4 ###################")
# (*) Hint: Start from the example in test pdf multivariate iris.py
# (1), (2) and (3) are in myfunc_assign4.py

# (4) Compute and compare the average accuracy of the previous classifiers
#     by applying it to the Iris dataset taking for each class label 50% of
#     the rows as training dataset and the remaining 50% of the rows as test
#     dataset.

# Matrix with 5th col (class col)
matrix_list = np.c_[data_matrix, class_vector]  # matrix with 5th col (class)
print("\nIris dataset with respective class (real one)\n", matrix_list)
print("\n--------------------------------------------------------------")

rows = np.size(data_matrix[:, 0])  # number of rows (150)
# col = np.size(data_matrix[0, :])  # number of columns

training_data = data_matrix[0:rows:2, :]  # matrix 75x4 built over "even" 75 rows (50%) of the original matrix
class_label_true_train = class_vector[0:rows:2]  # the true class vector of training dataset
test_data = data_matrix[1:rows:2, :]  # matrix 75x4 built over "odd" 75 rows (50%) of the original matrix
class_label_true = class_vector[1:rows:2]  # the true class vector of test dataset used to compute accuracy

# The build can be generalized to different sub-dataset of the original data-set, not only for the 50%.
# To change sub-dataset, the important thing is to choose a sub-dataset data_matrix[0:x, :] where x is a number
# multiple of 3 (number of classes) and 5 (4 features column + 1 class column)
# training_data = data_matrix[0:90, :]
# test_data = data_matrix[90:np.size(data_matrix[:, 0]), :]

print("\nTraining dataset (matrix 75x4 built over EVEN 75 rows of Iris dataset): \n", training_data)
print("\n--------------------------------------------------------------")

print("\nTest dataset (matrix 75x4 built over ODD 75 rows of Iris dataset): \n", test_data)
print("\n--------------------------------------------------------------")

# # kernel_estimator function has as output: three lists, one scalar and plot/subplots
# # KDE estimation and 3d plots  # to see plots need to uncomment inside function kernel
# kde_estimate, x_plot_list, y_plot_list, bandwdith = myfunc4.plot_3d_joint_probs(data_matrix)
# # Rendering of plots
# plt.show()

# ################### (1) ###################
bandwidth = 0.4  # bandwidth for KDE estimator
print("\n##### Bayes Classifier #####")
class_label_vector_test, prob_class = myfunc4.bayes_classifier(training_data, class_vector, test_data, classes, bandwidth)

print("\nThe p.m.f. of each class is: \n", prob_class,
      "\n\nClassification of Test Dataset gives a class label vector equal to: \n", class_label_vector_test)
# (4) Accuracy
print("\nThe accuracy of Bayes classifier respect the true associated features-class with bandwidth =", bandwidth,
      "is: \n", accuracy_score(class_label_true, class_label_vector_test))

# Distribution of test data after Bayes classification
# Sepal Length - Sepal Width - Petal Length
fig11 = plt.figure(11)
colors_of_classes = ListedColormap(['r', 'b', 'y'])
ax = plt.axes(projection='3d')
sc1 = ax.scatter(test_data[:, first_feature], test_data[:, second_feature], test_data[:, third_feature],
                 c=class_label_vector_test, cmap=colors_of_classes, edgecolor='k', s=40)
ax.set_xlabel(features[first_feature])  # Sepal Length
ax.set_ylabel(features[second_feature])  # Sepal Width
ax.set_zlabel(features[third_feature])  # Petal Length
plt.title("Distribution of test data after \nBayes classification" +
          " (bandwidth = %f" % bandwidth + ")")

# Sepal Length - Sepal Width - Petal Width
fig12 = plt.figure(12)
colors_of_classes = ListedColormap(['r', 'b', 'g'])
ax = plt.axes(projection='3d')
sc2 = ax.scatter(test_data[:, first_feature], test_data[:, second_feature], test_data[:, fourth_feature],
                 c=class_label_vector_test, cmap=colors_of_classes, edgecolor='k', s=40)
ax.set_xlabel(features[first_feature])  # Sepal Length
ax.set_ylabel(features[second_feature])  # Sepal Width
ax.set_zlabel(features[fourth_feature])  # Petal Width
plt.title("Distribution of test data after \nBayes classification" +
          " (bandwidth = %f" % bandwidth + ")")


# ################### (2) ###################
print("\n##### Naive Bayes Classifier #####")
class_label_vector_test, prob_class = myfunc4.naive_bayes_classifier(training_data, class_vector, test_data, classes, bandwidth)
print("\nThe p.m.f. of each class is: \n", prob_class,
      "\n\nClassification of Test Dataset gives a class label vector equal to: \n", class_label_vector_test)
# (4) Accuracy
print("\nThe accuracy of Naive Bayes classifier respect the true associated features-class with bandwidth =", bandwidth,
      "is: \n", accuracy_score(class_label_true, class_label_vector_test))

# Distribution of test data after Naive Bayes classification
# Sepal Length - Sepal Width - Petal Length
fig13 = plt.figure(13)
colors_of_classes = ListedColormap(['r', 'b', 'y'])
ax = plt.axes(projection='3d')
sc3 = ax.scatter(test_data[:, first_feature], test_data[:, second_feature], test_data[:, third_feature],
                 c=class_label_vector_test, cmap=colors_of_classes, edgecolor='k', s=40)
ax.set_xlabel(features[first_feature])  # Sepal Length
ax.set_ylabel(features[second_feature])  # Sepal Width
ax.set_zlabel(features[third_feature])  # Petal Length
plt.title("Distribution of test data after \nNaive Bayes classification" +
          " (bandwidth = %f" % bandwidth + ")")

# Sepal Length - Sepal Width - Petal Width
fig14 = plt.figure(14)
colors_of_classes = ListedColormap(['r', 'b', 'g'])
ax = plt.axes(projection='3d')
sc4 = ax.scatter(test_data[:, first_feature], test_data[:, second_feature], test_data[:, fourth_feature],
                 c=class_label_vector_test, cmap=colors_of_classes, edgecolor='k', s=40)
ax.set_xlabel(features[first_feature])  # Sepal Length
ax.set_ylabel(features[second_feature])  # Sepal Width
ax.set_zlabel(features[fourth_feature])  # Petal Width
plt.title("Distribution of test data after \nNaive Bayes classification" +
          " (bandwidth = %f" % bandwidth + ")")

# ################### (3) ###################
print("\n##### Gaussian Naive Bayes Classifier #####")
class_label_vector_test, prob_class = myfunc4.gauss_naive_bayes_classifier(training_data, class_vector, test_data, classes)

print("\nThe p.m.f. of each class is: \n", prob_class,
      "\n\nClassification of Test Dataset gives a class label vector equal to: \n", class_label_vector_test)
# (4) Accuracy
print("\nThe accuracy of Gaussian Naive Bayes classifier respect the true associated features-class is: \n",
      accuracy_score(class_label_true, class_label_vector_test))

# Distribution of test data after Gaussian Naive Bayes classification
# Sepal Length - Sepal Width - Petal Length
fig15 = plt.figure(15)
colors_of_classes = ListedColormap(['r', 'b', 'y'])
ax = plt.axes(projection='3d')
sc5 = ax.scatter(test_data[:, first_feature], test_data[:, second_feature], test_data[:, third_feature],
                 c=class_label_vector_test, cmap=colors_of_classes, edgecolor='k', s=40)
ax.set_xlabel(features[first_feature])  # Sepal Length
ax.set_ylabel(features[second_feature])  # Sepal Width
ax.set_zlabel(features[third_feature])  # Petal Length
plt.title("Distribution of test data after \nGaussian Naive Bayes classification")

# Sepal Length - Sepal Width - Petal Width
fig16 = plt.figure(16)
colors_of_classes = ListedColormap(['r', 'b', 'g'])
ax = plt.axes(projection='3d')
sc6 = ax.scatter(test_data[:, first_feature], test_data[:, second_feature], test_data[:, fourth_feature],
                 c=class_label_vector_test, cmap=colors_of_classes, edgecolor='k', s=40)
ax.set_xlabel(features[first_feature])  # Sepal Length
ax.set_ylabel(features[second_feature])  # Sepal Width
ax.set_zlabel(features[fourth_feature])  # Petal Width
plt.title("Distribution of test data after \nGaussian Naive Bayes classification")

# Rendering plots
print("\n..rendering of scatter-plot classifications and plot accuracy respect to different bandwidth..")
bandwidth_array = myfunc4.accuracy_plot(class_label_true, training_data, class_vector, test_data, classes)
plt.show()
print("done")
