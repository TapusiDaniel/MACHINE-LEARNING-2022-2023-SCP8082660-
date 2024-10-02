# MACHINE LEARNING 2022-2023 (SCP8082660)

This GitHub repository houses materials and projects related to the Machine Learning course I attended at the University of Padova during the 2022/2023 academic year. It includes my homework assignments and lab work, demonstrating practical applications of machine learning concepts.

# Course Overview

The Machine Learning course aimed to provide students with a solid understanding of the basic principles and methodologies related to the learning problem, alongside hands-on experience through simulations and practical exercises. By the end of the course, students were expected to:

* Understand the fundamental principles and main methodologies of machine learning
* Solve both supervised and unsupervised learning problems
* Apply machine learning methodologies to various scenarios and select the best techniques based on the data and problem characteristics
* Adapt software tools for practical machine learning tasks, including regression and classification
* Gain insight into advanced topics like boosting, sparsity, and deep learning

# Examination and Assessment

The course evaluation was based on two components:
* **Written Exam**: A closed-book exam where students were required to solve problems demonstrating their understanding of key machine learning tools and their ability to interpret results in practical scenarios
* **Homeworks**: These homeworks, completed at home, focused on practical competence in applying machine learning concepts. The homeworks were documented with an explanation of the methods used and the results obtained.

# Lab Exercises
[Lab 1: Linear Models for Classification (Wine Dataset)](https://github.com/TapusiDaniel/MACHINE-LEARNING-2022-2023-SCP8082660-/blob/main/Lab1/Classification_wineData.ipynb)

  In this lab, we worked with the Wine dataset from the UCI Machine Learning Repository. The dataset consists of 178 instances of wine derived from three different cultivars in Italy. Each instance contains 13 features based on a chemical analysis of the wines. We applied linear classification models to classify the wines into their respective cultivars based on these features.

[Lab 2: Regression on House Pricing Dataset (Variable Selection & Regularization)](https://github.com/TapusiDaniel/MACHINE-LEARNING-2022-2023-SCP8082660-/blob/main/Lab2/lab2_regularization.ipynb)

  This lab focused on applying linear regression models to the House Pricing dataset, a reduced version of a dataset containing house sale prices for King County, Washington (which includes Seattle). The dataset includes 18 features such as the number of bedrooms, number of bathrooms, and the house’s square footage. We used variable selection and regularization techniques (e.g., Lasso and Ridge regression) to improve our predictive model for house prices.

[Lab 3: Support Vector Machines (SVM)](https://github.com/TapusiDaniel/MACHINE-LEARNING-2022-2023-SCP8082660-/blob/main/Lab3/SVM.ipynb)

  In this lab, we further explored linear models, specifically focusing on Support Vector Machines (SVMs). We applied SVMs to a classification task and compared different kernel methods to see how they impact model performance. The practical applications of SVMs in classification tasks were explored, with a focus on tuning hyperparameters to optimize model results.

[Lab 4: Neural Networks for Regression (House Pricing Dataset)](https://github.com/TapusiDaniel/MACHINE-LEARNING-2022-2023-SCP8082660-/blob/main/Lab4/Lab4_part2.ipynb)

  For the final lab, we revisited the House Pricing dataset and applied neural networks to predict house prices. This lab provided hands-on experience in designing and training a neural network for regression tasks, highlighting the differences between using neural networks for classification (as in Homework 3) versus regression.

# Homework Assignments

[Homework 1: Linear Regression on a Combined Cycle Power Plant (CCPP) Data](https://github.com/TapusiDaniel/MACHINE-LEARNING-2022-2023-SCP8082660-/blob/main/HW1/LR_DanielCristianMarianTapusi_2065492.ipynb)

  In this assignment, we implemented a linear regression model to predict the net hourly electrical energy output (PE) of a combined cycle power plant (CCPP) based on ambient conditions. The dataset consists of 5281 data points collected from 2006 to 2011, and includes the following features:
  * Temperature (AT): Hourly average ambient temperature
  * Ambient Pressure (AP): Hourly average ambient pressure
  * Relative Humidity (RH): Hourly average relative humidity
  * Exhaust Vacuum (V): Hourly average exhaust vacuum

  These variables affect the performance of the gas and steam turbines in the plant, and our goal was to predict the energy output based on these environmental variables.

[Homework 2: SVM for Classification (with and without Kernels)](https://github.com/TapusiDaniel/MACHINE-LEARNING-2022-2023-SCP8082660-/blob/main/HW2/SVM_DanielCristianMarianTapusi_2065492.ipynb)

  In the second assignment, we explored Support Vector Machines (SVMs) for image classification using the MNIST dataset, which consists of 70,000 images of handwritten digits (0-9). Each image is 28x28 pixels, and we used SVMs to classify the digits with and without the use of kernel functions. By transforming the data using kernel methods, we were able to compare the performance of linear and nonlinear SVMs on this classic image classification problem.

[Homework 3: Neural Networks for Classification](https://github.com/TapusiDaniel/MACHINE-LEARNING-2022-2023-SCP8082660-/blob/main/HW3/NN_DanielCristianMarianTapusi_2065492.ipynb)

  The third homework focused on neural networks for image classification using the Fashion MNIST dataset, which contains images of clothes and accessories. Similar to MNIST, the images are 28x28 pixels, and each label corresponds to a different type of clothing (e.g., T-shirt, trousers, etc.). We implemented a neural network and trained it to classify these images, exploring the impact of different hyperparameters and architectures on the model’s performance.
