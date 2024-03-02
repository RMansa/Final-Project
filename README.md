# Final-Project

## Description
This repository contains code for three different machine learning projects:
1. Kannada MNIST Classification
2. Toxic Tweets NLP Analysis
3. Regression Problem

Each project focuses on a different aspect of machine learning, including classification, natural language processing (NLP), and regression analysis.

## Table of Contents
- [Project 1: Kannada MNIST - Classification Problem](#project-1-kannada-mnist---classification-problem)
- [Project 2: Toxic Tweets Dataset - NLP Problem](#project-2-toxic-tweets-dataset---nlp-problem)
- [Project 3: Regression Problem](#project-3-regression-problem)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [License](#license)

# Project 1: Kannada MNIST - Classification Problem
## Overview
This project aims to solve a 10-class classification problem using the Kannada MNIST dataset. Unlike the traditional MNIST dataset which contains Hindu numerals, this dataset consists of handwritten digits in the Kannada script. The dataset comprises 60,000 training images and 10,000 test images, each of size 28x28 pixels.

## Dataset
The Kannada MNIST dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/higgstachyon/kannada-mnist). More details about the dataset curation can be found in the paper titled ["Kannada-MNIST: A new handwritten digits dataset for the Kannada language"](https://arxiv.org/abs/1908.01242) by Vinay Uday Prabhu.

## Procedure
1. **Data Extraction**: Extract the dataset from the provided NPZ file or download it from the web.
2. **Dimensionality Reduction**: Perform Principal Component Analysis (PCA) to reduce the dimensionality of the images to 10 components.
3. **Model Application**: Apply the following machine learning models:
   - Decision Trees
   - Random Forest
   - Naive Bayes
   - K-NN Classifier
   - SVM
4. **Metrics Evaluation**:
   - Calculate Precision, Recall, F1-Score for each model.
   - Generate Confusion Matrix for each model.
   - Plot ROC-AUC curve for each model.
5. **Experimentation**:
   - Repeat the experiment for different component sizes: 15, 20, 25, 30.


# Project 2: Toxic Tweets Dataset - NLP Problem
## Overview
This project aims to analyze toxic tweets using natural language processing (NLP) techniques. The dataset contains labeled tweets, where toxic tweets are labeled as 1 and non-toxic tweets as 0. The objective is to build a model that can accurately classify tweets as toxic or non-toxic.

## Dataset
The Toxic Tweets dataset can be downloaded from [Kaggle](https://www.kaggle.com/datasets/ashwiniyer176/toxic-tweets-dataset). Credit goes to the original collectors of the dataset.

## Procedure
1. **Data Preparation**:
   - Convert the CSV file to a Pandas DataFrame.
2. **Text Representation**:
   - Convert the text to Bag of Words and TF-IDF representations.
3. **Model Application**:
   - Apply the following machine learning models:
     - Decision Trees
     - Random Forest
     - Naive Bayes Model
     - K-NN Classifier
     - SVM
4. **Metrics Evaluation**:
   - Calculate Precision, Recall, F1-Score for each model.
   - Generate Confusion Matrix for each model.
   - Plot ROC-AUC curve for each model.
  
  
# Project 3: Regression Problem
## Overview
This project focuses on predicting the current health of an organism based on measurements from two biological sensors measuring their biomarkers. The target variable represents the health status, where negative values indicate health lesser than the average case. Linear regression models are applied to the training data, and metrics such as Mean Squared Error (MSE) and Mean Absolute Error (MAE) are evaluated on the test split.

## Dataset
The dataset can be found in the shared folder:
- Training data: `p1-train.csv`
- Test data: `p1-test.csv`
The last column in both datasets represents the target variable, i.e., the current health of the organism.

## Procedure
1. **Data Preparation**:
   - Load the training and test datasets from the provided CSV files.
2. **Model Application**:
   - Apply the following regression models:
     - Linear Regression
     - Support Vector Regression (SVR)
3. **Metrics Evaluation**:
   - Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE) for each model.

## Dependencies
- Python 3.x
- Libraries:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib

# Conclusion

Collectively, the projects in this repository serve as a microcosm of the vast spectrum of challenges and opportunities in machine learning. From image classification and natural language understanding to predictive modeling in healthcare, each project encapsulates unique intricacies and methodologies inherent to its respective domain.

Moreover, the iterative process of data exploration, model development, and performance evaluation has equipped us with invaluable skills and insights that transcend the boundaries of individual projects. This process fosters a holistic understanding of machine learning principles and practices, enabling us to tackle diverse real-world problems with confidence and efficacy.
