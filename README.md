# Human Activity Recognition (HAR) Analysis Pipeline

This repository contains an end-to-end Machine Learning pipeline for Human Activity Recognition using wearable sensor data. Developed as part of the Data Science Topics course within the Master's in AI and Data Science.

The project implements a complete data science workflow, ranging from no library use algorithm implementation to rigorous model evaluation and hyperparameter tuning.

## Project Overview

The goal of this project is to classify human activities based on data acquired from 5 body-worn sensors using the FORTH-TRACE benchmark dataset.

Key Objectives:
* Processing raw IMU signals and handling outliers.
* Extracting robust temporal and spectral features.
* Implementing dimensionality reduction and feature selection.
* Training and evaluating supervised learning models (kNN and MLP).
* Analyzing Class Imbalance and Bias-Variance trade-offs.

## Dataset

* Source: [FORTH-TRACE Benchmark Dataset](https://github.com/spl-icsforth/FORTH_TRACE_DATASET).
* Participants: 15 subjects.
* Sensors: 5 IMU sensors placed on the wrists, chest, and legs.
* Activities: 16 distinct activities (e.g., Standing, Walking, Climbing Stairs, Talking).

## Methodology & Features

### Data Cleaning & Outlier Detection
* Univariate Analysis: Outlier detection using IQR and Z-Score (implemented from scratch).
* Multivariate Analysis: Outlier detection using K-Means Clustering (implemented from scratch using NumPy) and 3D visualization of clusters vs. noise.

### Feature Engineering
Extraction of time-domain and frequency-domain features based on the state-of-the-art literature, including:
* Time: Mean, Median, Variance, Skewness, Kurtosis, Zero Crossing Rate.
* Frequency: FFT-based metrics, Spectral Entropy, Welch’s Power Spectral Density (PSD).
* Physics-based: Signal Magnitude Area (SMA), Average Acceleration Energy (AAE).

### Feature Selection & Reduction
* PCA (Principal Component Analysis): Implemented for dimensionality reduction and variance explanation.
* Ranking Algorithms: Implementation of Fisher Score and ReliefF algorithms to select the most discriminative features.

### Classification & Model Evaluation
* Models: k-Nearest Neighbors (kNN) and Multi-Layer Perceptron (MLP).
* Validation: * Train-Test Split vs. 10x10 Fold Cross-Validation.
    * Nested Cross-Validation for hyperparameter tuning.
    * Bias-Variance and Overfitting/Underfitting analysis using Elbow Graphs.
* Class Imbalance: Implementation of random undersampling strategies to handle unbalanced activity classes.

## Tech Stack

* Language: Python
* Core Libraries:
    * `NumPy`
    * `Pandas`
    * `SciPy`
    * `Matplotlib`
    * `Scikit-learn`

## File Structure

* `mainActivity_A.py`: Core script for data loading, signal processing, outlier detection (Z-Score/K-Means), and feature extraction pipeline.
* `Part_b.py`: Machine Learning pipeline containing the training loops, cross-validation logic, kNN/MLP definitions, and class imbalance handling.
* `utils.py`: Helper functions for metrics and plotting.
* `figures_examples/`: Directory containing examples of generated plots and visual analysis.
* `data_example/`: .csv with samples of the dataset.

## 👥 Authors

* **Miguel**
* **Samuel**

---
*Developed for the Data Science Topics course.*
