# ML Model Selection and Evaluation Tool

This project provides a comprehensive tool for analyzing datasets and automatically selecting and evaluating the most appropriate machine learning models for both classification and regression tasks.

## Features

- Automatic dataset analysis (mean, median, mode, standard deviation)
- Automatic task type detection (classification or regression)
- Support for multiple ML algorithms:
  - Classification:
    - Decision Tree
    - K-Nearest Neighbors (KNN)
    - Logistic Regression
    - Naive Bayes
    - Random Forest
    - Support Vector Machine (SVM)
  - Regression:
    - Linear Regression
    - Ridge Regression
    - Lasso Regression
    - K-Nearest Neighbors (KNN)
    - Random Forest
    - Support Vector Machine (SVM)
- Automatic model evaluation with appropriate metrics
- Feature scaling and preprocessing

## Installation

1. Clone this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset in CSV format
2. Run the main script:
```bash
python main.py
```
3. Follow the prompts to:
   - Provide the path to your dataset
   - Specify the target column name
4. The script will:
   - Analyze your dataset
   - Determine if it's a classification or regression task
   - Train and evaluate multiple models
   - Display the results with appropriate metrics

## Output

For classification tasks, the script will show accuracy scores for each model.
For regression tasks, the script will show Mean Squared Error (MSE) and R² scores for each model.

## Requirements

- Python 3.7+
- NumPy
- Pandas
- scikit-learn 