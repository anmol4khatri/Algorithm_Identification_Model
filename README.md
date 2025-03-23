# Algorithm Identification and Model Comparison Tool

This project helps you identify the best machine learning algorithm for your dataset by automatically applying multiple models and comparing their performance through accuracy scores, MSE, R-squared metrics, and visual comparisons. The tool generates detailed comparison graphs and metrics to help you make an informed decision about which model is most suitable for your specific dataset and problem.

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