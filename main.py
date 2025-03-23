import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, classification_report, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Import all our model implementations
from models.decision_tree import DecisionTreeClassifier, DecisionTreeRegressor
from models.knn import KNNClassifier, KNNRegressor
from models.lasso import LassoRegression
from models.linear_regression import CustomLinearRegression
from models.logistic_regression import LogisticRegression
from models.naive_bayes import NaiveBayes
from models.random_forest import RandomForestClassifier, CustomRandomForestRegressor
from models.ridge import RidgeRegression
from models.svm import SVMClassifier, SVMRegressor

def analyze_dataset(data):
    """Analyze basic statistics and column types of the dataset"""
    print("\nColumn Analysis:")
    print("-" * 50)
    
    for column in data.columns:
        print(f"\nColumn: {column}")
        print(f"Data type: {data[column].dtype}")
        print(f"Unique values: {len(data[column].unique())}")
        print(f"Sample values: {data[column].dropna().head(3).tolist()}")
        
        # Determine appropriate task type for this column
        task = determine_task_type(data[column])
        print(f"Suggested task type: {task}")
        
        # Show basic stats for numeric columns
        if pd.api.types.is_numeric_dtype(data[column]):
            print(f"Mean: {data[column].mean():.2f}")
            print(f"Std: {data[column].std():.2f}")
        else:
            print(f"Most common value: {data[column].mode().iloc[0]}")
        
        print(f"Missing values: {data[column].isnull().sum()}")
        print("-" * 30)

def determine_task_type(y, original_dtype=None):
    """Determine if the task is classification or regression"""
    # If we have original dtype information, use it first
    if original_dtype is not None and (original_dtype == 'object' or pd.api.types.is_categorical_dtype(original_dtype)):
        return 'classification'
    
    # For numeric columns, check the nature of values
    if pd.api.types.is_numeric_dtype(y):
        unique_ratio = len(np.unique(y)) / len(y)
        
        # If unique values are less than 5% of total values or <= 20 unique values,
        # likely classification
        if unique_ratio < 0.05 or len(np.unique(y)) <= 20:
            return 'classification'
        
        # If values are mostly integers and few unique values, likely classification
        if np.all(y.dropna() == y.dropna().astype(int)) and len(np.unique(y)) <= 50:
            return 'classification'
    
    return 'regression'

def evaluate_classification_models(X_train, X_test, y_train, y_test):
    """Evaluate all classification models"""
    models = {
        'Decision Tree': DecisionTreeClassifier(),
        'KNN': KNNClassifier(),
        'Logistic Regression': LogisticRegression(),
        'Naive Bayes': NaiveBayes(),
        'Random Forest': RandomForestClassifier(),
        'SVM': SVMClassifier()
    }
    
    results = {}
    for name, model in models.items():
        try:
            print(f"\nTraining {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            # For multi-class problems, use weighted average
            f1 = f1_score(y_test, y_pred, average='weighted')
            results[name] = {
                'Accuracy': accuracy,
                'F1': f1
            }
            # Print results immediately for each model
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 Score: {f1:.4f}")
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            results[name] = None
    
    return results

def evaluate_regression_models(X_train, X_test, y_train, y_test):
    """Evaluate all regression models"""
    models = {
        'Linear Regression': CustomLinearRegression(),
        'Ridge Regression': RidgeRegression(),
        'Lasso Regression': LassoRegression(),
        'KNN': KNNRegressor(),
        'Random Forest': CustomRandomForestRegressor(),
        'SVM': SVMRegressor()
    }
    
    results = {}
    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            results[name] = {'MSE': mse, 'R2': r2}
        except Exception as e:
            print(f"Error with {name}: {str(e)}")
            results[name] = None
    
    return results

def convert_time_to_minutes(time_str):
    """Convert time string to normalized time value between 0 and 1"""
    try:
        if pd.isna(time_str):
            return np.nan
        # Split the time string into hours and minutes
        hours, minutes = map(int, time_str.split(':'))
        # Convert to minutes since midnight
        total_minutes = hours * 60 + minutes
        # Normalize to 0-1 range (divide by total minutes in a day)
        return total_minutes / 1440.0  # 1440 = 24*60
    except:
        return np.nan

def preprocess_data(data):
    """Preprocess the dataset and show information before and after"""
    # Display initial dataset information
    print("\nInitial Dataset Information:")
    print("-" * 50)
    print(f"Initial number of rows: {data.shape[0]}")
    print(f"Initial number of columns: {data.shape[1]}")
    print("\nInitial columns:")
    print(data.columns.tolist())
    print("\nInitial Dataset Description:")
    print(data.describe())
    
    initial_columns = set(data.columns)
    initial_rows = data.shape[0]
    
    # Convert Time column to normalized minutes if it exists
    if 'Time' in data.columns:
        print("\nConverting Time column to normalized values (0-1)...")
        data['Time'] = data['Time'].apply(convert_time_to_minutes)
        print(f"Time column range: {data['Time'].min():.3f} to {data['Time'].max():.3f}")
    
    # Remove columns with more than 50% missing values
    missing_percentages = data.isnull().sum() / len(data) * 100
    columns_to_drop = missing_percentages[missing_percentages > 50].index
    if len(columns_to_drop) > 0:
        print(f"\nDropping columns with >50% missing values:")
        print(columns_to_drop.tolist())
        data = data.drop(columns=columns_to_drop)
    
    # Convert date to numeric features
    if 'Date' in data.columns:
        try:
            data['Date'] = data['Date'].astype(str).str.extract('(\d{4})').astype(float)
            data['Year'] = data['Date']
            data = data.drop('Date', axis=1)
        except Exception as e:
            print(f"Warning: Could not process date column: {str(e)}")
            if 'Date' in data.columns:
                data = data.drop('Date', axis=1)
    
    # Handle missing values in rows
    rows_before = len(data)
    data = data.dropna(thresh=len(data.columns) - int(0.3 * len(data.columns)))  # Drop rows missing >30% of values
    rows_removed = rows_before - len(data)
    if rows_removed > 0:
        print(f"\nRemoved {rows_removed} rows with >30% missing values")
    
    # Handle remaining missing values in numeric columns
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        missing_before = data[col].isna().sum()
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].fillna(data[col].mean())
        missing_after = data[col].isna().sum()
        if missing_before > 0:
            print(f"\nColumn '{col}': Filled {missing_before} missing values")
    
    # Store original dtype for target column determination later
    original_dtypes = data.dtypes.copy()
    
    # Convert categorical variables to numeric
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        data[col] = pd.factorize(data[col])[0]
    
    # Add the original dtypes as an attribute to the dataframe
    data.original_dtypes = original_dtypes
    
    # Display final dataset information
    print("\nFinal Dataset Information:")
    print("-" * 50)
    print(f"Final number of rows: {data.shape[0]} (Removed: {initial_rows - data.shape[0]} rows)")
    print(f"Final number of columns: {data.shape[1]} (Removed: {len(initial_columns) - data.shape[1]} columns)")
    print("\nFinal columns:")
    print(data.columns.tolist())
    print("\nFinal Dataset Description:")
    print(data.describe())
    
    # Show removed columns
    removed_columns = initial_columns - set(data.columns)
    if removed_columns:
        print("\nRemoved columns:")
        print(list(removed_columns))
    
    return data

def plot_classification_results(results):
    """Create various visualization plots for classification results in a single window"""
    # Filter out None values
    results = {k: v for k, v in results.items() if v is not None}
    
    models = list(results.keys())
    accuracy_values = [results[model]['Accuracy'] for model in models]
    f1_values = [results[model]['F1'] for model in models]
    
    # Create a single figure with better spacing
    fig = plt.figure(figsize=(24, 36))  # Adjusted figure size ratio
    plt.suptitle('Model Analysis Dashboard', fontsize=16, y=0.98)  # Moved title up
    
    # Create grid with proper spacing
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)  # Added spacing parameters
    
    # Screen 1: Basic Comparisons
    ax1 = fig.add_subplot(gs[0, 0])  # Instead of plt.subplot(3, 4, 1)
    x = np.arange(len(models))
    width = 0.35
    ax1.bar(x - width/2, accuracy_values, width, label='Accuracy', color='skyblue')
    ax1.bar(x + width/2, f1_values, width, label='F1 Score', color='lightgreen')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Score')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')  # Improved label alignment
    ax1.legend()
    
    plt.subplot(gs[0, 1])
    plt.scatter(accuracy_values, f1_values, s=100)
    for i, model in enumerate(models):
        plt.annotate(model, (accuracy_values[i], f1_values[i]), xytext=(5, 5), textcoords='offset points')
    plt.title('Accuracy vs F1 Score')
    plt.xlabel('Accuracy')
    plt.ylabel('F1 Score')
    
    plt.subplot(gs[0, 2])
    plt.plot(models, accuracy_values, marker='o', label='Accuracy', linewidth=2)
    plt.plot(models, f1_values, marker='s', label='F1 Score', linewidth=2)
    plt.title('Performance Metrics Trend')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    
    ax = fig.add_subplot(gs[0, 3], projection='polar')
    angles = np.linspace(0, 2*np.pi, len(models), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    accuracy_values_np = np.array(accuracy_values)
    f1_values_np = np.array(f1_values)
    accuracy_values_plot = np.concatenate((accuracy_values_np, [accuracy_values_np[0]]))
    f1_values_plot = np.concatenate((f1_values_np, [f1_values_np[0]]))
    ax.plot(angles, accuracy_values_plot, 'o-', label='Accuracy')
    ax.plot(angles, f1_values_plot, 'o-', label='F1 Score')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(models)
    plt.legend()
    plt.title('Radar Plot')
    
    # Screen 2: Advanced Analysis (subplots 5-8)
    plt.subplot(gs[1, 0])
    plt.violinplot([accuracy_values, f1_values])
    plt.title('Distribution of Metrics')
    plt.ylabel('Score')
    plt.xticks([1, 2], ['Accuracy', 'F1 Score'])
    
    plt.subplot(gs[1, 1])
    plt.boxplot([accuracy_values, f1_values], labels=['Accuracy', 'F1 Score'])
    plt.title('Score Distributions')
    plt.ylabel('Score')
    
    plt.subplot(gs[1, 2])
    metrics_data = np.array([accuracy_values, f1_values])
    sns.heatmap(metrics_data, annot=True, fmt='.3f', 
                xticklabels=models, yticklabels=['Accuracy', 'F1'],
                cmap='YlOrRd')
    plt.title('Metrics Heatmap')
    
    plt.subplot(gs[1, 3])
    for i in range(len(models)):
        plt.plot([0, 1], [accuracy_values[i], f1_values[i]], '-o', label=models[i])
    plt.xticks([0, 1], ['Accuracy', 'F1'])
    plt.title('Parallel Coordinates Plot')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Screen 3: Comparative Analysis (subplots 9-12)
    plt.subplot(gs[2, 0])
    x = range(len(models))
    plt.bar(x, accuracy_values, label='Accuracy')
    plt.bar(x, f1_values, bottom=accuracy_values, label='F1 Score')
    plt.xticks(x, models, rotation=45)
    plt.legend()
    plt.title('Stacked Metrics')
    
    plt.subplot(gs[2, 1])
    plt.fill_between(range(len(models)), accuracy_values, alpha=0.5, label='Accuracy')
    plt.fill_between(range(len(models)), f1_values, alpha=0.5, label='F1 Score')
    plt.xticks(range(len(models)), models, rotation=45)
    plt.legend()
    plt.title('Area Plot of Metrics')
    
    ax = fig.add_subplot(gs[2, 2], projection='polar')
    ax.plot(angles, accuracy_values_plot, 'o-', label='Accuracy')
    ax.plot(angles, f1_values_plot, 'o-', label='F1 Score')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(models)
    plt.legend()
    plt.title('Radar Plot of Both Metrics')
    
    plt.subplot(gs[2, 3])
    plt.scatter(accuracy_values, f1_values, s=1000, alpha=0.5)
    for i, model in enumerate(models):
        plt.annotate(model, (accuracy_values[i], f1_values[i]))
    plt.xlabel('Accuracy Score')
    plt.ylabel('F1 Score')
    plt.title('Bubble Plot (Accuracy vs F1)')
    
    plt.subplots_adjust(top=0.95)  # Adjust top margin for main title
    plt.show()

def plot_regression_results(results):
    """Create various visualization plots for regression results in a single window"""
    # Filter out None values
    results = {k: v for k, v in results.items() if v is not None}
    
    models = list(results.keys())
    mse_values = [results[model]['MSE'] for model in models]
    r2_values = [results[model]['R2'] for model in models]
    
    # Create a single figure with better spacing
    fig = plt.figure(figsize=(24, 36))
    plt.suptitle('Regression Model Analysis Dashboard', fontsize=16, y=0.98)
    
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)
    
    # Screen 1: Basic Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.bar(models, mse_values)
    ax1.set_title('Model MSE Comparison')
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Mean Squared Error')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 2. Bar Plot for R2
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(models, r2_values)
    ax2.set_title('Model R² Score Comparison')
    ax2.set_xlabel('Models')
    ax2.set_ylabel('R² Score')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 3. Dual Metric Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    x = np.arange(len(models))
    width = 0.35
    
    ax3.bar(x - width/2, mse_values, width, label='MSE', color='skyblue')
    ax3.set_ylabel('Mean Squared Error')
    
    ax4 = ax3.twinx()
    ax4.bar(x + width/2, r2_values, width, label='R²', color='lightgreen')
    ax4.set_ylabel('R² Score')
    
    ax3.set_title('Model Performance Comparison (MSE vs R²)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45)
    
    # Add legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines4, labels4 = ax4.get_legend_handles_labels()
    ax3.legend(lines1 + lines4, labels1 + labels4, loc='upper right')
    
    # 4. Scatter Plot of MSE vs R2
    ax5 = fig.add_subplot(gs[0, 3])
    ax5.scatter(mse_values, r2_values, s=100)
    for i, model in enumerate(models):
        ax5.annotate(model, (mse_values[i], r2_values[i]), xytext=(5, 5), textcoords='offset points')
    ax5.set_xlabel('Mean Squared Error')
    ax5.set_ylabel('R² Score')
    ax5.set_title('MSE vs R² Score for Different Models')
    
    # Screen 2: Advanced Metrics
    ax6 = fig.add_subplot(gs[1, 0])
    ax6.violinplot([mse_values])
    ax6.set_title('MSE Distribution')
    ax6.set_ylabel('Mean Squared Error')
    
    # 2. Box Plot of R2
    ax7 = fig.add_subplot(gs[1, 1])
    ax7.boxplot(r2_values)
    ax7.set_title('R² Score Distribution')
    ax7.set_ylabel('R² Score')
    
    # 3. Scatter Matrix
    ax8 = fig.add_subplot(gs[1, 2])
    scatter = ax8.scatter(mse_values, r2_values, c=range(len(models)), cmap='viridis')
    plt.colorbar(scatter, ax=ax8, label='Model Index')  # Correct way to add colorbar
    ax8.set_title('MSE vs R² Score')
    ax8.set_xlabel('Mean Squared Error')
    ax8.set_ylabel('R² Score')
    
    # 4. Parallel Coordinates
    ax9 = fig.add_subplot(gs[1, 3])
    for i in range(len(models)):
        ax9.plot([0, 1], [mse_values[i], r2_values[i]], '-o')
    ax9.set_xticks([0, 1])
    ax9.set_xticklabels(['MSE', 'R²'])
    ax9.set_title('Parallel Coordinates Plot')
    
    # Screen 3: Comparative Analysis
    ax10 = fig.add_subplot(gs[2, 0])
    x = range(len(models))
    ax10.bar(x, mse_values, label='MSE')
    ax10.bar(x, r2_values, bottom=mse_values, label='R²')
    ax10.set_xticks(x)
    ax10.set_xticklabels(models, rotation=45)
    ax10.set_title('Stacked Metrics')
    ax10.legend()
    
    # 2. Area Plot
    ax11 = fig.add_subplot(gs[2, 1])
    ax11.fill_between(range(len(models)), mse_values, alpha=0.5, label='MSE')
    ax11.fill_between(range(len(models)), r2_values, alpha=0.5, label='R²')
    ax11.set_title('Area Plot of Metrics')
    ax11.legend()
    
    # 3. Radar Plot of Both Metrics
    ax = fig.add_subplot(gs[2, 2], projection='polar')
    angles = np.linspace(0, 2*np.pi, len(models), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))
    
    # Convert lists to numpy arrays and normalize
    mse_values_np = np.array(mse_values)
    r2_values_np = np.array(r2_values)
    
    # Normalize MSE (smaller is better)
    mse_normalized = 1 - (mse_values_np / np.max(mse_values_np))
    mse_values_plot = np.concatenate((mse_normalized, [mse_normalized[0]]))
    
    # Normalize R2 (already between 0 and 1 typically)
    r2_values_plot = np.concatenate((r2_values_np, [r2_values_np[0]]))
    
    ax.plot(angles, mse_values_plot, 'o-', label='MSE (normalized)')
    ax.plot(angles, r2_values_plot, 'o-', label='R²')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(models)
    ax.set_title('Radar Plot of Both Metrics')
    ax.legend()
    
    # 4. Bubble Plot
    ax12 = fig.add_subplot(gs[2, 3])
    sizes = np.array(r2_values) * 1000
    ax12.scatter(range(len(models)), mse_values, s=sizes, alpha=0.6)
    ax12.set_title('Bubble Plot (size = R² Score)')
    ax12.set_xlabel('Model Index')
    ax12.set_ylabel('Mean Squared Error')
    
    plt.subplots_adjust(top=0.95)  # Adjust top margin for main title
    plt.show()

def main():
    # Use the specific dataset path
    dataset_path = "dataset/Airplane_Crashes_and_Fatalities_Since_1908.csv"
    
    try:
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        data = pd.read_csv(dataset_path)
        
        # Add this diagnostic code
        print("\nTime Column Analysis:")
        print(f"Data type: {data['Time'].dtype}")
        print(f"Number of unique values: {len(data['Time'].unique())}")
        print(f"Total number of rows: {len(data['Time'])}")
        print(f"Unique ratio: {len(data['Time'].unique()) / len(data['Time'])}")
        print("Sample values:", data['Time'].head().tolist())
        
        # Analyze dataset before preprocessing
        print("\nInitial Dataset Analysis:")
        analyze_dataset(data)
        
        # Preprocess the data
        print("\nPreprocessing the data...")
        data = preprocess_data(data)
        
        # Separate features and target
        print("\nAvailable columns:")
        for i, col in enumerate(data.columns):
            print(f"{i+1}. {col}")
        
        print("\nPlease enter the number of the target column:")
        target_idx = int(input().strip()) - 1
        target_column = data.columns[target_idx]
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Determine task type using original dtype information
        task_type = determine_task_type(y, data.original_dtypes[target_column])
        print(f"\nTask Type: {task_type}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Evaluate models and create visualizations
        if task_type == 'classification':
            print("\nClassification Model Results:")
            results = evaluate_classification_models(X_train_scaled, X_test_scaled, y_train, y_test)
            print("\nSummary of Results:")
            print("-" * 50)
            for model, metrics in results.items():
                if metrics is not None:
                    print(f"{model}:")
                    print(f"  Accuracy: {metrics['Accuracy']:.4f}")
                    print(f"  F1 Score: {metrics['F1']:.4f}")
                    print("-" * 30)
            
            # Generate classification visualizations
            plot_classification_results(results)
            print("\nVisualization plots have been displayed")
        
        else:
            print("\nRegression Model Results:")
            results = evaluate_regression_models(X_train_scaled, X_test_scaled, y_train, y_test)
            for model, metrics in results.items():
                if metrics is not None:
                    print(f"{model}:")
                    print(f"  MSE: {metrics['MSE']:.4f}")
                    print(f"  R2: {metrics['R2']:.4f}")
            
            # Generate regression visualizations
            plot_regression_results(results)
            print("\nVisualization plots have been displayed:")
    
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main() 