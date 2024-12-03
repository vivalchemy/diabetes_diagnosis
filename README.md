# Diabetes diagnosis using ensemble model

## Project Overview

The `Diabetes diagnosis using ensemble model` project aims to predict the outcome of diabetes diagnosis based on various clinical features using machine learning models. The project implements data preprocessing, model training, and evaluation for diabetes classification using an ensemble approach. It leverages several classification algorithms, including Random Forest, Extra Trees, and XGBoost, combined into a voting classifier for improved performance.

## Requirements

- Python version: `>=3.12`
- Libraries:
  - `joblib>=1.4.2`
  - `kagglehub>=0.3.3`
  - `matplotlib>=3.9.2`
  - `pandas>=2.2.3`
  - `scikit-learn>=1.5.2`
  - `seaborn>=0.13.2`
  - `xgboost>=2.1.2`
  - `jupyter>=1.1.1`

## Installation

To install the required dependencies, you can create a virtual environment and install the packages listed in the `pyproject.toml` file:

```bash
# Create a virtual environment
uv venv .venv

# Activate the virtual environment
# On Windows
.venv\Scripts\activate
# On macOS/Linux
source .venv/bin/activate

# Install dependencies
uv sync 

# To run the project
uv run --with jupyter jupyter lab
```

## Dataset

The project uses a dataset `diabetes.csv`, which contains information about various clinical features for diagnosing diabetes. The dataset is preprocessed to handle missing values and categorize features like BMI and Age into meaningful categories.

## Usage

To train the model, run the following script:

```bash
python main.py
```

This will:
- Load and preprocess the diabetes dataset.
- Handle missing values using KNN imputation.
- Create new categorical features for BMI and Age.
- Encode categorical variables using One-Hot Encoding.
- Split the data into training and test sets.
- Scale the features using StandardScaler.
- Tune hyperparameters for Random Forest, Extra Trees, and XGBoost models.
- Train a voting classifier combining these models.
- Evaluate the ensemble model's performance using accuracy, precision, recall, F1-score, and Matthews Correlation Coefficient (MCC).
- Perform cross-validation to evaluate the model further.
- Save the trained ensemble model as `diabetes_ensemble_model.joblib`.

## Model Performance

The ensemble model's performance is evaluated based on the following metrics:

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **MCC (Matthews Correlation Coefficient)**

The performance metrics are displayed in a bar chart for easy visualization.

## Model Saving

The final trained ensemble model is saved as `diabetes_ensemble_model.joblib` using `joblib` for future use or deployment.

## Visualization

The project includes visualizations of data distributions for categorical features (BMI and Age groups) and model performance metrics using `matplotlib` and `seaborn`.

## File Structure

```
.
├── diabetes.csv                  # The diabetes dataset
├── diabetes_ensemble_model.joblib # Trained ensemble model
├── main.py                        # Script for training and evaluation
├── pyproject.toml                 # Project dependencies and configuration
├── README.md                      # Project overview and instructions
└── requirements.txt               # Python dependencies
```

## License

This project is licensed under the MIT License.
