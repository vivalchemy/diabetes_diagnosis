import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

# Load the dataset
df = pd.read_csv("./diabetes.csv")

# Separate target variable
y = df["Outcome"]
X = df.drop("Outcome", axis=1)

# Impute missing values using KNN imputer
imputer = KNNImputer()
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Feature Engineering: Create BMI and Age categories
X_imputed["BMI_Category"] = pd.cut(
    X_imputed["BMI"],
    bins=[-1, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"],
)
X_imputed["Age_Group"] = pd.cut(
    X_imputed["Age"],
    bins=[-1, 30, 40, 50, 60, 100],
    labels=["<30", "30-40", "40-50", "50-60", ">60"],
)

# One-hot encode the categorical features
categorical_features = ["BMI_Category", "Age_Group"]
encoder = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), categorical_features)],
    remainder="passthrough",
)
X_encoded = encoder.fit_transform(X_imputed)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42
)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter Tuning for individual models
param_grid_rf = {"n_estimators": [100, 200, 300], "max_depth": [5, 10, 15]}
param_grid_etc = {"n_estimators": [100, 200, 300], "max_depth": [5, 10, 15]}
param_grid_xgb = {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7]}

rf_grid = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5)
etc_grid = GridSearchCV(ExtraTreesClassifier(), param_grid_etc, cv=5)
xgb_grid = GridSearchCV(XGBClassifier(), param_grid_xgb, cv=5)

rf_grid.fit(X_train_scaled, y_train)
etc_grid.fit(X_train_scaled, y_train)
xgb_grid.fit(X_train_scaled, y_train)

# Evaluate individual models
print("Individual Models:")
print(
    f"Random Forest Accuracy: {accuracy_score(y_test, rf_grid.predict(X_test_scaled)):.2f}"
)
print(
    f"Extra Trees Classifier Accuracy: {accuracy_score(y_test, etc_grid.predict(X_test_scaled)):.2f}"
)
print(
    f"XGBoost Accuracy: {accuracy_score(y_test, xgb_grid.predict(X_test_scaled)):.2f}"
)

# Ensemble Model
voting_clf = VotingClassifier(
    estimators=[
        ("rf", rf_grid.best_estimator_),
        ("etc", etc_grid.best_estimator_),
        ("xgb", xgb_grid.best_estimator_),
    ],
    voting="soft",
)
voting_clf.fit(X_train_scaled, y_train)

# Evaluate the ensemble model
print("\nEnsemble Model:")
y_pred = voting_clf.predict(X_test_scaled)
print(f"Ensemble Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print(f"Ensemble Precision: {precision_score(y_test, y_pred):.2f}")
print(f"Ensemble Recall: {recall_score(y_test, y_pred):.2f}")
print(f"Ensemble F1-Score: {f1_score(y_test, y_pred):.2f}")
print(f"Ensemble MCC: {matthews_corrcoef(y_test, y_pred):.2f}")

# Perform cross-validation on the entire dataset with scaled data
print("\nCross-Validation Results:")
X_scaled = scaler.fit_transform(X_encoded)
scores = cross_val_score(voting_clf, X_scaled, y, cv=5, scoring="accuracy")
print(f"Cross-Validation Accuracy: {scores.mean():.2f}")  # Load the dataset
