# =========================================================
# Predictive Maintenance Final Project
# Covers Sessions 1, 2, 3, 4
# Improved Accuracy Version
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score,
    mean_squared_error,
    r2_score
)

# =========================================================
# Session 1 & 2: Load and Clean Data
# =========================================================
print("\n[1] Loading and Cleaning Data")

df = pd.read_csv("predictive_maintenance.csv")

print("Dataset shape:", df.shape)
print(df.head())

# Drop ID columns
df_clean = df.drop(['UDI', 'Product ID'], axis=1)

# Encode categorical column
le = LabelEncoder()
df_clean['Type'] = le.fit_transform(df_clean['Type'])

# Features & Target
X = df_clean.drop(['Target', 'Failure Type'], axis=1)
y = df_clean['Target']

# =========================================================
# Session 2: EDA & Visualization
# =========================================================
print("\n[2] Exploratory Data Analysis")

plt.figure(figsize=(10, 6))
sns.heatmap(df_clean.drop(['Failure Type'], axis=1).corr(),
            annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(6, 4))
sns.countplot(x='Target', data=df_clean)
plt.title("Target Distribution (0 = No Failure, 1 = Failure)")
plt.show()

# =========================================================
# Feature Scaling (Important Improvement)
# =========================================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================================================
# Train-Test Split (Stratified)
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================================================
# Session 3 & 4: Supervised Learning
# =========================================================
print("\n[3] Supervised Learning Models")

# ---- Decision Tree (Tuned) ----
dt_model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42
)

dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\nDecision Tree Results")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Recall (Failure):", recall_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_dt),
            annot=True, fmt='d', cmap='Blues')
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ---- Random Forest (Higher Accuracy) ----
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    class_weight='balanced',
    random_state=42
)

rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\nRandom Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Recall (Failure):", recall_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred_rf),
            annot=True, fmt='d', cmap='Greens')
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# =========================================================
# Session 4: Unsupervised Learning (Clustering)
# =========================================================
print("\n[4] Unsupervised Learning - KMeans")

kmeans = KMeans(n_clusters=3, random_state=42)
df_clean['Cluster'] = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_clean,
    x='Rotational speed [rpm]',
    y='Air temperature [K]',
    hue='Cluster',
    palette='viridis'
)
plt.title("K-Means Clustering: Speed vs Temperature")
plt.show()

# =========================================================
# Session 3: Linear Regression + R²
# =========================================================
print("\n[5] Linear Regression (Regression Task)")

X_reg = df_clean[['Process temperature [K]']]
y_reg = df_clean['Air temperature [K]']

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

reg_model = LinearRegression()
reg_model.fit(X_train_r, y_train_r)

y_pred_r = reg_model.predict(X_test_r)

r2 = r2_score(y_test_r, y_pred_r)
mse = mean_squared_error(y_test_r, y_pred_r)

print("R-Squared:", r2)
print("Mean Squared Error:", mse)

plt.figure(figsize=(8, 6))
plt.scatter(X_test_r, y_test_r, alpha=0.5, label="Actual")
plt.plot(X_test_r, y_pred_r, color='red', label="Regression Line")
plt.xlabel("Process Temperature [K]")
plt.ylabel("Air Temperature [K]")
plt.title("Linear Regression: Process vs Air Temperature")
plt.legend()
plt.show()

print("\n✅ Project Executed Successfully!")
