import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# --- Step 1: Data Loading ---

try:
    # This simple command will work for a clean, standard CSV file.
    # Note we are now reading the NEW, clean file.
    df = pd.read_csv('hero_data.csv')
    print("✅ Clean file loaded successfully!")

    print("\n--- Data Summary After Loading ---")
    df.info()

except FileNotFoundError:
    print("❌ Error: 'hero_data.csv' not found. Please make sure you have followed the steps to create this new file.")
    exit()
except Exception as e:
    print(f"❌ An unexpected error occurred while reading the file: {e}")
    exit()

# --- Step 2: Data Cleaning and Preprocessing ---

# Use the last column name as the target variable
target_column = df.columns[-1]
print(f"\nIdentified target column: '{target_column}'")

# Define feature types based on data types
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

if target_column in categorical_features:
    categorical_features.remove(target_column)
if target_column in numerical_features:
    numerical_features.remove(target_column)

df.dropna(subset=[target_column], inplace=True)
df[target_column] = df[target_column].astype(int)

# Map target variable (assuming 0-4 mapping)
purchase_type_mapping = {
    0: 'Loyal', 1: 'Upgrade_Hero', 2: 'Churn_Premium',
    3: 'Churn_Commuter', 4: 'Churn_Scooter'
}
df['Purchase_Intent'] = df[target_column].map(purchase_type_mapping)
df.dropna(subset=['Purchase_Intent'], inplace=True)

if df.empty:
    print("❌ Error: DataFrame is empty after cleaning. No valid target values found.")
    exit()

# Impute (fill) missing values for features
for col in numerical_features:
    df[col].fillna(df[col].median(), inplace=True)
for col in categorical_features:
    df[col].fillna(df[col].mode()[0], inplace=True)

X = df.drop(columns=[target_column, 'Purchase_Intent'])
y = df['Purchase_Intent']

print(f"\n✅ Data cleaning complete. Shape of training data: {X.shape}")

# --- Step 3: Model Pipeline ---

categorical_transformer = OneHotEncoder(handle_unknown='ignore')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'))
])

# --- Step 4: Training, Evaluation, and Visualization ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model_pipeline.fit(X_train, y_train)
print("\n✅ Model training complete.")

y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print(f"\n📊 Model Accuracy: {accuracy:.2f}")
print("\n📋 Classification Report:\n", report)

try:
    ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    final_feature_names = numerical_features + list(ohe_feature_names)

    plt.figure(figsize=(25, 20))
    plot_tree(model_pipeline.named_steps['classifier'], feature_names=final_feature_names, class_names=sorted(y.unique()), filled=True, rounded=True, fontsize=9)
    plt.title("Decision Tree for Predicting Future Purchase Intent", fontsize=20)
    plt.savefig("decision_tree.png")
    print("\n✅ Decision tree visualization saved as decision_tree.png")
except Exception as e:
    print(f"\n❌ Could not generate the decision tree plot. Error: {e}")

# --- Step 3: Model Pipeline ---

# We are reducing max_depth to 3 and adding min_samples_leaf=20 for a cleaner, more robust tree.
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=20,
        random_state=42,
        class_weight='balanced'
    ))
])

# --- Step 4: Training and Evaluation ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model_pipeline.fit(X_train, y_train)
print("\n✅ Model training complete.")

y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print(f"\n📊 Model Accuracy: {accuracy:.2f}")
print("\n📋 Classification Report:\n", report)


# --- Step 5: Visualization ---
try:
    # Get the feature names after all preprocessing
    ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    final_feature_names = numerical_features + list(ohe_feature_names)

    plt.figure(figsize=(20, 12)) # Adjusted figure size for better fit
    plot_tree(
        model_pipeline.named_steps['classifier'],
        feature_names=final_feature_names,
        class_names=sorted(y.unique()),
        filled=True,
        rounded=True,
        fontsize=10,
        proportion=True, # Shows proportions of classes instead of raw counts
        precision=2      # Limits decimal places
    )
    plt.title("Pruned Decision Tree for Predicting Future Purchase Intent (Max Depth = 3)", fontsize=16)
    plt.savefig("decision_tree_pruned.png", dpi=300) # Save with higher resolution
    print("\n✅ Pruned decision tree visualization saved as decision_tree_pruned.png")

except Exception as e:
    print(f"\n❌ Could not generate the decision tree plot. Error: {e}")

# --- BONUS: View the Most Important Features ---
# This helps understand what the model learned, even without looking at the tree.

print("\n--- Top 10 Most Important Features ---")
try:
    # Get importance scores from the trained classifier
    importances = model_pipeline.named_steps['classifier'].feature_importances_
    
    # Create a pandas Series for easier viewing
    feature_importance_df = pd.Series(importances, index=final_feature_names).sort_values(ascending=False)
    
    print(feature_importance_df.head(10))

except Exception as e:
    print(f"Could not calculate feature importances. Error: {e}")