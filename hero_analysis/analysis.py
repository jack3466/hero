import os
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

# --- Debug: Check current directory and files ---
print("Current Working Directory:", os.getcwd())
print("Files in this directory:", os.listdir())

# --- Step 1: Data Loading ---
try:
    df = pd.read_csv('hero_data.csv', encoding='latin1', on_bad_lines='skip', engine='python')  
    print("File loaded successfully.")
except FileNotFoundError:
    print("Error: 'hero_data.csv' not found. Please ensure the file is in the correct directory.")
    exit()

# --- Step 2: Data Cleaning and Preprocessing ---

# This custom cleaning logic handles the specific line-break issues in your file
i = 0
while i < len(df) - 1:
    # A row is considered 'broken' if most of its values are empty
    if df.iloc[i+1].isnull().sum() > df.shape[1] * 0.7:
        part1 = df.iloc[i].dropna().astype(str).values
        part2 = df.iloc[i+1].dropna().astype(str).values
        full_row_values = np.concatenate([part1, part2])
        
        if len(full_row_values) == len(df.columns):
            df.iloc[i] = full_row_values
        df.drop(df.index[i+1], inplace=True)
    else:
        i += 1
df.reset_index(drop=True, inplace=True)
df.dropna(how='all', inplace=True)

# Define and map the target variable
purchase_type_mapping = {
    0: 'Loyal',
    1: 'Upgrade_Hero',
    2: 'Churn_Premium',
    3: 'Churn_Commuter',
    4: 'Churn_Scooter'
}
df['Next_Purchase_Type'] = pd.to_numeric(df['Next_Purchase_Type'], errors='coerce')
df.dropna(subset=['Next_Purchase_Type'], inplace=True)
df['Next_Purchase_Type'] = df['Next_Purchase_Type'].astype(int)
df['Purchase_Intent'] = df['Next_Purchase_Type'].apply(lambda x: purchase_type_mapping.get(x))
df.dropna(subset=['Purchase_Intent'], inplace=True)

# Define features (X) and target (y)
X = df.drop(columns=['Next_Purchase_Type', 'Purchase_Intent'])
y = df['Purchase_Intent']

# Identify feature types for preprocessing
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(include=np.number).columns

# Create the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# --- Step 3: Model Building and Training ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', DecisionTreeClassifier(max_depth=5, random_state=42, class_weight='balanced'))])
model_pipeline.fit(X_train, y_train)
print("\nModel training complete.")

# --- Step 4: Evaluation and Interpretation ---
y_pred = model_pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, zero_division=0)

print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", report)

# Visualize the tree
ohe_feature_names = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
final_feature_names = list(numerical_features) + list(ohe_feature_names)

plt.figure(figsize=(25, 20))
plot_tree(model_pipeline.named_steps['classifier'],
          feature_names=final_feature_names,
          class_names=sorted(y.unique()),
          filled=True,
          rounded=True,
          fontsize=9)
plt.title("Decision Tree for Predicting Future Purchase Intent", fontsize=20)
plt.savefig("decision_tree.png")
print("\nDecision tree visualization saved as decision_tree.png")