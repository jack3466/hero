import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

# === Load Data ===
try:
    df = pd.read_csv("cleaned_data.csv")
    print("✅ Clean file loaded successfully!\n")
except Exception as e:
    print(f"❌ Failed to load cleaned data: {e}")
    exit()

# === Show Summary ===
print("\n--- Data Summary After Loading ---")
print(df.info())

# === Drop rows where target is NaN ===
TARGET_COLUMN = 'Next_Purchase_Type'
df = df.dropna(subset=[TARGET_COLUMN])

# === Encode Categorical Columns ===
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# === Separate Features and Target ===
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

# === Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Data cleaned. Training shape: {X_train.shape}")

# === Train Model ===
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))
