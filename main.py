import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# ================= 1. LOAD DATA =================
# We load both files
try:
    labels_df = pd.read_csv("final_segmented_ecommerce_data.csv")
    rfm_df = pd.read_csv("rfm_data_before_clustering.csv")
    
    # Ensure CustomerID is treated as a string to avoid type-mismatch errors
    labels_df['CustomerID'] = labels_df['CustomerID'].astype(str)
    rfm_df['CustomerID'] = rfm_df['CustomerID'].astype(str)

    # ================= 2. MERGE DATA =================
    # Merging behavior (R,F,M) with the Label on CustomerID
    df = pd.merge(rfm_df, labels_df[['CustomerID', 'Cluster_Label']], on='CustomerID')
    
except KeyError:
    print("❌ Error: 'CustomerID' not found. Please re-run the export code in your Notebook.")
    exit()

# ================= 3. FEATURE SELECTION =================
features = ['Recency', 'Frequency', 'Monetary']
target = 'Cluster_Label'

X = df[features]
y = df[target]

# ================= 4. ENCODE & SPLIT =================
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Stratified split to keep the cluster distribution balanced
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

# ================= 5. MODEL TRAINING =================
# Using the optimized Random Forest parameters
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10, 
    class_weight='balanced',
    random_state=42
)

print("🚀 Training Customer Segmentation Classifier...")
model.fit(X_train, y_train)

# ================= 6. EVALUATION =================
print("\n--- Final Model Performance Report ---")
print(classification_report(y_test, model.predict(X_test), target_names=le.classes_))

# ================= 7. SAVE ASSETS =================
joblib.dump(model, "customer_model.pkl")
joblib.dump(le, "segment_encoder.pkl")
joblib.dump(features, "model_features.pkl")

print("\n✅ Assets saved. Ready for Streamlit!")