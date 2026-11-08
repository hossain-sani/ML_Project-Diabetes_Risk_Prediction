import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

print("Loading preprocessed dataset...")
df = pd.read_csv(Path(__file__).resolve().parents[1] / "data" / "processed" / "diabetes_brfss2014_preprocessed.csv")
X = df.drop(columns=['DIABETES_T2'])
y = df['DIABETES_T2']

# 2/3 train, 1/3 test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1/3, random_state=42, stratify=y
)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Train target distribution:\n{y_train.value_counts(normalize=True)}")

# Define categorical columns
categorical_cols = [
    'GEN_HEALTH', 'AGE_GRP', 'BMI_CAT', 'CHECKUP_FREQ', 'INCOME_GRP', 'RACE_GRP',
    'METRO_STATUS', 'EMPLOYMENT', 'MARITAL_STATUS', 'EDUCATION', 'SLEEP_GRP',
    'HEALTH_COVERAGE_SRC', 'HOME_OWNERSHIP', 'SMOKING_STATUS'
]
binary_cols = [c for c in X.columns if c not in categorical_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
        ('bin', 'passthrough', binary_cols)
    ]
)

print("Transforming features...")
X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print(f"Encoded feature shape: {X_train_encoded.shape}")

print("Applying SMOTE to training set...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_encoded, y_train)
print(f"SMOTE balanced train shape: {X_train_smote.shape}, target: {y_train_smote.value_counts().to_dict()}")

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    print(f"\nTraining {name}...")
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_te)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_te)
    else:
        y_prob = y_pred

    acc = accuracy_score(y_te, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_te, y_pred).ravel()
    sens = tp / (tp + fn) # sensitivity / recall
    spec = tn / (tn + fp) # specificity
    auc = roc_auc_score(y_te, y_prob)

    print(f"{name:20s} | Acc: {acc:.4f} | Sens: {sens:.4f} | Spec: {spec:.4f} | AUC: {auc:.4f}")
    return {"Model": name, "Accuracy": acc, "Sensitivity": sens, "Specificity": spec, "AUC": auc}

results = []

# 1. Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
results.append(evaluate_model("Logistic Regression", lr, X_train_smote, y_train_smote, X_test_encoded, y_test))

# 2. Decision Tree
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, random_state=42)
results.append(evaluate_model("Decision Tree", dt, X_train_smote, y_train_smote, X_test_encoded, y_test))

# 3. Gaussian Naive Bayes
gnb = GaussianNB()
results.append(evaluate_model("Gaussian Naive Bayes", gnb, X_train_smote, y_train_smote, X_test_encoded, y_test))

# 4. Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
results.append(evaluate_model("Random Forest", rf, X_train_smote, y_train_smote, X_test_encoded, y_test))

# 5. Linear SVM
lsvm = CalibratedClassifierCV(LinearSVC(max_iter=2000, random_state=42))
results.append(evaluate_model("Linear SVM", lsvm, X_train_smote, y_train_smote, X_test_encoded, y_test))

# 6. Neural Network (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=20, random_state=42)
results.append(evaluate_model("Neural Network", mlp, X_train_smote, y_train_smote, X_test_encoded, y_test))

res_df = pd.DataFrame(results)
print("\n" + "="*50)
print("Experiment Results:")
print(res_df.to_string(index=False))
