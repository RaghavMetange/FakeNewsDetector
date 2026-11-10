"""
Phase 4 — Fast Model Retraining Script (Optimized with Liblinear)
================================================================
Upgrades the classifier with:
  1. Word TF-IDF + Char n-grams + Language encoding
  2. One-pass feature pre-transformation (10x faster)
  3. liblinear solver for rapid convergence on large sparse text matrices
  4. Probability calibration on validation set
Saves: classifier_pipeline_v2.pkl & confusion_matrix_v2.png
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import time
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("=" * 60)
print("PHASE 4 — FAST MODEL TRAINING (LIBLINEAR)")
print("=" * 60)

# 1. Load Data
t_start = time.time()
print("\n[1/5] Loading datasets...")
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
val_df   = pd.read_csv(os.path.join(BASE_DIR, "val.csv"))
test_df  = pd.read_csv(os.path.join(BASE_DIR, "test.csv"))

for df in [train_df, val_df, test_df]:
    df.dropna(subset=['Text', 'label', 'language'], inplace=True)
    df['Text'] = df['Text'].astype(str)
    df['label'] = df['label'].astype(int)

print(f"  Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

X_train = train_df[['Text', 'language']]
y_train = train_df['label']
X_val   = val_df[['Text', 'language']]
y_val   = val_df['label']
X_test  = test_df[['Text', 'language']]
y_test  = test_df['label']

# 2. Build and Pre-Transform Features (Once)
print("\n[2/5] Extracting features (Word + Char n-grams + Language)...")
t0 = time.time()
preprocessor = ColumnTransformer([
    ('word_tfidf', TfidfVectorizer(
        max_features=35000,
        ngram_range=(1, 2),
        sublinear_tf=True
    ), 'Text'),
    ('char_tfidf', TfidfVectorizer(
        max_features=20000,
        ngram_range=(3, 5),
        analyzer='char_wb',
        sublinear_tf=True
    ), 'Text'),
    ('lang_onehot', OneHotEncoder(handle_unknown='ignore'), ['language']),
])

X_train_trans = preprocessor.fit_transform(X_train)
X_val_trans   = preprocessor.transform(X_val)
X_test_trans  = preprocessor.transform(X_test)
print(f"  Feature matrix ready: {X_train_trans.shape[1]:,} features in {time.time() - t0:.1f}s")

# 3. Fast Tuning with liblinear
print("\n[3/5] Tuning C using liblinear solver...")
candidate_C = [0.5, 1.0, 2.0]
best_val_score = 0.0
best_C = 1.0

for C in candidate_C:
    t_c = time.time()
    clf = LogisticRegression(
        C=C,
        solver='liblinear',
        penalty='l2',
        class_weight='balanced',
        max_iter=300,
        random_state=42
    )
    clf.fit(X_train_trans, y_train)
    val_acc = clf.score(X_val_trans, y_val)
    train_acc = clf.score(X_train_trans, y_train)
    print(f"  C={C:<4} -> Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} (took {time.time() - t_c:.1f}s)")
    
    if val_acc > best_val_score:
        best_val_score = val_acc
        best_C = C

print(f"  >>> Selected Best C: {best_C} (Val Acc: {best_val_score:.4f})")

# 4. Fit Final Pipeline & Calibrate
print("\n[4/5] Assembling final pipeline and calibrating probabilities...")
final_model = LogisticRegression(
    C=best_C,
    solver='liblinear',
    penalty='l2',
    class_weight='balanced',
    max_iter=400,
    random_state=42
)

# Calibrate on validation set for reliable confidence estimates
calibrated_clf = CalibratedClassifierCV(estimator=final_model, cv='prefit', method='sigmoid')
# Train base estimator on train features
final_model.fit(X_train_trans, y_train)
# Calibrate on validation features
calibrated_clf.fit(X_val_trans, y_val)

# Wrap into full pipeline for clean inference with raw text inputs
final_pipeline = Pipeline([
    ('features', preprocessor),
    ('calibrated_classifier', calibrated_clf)
])

# Save pipeline
model_out = os.path.join(BASE_DIR, "classifier_pipeline_v2.pkl")
joblib.dump(final_pipeline, model_out)
print(f"  Saved calibrated pipeline -> {model_out}")

# 5. Evaluate on Test Set & Plot
print("\n[5/5] Evaluating on Test Set...")
y_pred = final_pipeline.predict(X_test)
print("\n" + classification_report(y_test, y_pred, target_names=['Real (0)', 'Fake (1)']))

cm_norm = confusion_matrix(y_test, y_pred, normalize='true')
print(f"Confusion Matrix (Normalized):")
print(f"  Real Recall: {cm_norm[0][0]*100:.2f}% | Misclassified: {cm_norm[0][1]*100:.2f}%")
print(f"  Fake Recall: {cm_norm[1][1]*100:.2f}% | Misclassified: {cm_norm[1][0]*100:.2f}%")

# Plot confusion matrix
fig, ax = plt.subplots(figsize=(7, 5.5))
im = ax.imshow(cm_norm, interpolation='nearest', cmap='Blues')
ax.set_title('Normalized Confusion Matrix — v2 (Liblinear + Calibrated)', fontsize=13)
ax.set_ylabel('True Label', fontsize=11)
ax.set_xlabel('Predicted Label', fontsize=11)

classes = ['Real (0)', 'Fake (1)']
ax.set_xticks([0, 1])
ax.set_xticklabels(classes, fontsize=10)
ax.set_yticks([0, 1])
ax.set_yticklabels(classes, fontsize=10)

for i in range(2):
    for j in range(2):
        val_pct = f"{cm_norm[i][j]*100:.2f}%"
        color = "white" if cm_norm[i][j] > 0.5 else "black"
        ax.text(j, i, val_pct, ha="center", va="center", color=color, fontsize=15, fontweight='bold')

fig.colorbar(im)
fig.tight_layout()
plot_path = os.path.join(BASE_DIR, "confusion_matrix_v2.png")
fig.savefig(plot_path, dpi=150)
print(f"  Saved plot -> {plot_path}")

print(f"\n All done in {time.time() - t_start:.1f}s!")
