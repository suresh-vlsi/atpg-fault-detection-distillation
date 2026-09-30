# ============================================================
# ATGP FAULT DETECTION - TEACHER MODEL
# ============================================================
#
# Purpose:
#   Train a high-capacity ML "Teacher Model" using the
#   ATPG fault-detection dataset.
#
# Dataset:
#   dataset/fault_detection_dataset.csv
#
# Columns:
#   A
#   B
#   C
#   fault_net
#   stuck_value
#   good_output
#   faulty_output
#   detected
#
# Target:
#   detected
#
# Model:
#   Random Forest Classifier
#
# Output:
#   models/teacher_model.pkl
#   models/teacher_features.pkl
#
# ============================================================


import os
import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/fault_detection_dataset.csv"

MODEL_DIR = "models"

TEACHER_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "teacher_model.pkl"
)

FEATURE_INFO_PATH = os.path.join(
    MODEL_DIR,
    "teacher_features.pkl"
)

RANDOM_STATE = 42


# ============================================================
# FAULT-NET ENCODING
# ============================================================

FAULT_NET_MAP = {
    "A": 0,
    "B": 1,
    "C": 2,
    "N1": 3,
    "N2": 4
}


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("ATPG FAULT DETECTION - TEACHER MODEL")
print("=" * 70)
print()


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    print("ERROR: Dataset not found.")
    print()
    print("Expected:")
    print(DATASET_PATH)
    print()

    raise FileNotFoundError(DATASET_PATH)


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")
print()

df = pd.read_csv(DATASET_PATH)


print("Dataset shape:")
print(df.shape)
print()

print("Columns:")
print(list(df.columns))
print()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "A",
    "B",
    "C",
    "fault_net",
    "stuck_value",
    "good_output",
    "faulty_output",
    "detected"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("ERROR: Missing required columns:")
    print(missing_columns)
    print()

    raise ValueError(
        "Dataset does not contain the expected columns."
    )


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print()

print("Total samples:")
print(len(df))

print()

print("Detected samples:")
print(int(df["detected"].sum()))

print()

print("Undetected samples:")
print(int((df["detected"] == 0).sum()))

print()

detection_rate = df["detected"].mean() * 100

print("Detection rate:")
print(f"{detection_rate:.2f}%")

print()


# ============================================================
# SHOW SAMPLE DATA
# ============================================================

print("=" * 70)
print("SAMPLE DATA")
print("=" * 70)

print()

print(df.head(10).to_string(index=False))

print()


# ============================================================
# CLEAN DATA
# ============================================================

print("=" * 70)
print("DATA CLEANING")
print("=" * 70)

print()


# Convert primary input columns to integers

df["A"] = pd.to_numeric(
    df["A"],
    errors="coerce"
)

df["B"] = pd.to_numeric(
    df["B"],
    errors="coerce"
)

df["C"] = pd.to_numeric(
    df["C"],
    errors="coerce"
)


# Convert stuck-at value

df["stuck_value"] = pd.to_numeric(
    df["stuck_value"],
    errors="coerce"
)


# Convert outputs

df["good_output"] = pd.to_numeric(
    df["good_output"],
    errors="coerce"
)

df["faulty_output"] = pd.to_numeric(
    df["faulty_output"],
    errors="coerce"
)


# Convert target

df["detected"] = pd.to_numeric(
    df["detected"],
    errors="coerce"
)


# ============================================================
# ENCODE FAULT NET
# ============================================================

print("Encoding fault-net names...")

df["fault_net_encoded"] = (
    df["fault_net"]
    .astype(str)
    .map(FAULT_NET_MAP)
)


# Check for unknown fault nets

unknown_fault_nets = df[
    df["fault_net_encoded"].isna()
]["fault_net"].unique()


if len(unknown_fault_nets) > 0:

    print()
    print("ERROR: Unknown fault-net values:")
    print(unknown_fault_nets)
    print()

    raise ValueError(
        "Unknown fault-net encountered in dataset."
    )


# ============================================================
# REMOVE INVALID ROWS
# ============================================================

before_cleaning = len(df)

df = df.dropna()

after_cleaning = len(df)

removed_rows = before_cleaning - after_cleaning

print()

print("Rows before cleaning:")
print(before_cleaning)

print()

print("Rows after cleaning:")
print(after_cleaning)

print()

print("Rows removed:")
print(removed_rows)

print()


# ============================================================
# FEATURE SELECTION
# ============================================================
#
# Features available to the teacher:
#
#   A
#   B
#   C
#   fault_net_encoded
#   stuck_value
#   good_output
#   faulty_output
#
# Target:
#
#   detected
#
# ============================================================


FEATURE_COLUMNS = [
    "A",
    "B",
    "C",
    "fault_net_encoded",
    "stuck_value",
    "good_output",
    "faulty_output"
]

TARGET_COLUMN = "detected"


X = df[FEATURE_COLUMNS].copy()

y = df[TARGET_COLUMN].astype(int)


# ============================================================
# DISPLAY FEATURES
# ============================================================

print("=" * 70)
print("FEATURE MATRIX")
print("=" * 70)

print()

print("Feature columns:")
print(FEATURE_COLUMNS)

print()

print("Feature matrix shape:")
print(X.shape)

print()

print("Target shape:")
print(y.shape)

print()


# ============================================================
# DISPLAY CLASS DISTRIBUTION
# ============================================================

print("=" * 70)
print("TARGET DISTRIBUTION")
print("=" * 70)

print()

class_counts = y.value_counts().sort_index()

for class_value, count in class_counts.items():

    if class_value == 0:
        label = "UNDETECTED"
    else:
        label = "DETECTED"

    percentage = count / len(y) * 100

    print(
        f"Class {class_value} ({label}) : "
        f"{count} samples ({percentage:.2f}%)"
    )

print()


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)


print("Training samples:")
print(len(X_train))

print()

print("Testing samples:")
print(len(X_test))

print()


# ============================================================
# CREATE TEACHER MODEL
# ============================================================

print("=" * 70)
print("TRAINING TEACHER MODEL")
print("=" * 70)

print()

print("Model: Random Forest Classifier")
print("Number of trees: 200")
print("Random state:", RANDOM_STATE)
print()


teacher_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    class_weight="balanced"
)


# ============================================================
# TRAIN
# ============================================================

teacher_model.fit(
    X_train,
    y_train
)


print("Teacher model training completed.")
print()


# ============================================================
# PREDICTION
# ============================================================

print("=" * 70)
print("TEACHER MODEL PREDICTION")
print("=" * 70)

print()


y_pred = teacher_model.predict(X_test)

y_probability = teacher_model.predict_proba(X_test)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("Accuracy : {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall   : {:.2f}%".format(recall * 100))
print("F1 Score : {:.2f}%".format(f1 * 100))

print()


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)

print()

print("Format:")
print("[[TN  FP]")
print(" [FN  TP]]")

print()


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print()

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Undetected",
            "Detected"
        ],
        zero_division=0
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("=" * 70)
print("TEACHER FEATURE IMPORTANCE")
print("=" * 70)

print()


feature_importance = teacher_model.feature_importances_


importance_data = sorted(
    zip(
        FEATURE_COLUMNS,
        feature_importance
    ),
    key=lambda x: x[1],
    reverse=True
)


for feature, importance in importance_data:

    print(
        f"{feature:<25} : "
        f"{importance:.6f}"
    )

print()


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# SAVE TEACHER MODEL
# ============================================================

print("=" * 70)
print("SAVING TEACHER MODEL")
print("=" * 70)

print()


with open(
    TEACHER_MODEL_PATH,
    "wb"
) as file:

    pickle.dump(
        teacher_model,
        file
    )


print("Teacher model saved:")
print(TEACHER_MODEL_PATH)

print()


# ============================================================
# SAVE FEATURE INFORMATION
# ============================================================

feature_information = {

    "feature_columns": FEATURE_COLUMNS,

    "target_column": TARGET_COLUMN,

    "fault_net_map": FAULT_NET_MAP,

    "random_state": RANDOM_STATE,

    "model_type": "RandomForestClassifier",

    "n_estimators": 200

}


with open(
    FEATURE_INFO_PATH,
    "wb"
) as file:

    pickle.dump(
        feature_information,
        file
    )


print("Feature information saved:")
print(FEATURE_INFO_PATH)

print()


# ============================================================
# TEACHER MODEL TEST EXAMPLES
# ============================================================

print("=" * 70)
print("TEACHER MODEL - SAMPLE PREDICTIONS")
print("=" * 70)

print()


sample_count = min(
    10,
    len(X_test)
)


sample_X = X_test.iloc[
    :sample_count
]

sample_y = y_test.iloc[
    :sample_count
]

sample_predictions = teacher_model.predict(
    sample_X
)

sample_probabilities = teacher_model.predict_proba(
    sample_X
)


for i in range(sample_count):

    actual = int(
        sample_y.iloc[i]
    )

    predicted = int(
        sample_predictions[i]
    )

    probability_detected = (
        sample_probabilities[i][1]
        if len(sample_probabilities[i]) > 1
        else 0.0
    )

    print(
        f"Sample {i + 1:02d} | "
        f"Actual={actual} | "
        f"Predicted={predicted} | "
        f"P(Detected)={probability_detected:.4f}"
    )

print()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 70)
print("TEACHER MODEL SUMMARY")
print("=" * 70)

print()

print(f"Dataset samples       : {len(df)}")
print(f"Training samples      : {len(X_train)}")
print(f"Testing samples       : {len(X_test)}")

print()

print(
    "Teacher accuracy      : "
    f"{accuracy * 100:.2f}%"
)

print(
    "Teacher precision     : "
    f"{precision * 100:.2f}%"
)

print(
    "Teacher recall        : "
    f"{recall * 100:.2f}%"
)

print(
    "Teacher F1 score      : "
    f"{f1 * 100:.2f}%"
)

print()

print("Teacher model:")
print(TEACHER_MODEL_PATH)

print()

print("Feature metadata:")
print(FEATURE_INFO_PATH)

print()

print("=" * 70)
print("TEACHER MODEL TRAINING COMPLETE")
print("=" * 70)

print()