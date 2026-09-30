import os
import pickle
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/fault_detection_dataset.csv"
MODEL_PATH = "models/student_model.pkl"
FEATURE_PATH = "models/student_features.pkl"

OUTPUT_PATH = "dataset/final_demo_results.csv"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def encode_fault_net(value):
    """
    Fallback encoding for the fault-net names used in this project.
    """
    mapping = {
        "A": 0,
        "B": 1,
        "C": 2,
        "N1": 3,
        "N2": 4
    }

    value = str(value).strip()

    if value in mapping:
        return mapping[value]

    return -1


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ATPG FAULT DETECTION - FINAL STUDENT MODEL DEMO")
print("=" * 70)


# ============================================================
# 1. CHECK FILES
# ============================================================

section("CHECKING PROJECT FILES")

required_files = [
    DATASET_PATH,
    MODEL_PATH,
    FEATURE_PATH
]

for path in required_files:
    if os.path.exists(path):
        print(f"[OK] {path}")
    else:
        print(f"[ERROR] Missing: {path}")
        raise FileNotFoundError(path)


# ============================================================
# 2. LOAD DATASET
# ============================================================

section("LOADING DATASET")

df = pd.read_csv(DATASET_PATH)

print(f"Dataset shape : {df.shape}")
print(f"Columns       : {list(df.columns)}")


# ============================================================
# 3. LOAD STUDENT MODEL
# ============================================================

section("LOADING STUDENT MODEL")

student_model = load_pickle(MODEL_PATH)
feature_metadata = load_pickle(FEATURE_PATH)

print(f"Student model loaded : {MODEL_PATH}")
print(f"Feature metadata     : {FEATURE_PATH}")


# ============================================================
# 4. DETERMINE FEATURE LIST
# ============================================================

section("FEATURE INFORMATION")

print("Feature metadata type:")
print(type(feature_metadata))

feature_names = None

if isinstance(feature_metadata, dict):

    print("Metadata keys:")
    print(list(feature_metadata.keys()))

    possible_keys = [
        "features",
        "feature_names",
        "selected_features",
        "columns"
    ]

    for key in possible_keys:
        if key in feature_metadata:
            feature_names = feature_metadata[key]
            break

elif isinstance(feature_metadata, (list, tuple)):
    feature_names = list(feature_metadata)


# ------------------------------------------------------------
# If metadata does not explicitly contain feature names,
# use the feature structure used by our student model.
# ------------------------------------------------------------

if feature_names is None:

    feature_names = [
        "A",
        "B",
        "C",
        "fault_net_encoded",
        "stuck_value",
        "good_output",
        "faulty_output"
    ]

print()
print("Features expected by student model:")

for i, feature in enumerate(feature_names, 1):
    print(f"{i:02d}. {feature}")


# ============================================================
# 5. CREATE MODEL INPUT
# ============================================================

section("PREPARING INPUT FEATURES")


work = df.copy()


# ------------------------------------------------------------
# Create encoded fault-net feature if necessary
# ------------------------------------------------------------

if "fault_net_encoded" in feature_names:

    if "fault_net_encoded" not in work.columns:

        if "fault_net" not in work.columns:
            raise ValueError(
                "Dataset contains neither fault_net nor fault_net_encoded"
            )

        work["fault_net_encoded"] = (
            work["fault_net"]
            .apply(encode_fault_net)
        )


# ------------------------------------------------------------
# Verify required columns
# ------------------------------------------------------------

missing = [
    feature
    for feature in feature_names
    if feature not in work.columns
]

if missing:

    print()
    print("Missing features:")
    print(missing)

    print()
    print("Available columns:")
    print(list(work.columns))

    raise ValueError(
        f"Cannot prepare model input. Missing features: {missing}"
    )


X = work[feature_names].copy()


# Convert everything to numeric
for column in X.columns:
    X[column] = pd.to_numeric(
        X[column],
        errors="coerce"
    )


if X.isnull().any().any():

    print("WARNING: NaN values detected.")

    X = X.fillna(0)


print(f"Input matrix shape : {X.shape}")


# ============================================================
# 6. STUDENT PREDICTION
# ============================================================

section("RUNNING STUDENT MODEL")


prediction = student_model.predict(X)

prediction = np.asarray(prediction).astype(int)


# ------------------------------------------------------------
# Probability if available
# ------------------------------------------------------------

probability = None

if hasattr(student_model, "predict_proba"):

    probabilities = student_model.predict_proba(X)

    if probabilities.shape[1] >= 2:
        probability = probabilities[:, 1]

    else:
        probability = probabilities[:, 0]

else:

    probability = prediction.astype(float)


# ============================================================
# 7. CREATE RESULTS
# ============================================================

results = pd.DataFrame()

results["sample"] = np.arange(1, len(df) + 1)

results["A"] = df["A"]
results["B"] = df["B"]
results["C"] = df["C"]

if "fault_net" in df.columns:
    results["fault_net"] = df["fault_net"]

results["stuck_value"] = df["stuck_value"]
results["good_output"] = df["good_output"]
results["faulty_output"] = df["faulty_output"]

if "detected" in df.columns:
    results["actual_detected"] = df["detected"]

results["student_prediction"] = prediction
results["student_probability"] = probability

results["prediction_label"] = np.where(
    prediction == 1,
    "FAULT DETECTED",
    "FAULT NOT DETECTED"
)


# ============================================================
# 8. DISPLAY DEMO RESULTS
# ============================================================

section("FINAL STUDENT MODEL PREDICTIONS")

display_columns = [
    "sample",
    "student_prediction",
    "student_probability",
    "prediction_label"
]

if "actual_detected" in results.columns:
    display_columns.insert(
        1,
        "actual_detected"
    )

print(
    results[display_columns]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# 9. CALCULATE ACCURACY
# ============================================================

if "actual_detected" in results.columns:

    actual = results["actual_detected"].astype(int)

    accuracy = np.mean(
        prediction == actual
    )

    print()
    print(f"Student accuracy on dataset : {accuracy * 100:.2f}%")


# ============================================================
# 10. SAVE RESULTS
# ============================================================

section("SAVING FINAL DEMO RESULTS")

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)

print(f"Results saved:")
print(OUTPUT_PATH)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

section("FINAL DEMO COMPLETE")

print("Student model       : LOADED")
print("Inference           : SUCCESSFUL")
print("Predictions         : GENERATED")
print("Results CSV         : SAVED")

print()
print("Output file:")
print(OUTPUT_PATH)

print()
print("=" * 70)
print("ATPG FAULT DETECTION DEMO COMPLETED SUCCESSFULLY")
print("=" * 70)