# ============================================================
# student_model.py
# ATGP FAULT DETECTION - STUDENT MODEL
#
# Knowledge Distillation:
# Teacher model -> soft predictions -> Student model
# ============================================================

import os
import pickle
import warnings

import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = "dataset/fault_detection_dataset.csv"

TEACHER_MODEL_PATH = "models/teacher_model.pkl"
TEACHER_FEATURES_PATH = "models/teacher_features.pkl"

STUDENT_MODEL_PATH = "models/student_model.pkl"
STUDENT_FEATURES_PATH = "models/student_features.pkl"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_pickle(path):
    """Load a pickle file."""
    with open(path, "rb") as file:
        return pickle.load(file)


def save_pickle(obj, path):
    """Save an object as pickle."""
    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(path, "wb") as file:
        pickle.dump(obj, file)


# ============================================================
# LOAD DATASET
# ============================================================

print_header("ATPG FAULT DETECTION - STUDENT MODEL")

print("\nLoading dataset...")

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_PATH}"
    )

df = pd.read_csv(DATASET_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nColumns:")
print(list(df.columns))


# ============================================================
# DATASET INFORMATION
# ============================================================

if "detected" not in df.columns:
    raise ValueError(
        "Dataset does not contain the 'detected' column."
    )

print_header("DATASET INFORMATION")

total_samples = len(df)

detected_samples = int(df["detected"].sum())
undetected_samples = total_samples - detected_samples

detection_rate = (
    detected_samples / total_samples
    if total_samples > 0
    else 0
)

print(f"\nTotal samples      : {total_samples}")
print(f"Detected samples   : {detected_samples}")
print(f"Undetected samples : {undetected_samples}")
print(f"Detection rate     : {detection_rate:.2%}")


# ============================================================
# LOAD TEACHER MODEL
# ============================================================

print_header("LOADING TEACHER MODEL")

if not os.path.exists(TEACHER_MODEL_PATH):
    raise FileNotFoundError(
        f"Teacher model not found:\n{TEACHER_MODEL_PATH}"
    )

if not os.path.exists(TEACHER_FEATURES_PATH):
    raise FileNotFoundError(
        f"Teacher feature metadata not found:\n"
        f"{TEACHER_FEATURES_PATH}"
    )

teacher_model = load_pickle(
    TEACHER_MODEL_PATH
)

teacher_features = load_pickle(
    TEACHER_FEATURES_PATH
)

print("\nTeacher model loaded:")
print(TEACHER_MODEL_PATH)

print("\nFeature metadata loaded:")
print(TEACHER_FEATURES_PATH)


# ============================================================
# DETERMINE TEACHER FEATURE INFORMATION
# ============================================================

print_header("TEACHER FEATURE INFORMATION")

print("\nTeacher metadata type:")
print(type(teacher_features).__name__)

print("\nTeacher metadata:")

if isinstance(teacher_features, dict):

    for key, value in teacher_features.items():

        if isinstance(value, dict):
            print(f"{key}:")
            print(value)

        elif isinstance(value, list):
            print(f"{key}: {value}")

        else:
            print(f"{key}: {value}")

else:
    print(teacher_features)


# ============================================================
# EXTRACT FEATURE LIST
# ============================================================

teacher_feature_list = None
fault_mapping = None


if isinstance(teacher_features, dict):

    # Possible names used by different versions
    possible_feature_keys = [
        "features",
        "feature_columns",
        "feature_names",
        "input_features"
    ]

    for key in possible_feature_keys:

        if key in teacher_features:

            value = teacher_features[key]

            if isinstance(value, list):
                teacher_feature_list = value
                break


    # --------------------------------------------------------
    # Search for fault-net encoding
    # --------------------------------------------------------

    possible_mapping_keys = [
        "fault_net_mapping",
        "fault_mapping",
        "categorical_mapping"
    ]

    for key in possible_mapping_keys:

        if key in teacher_features:

            value = teacher_features[key]

            # Direct mapping
            if isinstance(value, dict):

                # Example:
                # {"A": 0, "B": 1, "C": 2}

                if all(
                    isinstance(v, (int, np.integer))
                    for v in value.values()
                ):
                    fault_mapping = value
                    break

                # Nested mapping:
                # {"fault_net": {"A": 0, ...}}

                if "fault_net" in value:

                    nested = value["fault_net"]

                    if isinstance(nested, dict):
                        fault_mapping = nested
                        break


# ============================================================
# FALLBACK FEATURE LIST
# ============================================================

if teacher_feature_list is None:

    print(
        "\nTeacher feature list was not explicitly found."
    )

    print(
        "Using the known teacher feature structure."
    )

    teacher_feature_list = [
        "A",
        "B",
        "C",
        "fault_net_encoded",
        "stuck_value",
        "good_output",
        "faulty_output"
    ]


# ============================================================
# FAULT-NET ENCODING
# ============================================================

print_header("FEATURE PREPARATION")

print("\nPreparing fault_net encoding...")


if "fault_net_encoded" in teacher_feature_list:

    if "fault_net_encoded" not in df.columns:

        if "fault_net" not in df.columns:

            raise ValueError(
                "Dataset contains neither 'fault_net' "
                "nor 'fault_net_encoded'."
            )

        # ----------------------------------------------------
        # Use mapping saved by teacher
        # ----------------------------------------------------

        if fault_mapping is not None:

            print(
                "\nUsing fault-net encoding from teacher metadata:"
            )

            print(fault_mapping)

            df["fault_net_encoded"] = (
                df["fault_net"]
                .astype(str)
                .map(fault_mapping)
            )

        # ----------------------------------------------------
        # Fallback
        # ----------------------------------------------------

        else:

            print(
                "\nTeacher mapping was not found."
            )

            print(
                "Creating deterministic encoding."
            )

            unique_faults = sorted(
                df["fault_net"]
                .astype(str)
                .unique()
            )

            fault_mapping = {
                fault: index
                for index, fault
                in enumerate(unique_faults)
            }

            print("\nGenerated fault-net mapping:")

            for fault, value in fault_mapping.items():
                print(
                    f"  {fault} -> {value}"
                )

            df["fault_net_encoded"] = (
                df["fault_net"]
                .astype(str)
                .map(fault_mapping)
            )


# ============================================================
# CHECK FOR UNKNOWN FAULTS
# ============================================================

if "fault_net_encoded" in df.columns:

    if df["fault_net_encoded"].isna().any():

        unknown_faults = sorted(
            df.loc[
                df["fault_net_encoded"].isna(),
                "fault_net"
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Unknown fault_net values found:\n"
            + str(unknown_faults)
        )


# ============================================================
# FEATURE LIST
# ============================================================

feature_columns = [
    "A",
    "B",
    "C",
    "fault_net_encoded",
    "stuck_value",
    "good_output",
    "faulty_output"
]


print("\nFeatures used by teacher:")

for index, feature in enumerate(
    feature_columns,
    start=1
):

    print(
        f"{index:02d}. {feature}"
    )


# ============================================================
# VERIFY FEATURES
# ============================================================

missing_features = [
    feature
    for feature in feature_columns
    if feature not in df.columns
]


if missing_features:

    raise ValueError(
        "Missing features in dataset: "
        + str(missing_features)
    )


print("\nAll required features are available.")


# ============================================================
# PREPARE INPUT DATA
# ============================================================

X = df[
    feature_columns
].astype(float)

y_true = (
    df["detected"]
    .astype(int)
    .values
)


print_header("FEATURE MATRIX")

print("\nFeature matrix shape:")
print(X.shape)

print("\nFeature matrix:")

print(
    X.head(10).to_string(index=False)
)


# ============================================================
# TEACHER PREDICTIONS
# ============================================================

print_header("TEACHER PREDICTIONS")

print("\nGenerating teacher predictions...")


# ------------------------------------------------------------
# Get teacher probabilities
# ------------------------------------------------------------

if hasattr(
    teacher_model,
    "predict_proba"
):

    teacher_probability = (
        teacher_model
        .predict_proba(X)[:, 1]
    )

elif hasattr(
    teacher_model,
    "decision_function"
):

    scores = teacher_model.decision_function(X)

    teacher_probability = (
        1.0 /
        (
            1.0 +
            np.exp(-scores)
        )
    )

else:

    teacher_prediction = (
        teacher_model
        .predict(X)
    )

    teacher_probability = (
        np.asarray(
            teacher_prediction,
            dtype=float
        )
    )


teacher_probability = np.asarray(
    teacher_probability,
    dtype=float
)


# ------------------------------------------------------------
# Teacher hard predictions
# ------------------------------------------------------------

teacher_prediction = (
    teacher_probability >= 0.5
).astype(int)


# ============================================================
# TEACHER PERFORMANCE ON COMPLETE DATASET
# ============================================================

teacher_accuracy = accuracy_score(
    y_true,
    teacher_prediction
)

teacher_precision = precision_score(
    y_true,
    teacher_prediction,
    zero_division=0
)

teacher_recall = recall_score(
    y_true,
    teacher_prediction,
    zero_division=0
)

teacher_f1 = f1_score(
    y_true,
    teacher_prediction,
    zero_division=0
)


print("\nTeacher performance:")

print(
    f"Accuracy  : {teacher_accuracy:.2%}"
)

print(
    f"Precision : {teacher_precision:.2%}"
)

print(
    f"Recall    : {teacher_recall:.2%}"
)

print(
    f"F1 score  : {teacher_f1:.2%}"
)


# ============================================================
# DISPLAY TEACHER OUTPUTS
# ============================================================

print("\nSample teacher outputs:")

print(
    "-" * 70
)

print(
    f"{'Sample':<10}"
    f"{'Actual':<10}"
    f"{'Teacher':<10}"
    f"{'Probability':<15}"
)

print(
    "-" * 70
)


for i in range(
    min(10, len(df))
):

    print(
        f"{i + 1:<10}"
        f"{y_true[i]:<10}"
        f"{teacher_prediction[i]:<10}"
        f"{teacher_probability[i]:<15.4f}"
    )


# ============================================================
# KNOWLEDGE DISTILLATION
# ============================================================

print_header("KNOWLEDGE DISTILLATION")

print(
    "\nTeacher probabilities will be used as"
)

print(
    "soft targets for the student model."
)

print(
    "\nSoft-target examples:"
)


for i in range(
    min(10, len(df))
):

    print(
        f"Sample {i + 1:02d} : "
        f"{teacher_probability[i]:.4f}"
    )


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print_header("TRAIN / TEST SPLIT")

# ------------------------------------------------------------
# Deterministic split
#
# The dataset is very small (80 samples), so we use the first
# 80% for training and the remaining 20% for testing.
# ------------------------------------------------------------

split_index = int(
    0.8 * len(df)
)

X_train = X.iloc[
    :split_index
].copy()

X_test = X.iloc[
    split_index:
].copy()

y_train = y_true[
    :split_index
]

y_test = y_true[
    split_index:
]

soft_train = teacher_probability[
    :split_index
]

soft_test = teacher_probability[
    split_index:
]


print(
    f"\nTraining samples : {len(X_train)}"
)

print(
    f"Testing samples  : {len(X_test)}"
)


# ============================================================
# STUDENT MODEL
# ============================================================

print_header("TRAINING STUDENT MODEL")

print(
    "\nStudent architecture:"
)

print(
    "MLPRegressor"
)

print(
    "Hidden layer: 16 neurons"
)

print(
    "Hidden layer: 8 neurons"
)

print(
    "Output: teacher detection probability"
)


# ------------------------------------------------------------
# Student predicts the teacher's probability.
#
# This is the distillation target.
# ------------------------------------------------------------

student_model = MLPRegressor(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="lbfgs",
    alpha=0.001,
    max_iter=2000,
    random_state=42
)


print(
    "\nTraining student..."
)

student_model.fit(
    X_train,
    soft_train
)

print(
    "Student training complete."
)


# ============================================================
# STUDENT PREDICTIONS
# ============================================================

print_header("STUDENT PREDICTIONS")


student_probability = (
    student_model
    .predict(X_test)
)


# ------------------------------------------------------------
# Keep probability inside [0, 1]
# ------------------------------------------------------------

student_probability = np.clip(
    student_probability,
    0.0,
    1.0
)


student_prediction = (
    student_probability >= 0.5
).astype(int)


# ============================================================
# STUDENT PERFORMANCE
# ============================================================

student_accuracy = accuracy_score(
    y_test,
    student_prediction
)

student_precision = precision_score(
    y_test,
    student_prediction,
    zero_division=0
)

student_recall = recall_score(
    y_test,
    student_prediction,
    zero_division=0
)

student_f1 = f1_score(
    y_test,
    student_prediction,
    zero_division=0
)


# ============================================================
# DISTILLATION ERROR
# ============================================================

mae = mean_absolute_error(
    soft_test,
    student_probability
)

mse = mean_squared_error(
    soft_test,
    student_probability
)

rmse = np.sqrt(
    mse
)


# ============================================================
# DISPLAY STUDENT RESULTS
# ============================================================

print(
    "\nStudent predictions:"
)

print(
    "-" * 75
)

print(
    f"{'Sample':<10}"
    f"{'Actual':<10}"
    f"{'Teacher P':<12}"
    f"{'Student P':<12}"
    f"{'Student':<10}"
)

print(
    "-" * 75
)


for i in range(
    len(X_test)
):

    print(
        f"{i + split_index + 1:<10}"
        f"{y_test[i]:<10}"
        f"{soft_test[i]:<12.4f}"
        f"{student_probability[i]:<12.4f}"
        f"{student_prediction[i]:<10}"
    )


# ============================================================
# STUDENT SUMMARY
# ============================================================

print_header("STUDENT MODEL SUMMARY")

print(
    f"\nDataset samples       : {len(df)}"
)

print(
    f"Training samples      : {len(X_train)}"
)

print(
    f"Testing samples       : {len(X_test)}"
)

print()

print(
    f"Student accuracy      : "
    f"{student_accuracy:.2%}"
)

print(
    f"Student precision     : "
    f"{student_precision:.2%}"
)

print(
    f"Student recall        : "
    f"{student_recall:.2%}"
)

print(
    f"Student F1 score      : "
    f"{student_f1:.2%}"
)

print()

print(
    "Distillation error"
)

print(
    f"MAE                   : {mae:.6f}"
)

print(
    f"MSE                   : {mse:.6f}"
)

print(
    f"RMSE                  : {rmse:.6f}"
)


# ============================================================
# TEACHER VS STUDENT
# ============================================================

print_header("TEACHER VS STUDENT")

print(
    f"\n{'Metric':<20}"
    f"{'Teacher':<15}"
    f"{'Student':<15}"
)

print(
    "-" * 50
)

print(
    f"{'Accuracy':<20}"
    f"{teacher_accuracy:<15.2%}"
    f"{student_accuracy:<15.2%}"
)

print(
    f"{'Precision':<20}"
    f"{teacher_precision:<15.2%}"
    f"{student_precision:<15.2%}"
)

print(
    f"{'Recall':<20}"
    f"{teacher_recall:<15.2%}"
    f"{student_recall:<15.2%}"
)

print(
    f"{'F1 Score':<20}"
    f"{teacher_f1:<15.2%}"
    f"{student_f1:<15.2%}"
)


# ============================================================
# SAVE STUDENT MODEL
# ============================================================

print_header("SAVING STUDENT MODEL")

os.makedirs(
    "models",
    exist_ok=True
)


# ------------------------------------------------------------
# Save model
# ------------------------------------------------------------

save_pickle(
    student_model,
    STUDENT_MODEL_PATH
)

print(
    "\nStudent model saved:"
)

print(
    STUDENT_MODEL_PATH
)


# ------------------------------------------------------------
# Save feature metadata
# ------------------------------------------------------------

student_metadata = {
    "features": feature_columns,

    "fault_net_mapping": fault_mapping,

    "model_type": "MLPRegressor",

    "hidden_layer_sizes": (
        16,
        8
    ),

    "distillation_target":
        "teacher_probability",

    "threshold": 0.5
}


save_pickle(
    student_metadata,
    STUDENT_FEATURES_PATH
)

print(
    "\nStudent feature metadata saved:"
)

print(
    STUDENT_FEATURES_PATH
)


# ============================================================
# SAVE DISTILLATION RESULTS
# ============================================================

results = pd.DataFrame(
    {
        "actual": y_test,
        "teacher_probability": soft_test,
        "student_probability":
            student_probability,
        "student_prediction":
            student_prediction
    }
)


results_path = (
    "dataset/"
    "student_distillation_results.csv"
)


results.to_csv(
    results_path,
    index=False
)


print(
    "\nDistillation results saved:"
)

print(
    results_path
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print_header(
    "STUDENT MODEL TRAINING COMPLETE"
)

print(
    "\nTeacher model:"
)

print(
    f"  {TEACHER_MODEL_PATH}"
)

print(
    "\nStudent model:"
)

print(
    f"  {STUDENT_MODEL_PATH}"
)

print(
    "\nStudent feature metadata:"
)

print(
    f"  {STUDENT_FEATURES_PATH}"
)

print(
    "\nDistillation results:"
)

print(
    f"  {results_path}"
)

print(
    "\nKnowledge distillation completed successfully."
)