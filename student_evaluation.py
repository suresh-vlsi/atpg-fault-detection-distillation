# ============================================================
# student_evaluation.py
# ============================================================
# ATPG FAULT DETECTION - STUDENT MODEL EVALUATION
#
# Purpose:
#   1. Load the generated ATPG fault-detection dataset
#   2. Load the trained teacher model
#   3. Load the trained student model
#   4. Reconstruct the exact features required by the models
#   5. Encode fault_net consistently
#   6. Evaluate teacher and student predictions
#   7. Compare teacher vs student
#   8. Calculate classification metrics
#   9. Calculate RMSE
#  10. Save detailed evaluation results
#
# Dataset:
#   dataset/fault_detection_dataset.csv
#
# Models:
#   models/teacher_model.pkl
#   models/student_model.pkl
#
# Metadata:
#   models/teacher_features.pkl
#   models/student_features.pkl
#
# Output:
#   dataset/student_evaluation_results.csv
# ============================================================


import os
import sys
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = os.path.join(
    "dataset",
    "fault_detection_dataset.csv"
)

TEACHER_MODEL_PATH = os.path.join(
    "models",
    "teacher_model.pkl"
)

TEACHER_FEATURES_PATH = os.path.join(
    "models",
    "teacher_features.pkl"
)

STUDENT_MODEL_PATH = os.path.join(
    "models",
    "student_model.pkl"
)

STUDENT_FEATURES_PATH = os.path.join(
    "models",
    "student_features.pkl"
)

OUTPUT_PATH = os.path.join(
    "dataset",
    "student_evaluation_results.csv"
)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def print_header(title):
    """
    Print a consistent section header.
    """

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def file_check(path, description):
    """
    Check whether a required file exists.
    """

    if not os.path.exists(path):
        print()
        print("ERROR: Required file not found.")
        print(f"File       : {path}")
        print(f"Description: {description}")
        print()

        raise FileNotFoundError(path)


# ============================================================
# LOAD DATASET
# ============================================================

print_header("ATPG FAULT DETECTION - STUDENT MODEL EVALUATION")

print()
print("Loading dataset...")
print()

file_check(
    DATASET_PATH,
    "ATPG fault-detection dataset"
)

df = pd.read_csv(DATASET_PATH)


print("Dataset shape:")
print(df.shape)

print()
print("Columns:")
print(list(df.columns))


# ============================================================
# DATASET INFORMATION
# ============================================================

print_header("DATASET INFORMATION")

if "detected" not in df.columns:
    raise ValueError(
        "Dataset does not contain required target column 'detected'."
    )


total_samples = len(df)

detected_samples = int(
    df["detected"].sum()
)

undetected_samples = (
    total_samples - detected_samples
)

detection_rate = (
    detected_samples / total_samples
    if total_samples > 0
    else 0.0
)


print()
print(f"Total samples     : {total_samples}")
print(f"Detected samples  : {detected_samples}")
print(f"Undetected samples: {undetected_samples}")
print(
    f"Detection rate    : {detection_rate * 100:.2f}%"
)


# ============================================================
# LOAD TEACHER MODEL
# ============================================================

print_header("LOADING TEACHER MODEL")

file_check(
    TEACHER_MODEL_PATH,
    "trained teacher model"
)

file_check(
    TEACHER_FEATURES_PATH,
    "teacher feature metadata"
)

try:

    teacher_model = joblib.load(
        TEACHER_MODEL_PATH
    )

    teacher_features_metadata = joblib.load(
        TEACHER_FEATURES_PATH
    )

except Exception as exc:

    print()
    print("ERROR while loading teacher model:")
    print(exc)

    raise


print()
print("Teacher model loaded:")
print(TEACHER_MODEL_PATH)

print()
print("Teacher feature metadata loaded:")
print(TEACHER_FEATURES_PATH)


# ============================================================
# LOAD STUDENT MODEL
# ============================================================

print_header("LOADING STUDENT MODEL")

file_check(
    STUDENT_MODEL_PATH,
    "trained student model"
)

file_check(
    STUDENT_FEATURES_PATH,
    "student feature metadata"
)

try:

    student_model = joblib.load(
        STUDENT_MODEL_PATH
    )

    student_features_metadata = joblib.load(
        STUDENT_FEATURES_PATH
    )

except Exception as exc:

    print()
    print("ERROR while loading student model:")
    print(exc)

    raise


print()
print("Student model loaded:")
print(STUDENT_MODEL_PATH)

print()
print("Student feature metadata loaded:")
print(STUDENT_FEATURES_PATH)


# ============================================================
# METADATA EXTRACTION
# ============================================================

def extract_feature_list(metadata):
    """
    Extract feature names from different possible
    metadata formats.

    Supported examples:

        {
            "features": [...]
        }

        {
            "feature_names": [...]
        }

        {
            "input_features": [...]
        }

        [...]
    """

    if isinstance(metadata, dict):

        possible_keys = [
            "features",
            "feature_names",
            "input_features",
            "feature_list",
            "columns",
        ]

        for key in possible_keys:

            if key in metadata:

                value = metadata[key]

                if isinstance(value, (list, tuple)):
                    return list(value)

    elif isinstance(metadata, (list, tuple)):

        return list(metadata)

    return None


def extract_fault_mapping(metadata):
    """
    Extract the fault-net encoding mapping.

    The mapping may be stored under different names.
    """

    if not isinstance(metadata, dict):
        return None

    possible_keys = [
        "fault_net_mapping",
        "fault_mapping",
        "fault_net_encoder",
        "label_mapping",
        "encoder",
    ]

    for key in possible_keys:

        if key in metadata:

            return metadata[key]

    return None


teacher_feature_names = extract_feature_list(
    teacher_features_metadata
)

student_feature_names = extract_feature_list(
    student_features_metadata
)


if teacher_feature_names is None:

    print()
    print(
        "WARNING: Could not extract teacher feature names."
    )

    teacher_feature_names = [
        "A",
        "B",
        "C",
        "fault_net_encoded",
        "stuck_value",
        "good_output",
        "faulty_output",
    ]


if student_feature_names is None:

    print()
    print(
        "WARNING: Could not extract student feature names."
    )

    student_feature_names = [
        "A",
        "B",
        "C",
        "fault_net_encoded",
        "stuck_value",
        "good_output",
        "faulty_output",
    ]


# ============================================================
# FEATURE INFORMATION
# ============================================================

print_header("FEATURE INFORMATION")

print()
print("Features used by teacher:")

for index, feature in enumerate(
    teacher_feature_names,
    start=1
):

    print(
        f"{index:02d}. {feature}"
    )


print()
print("Features used by student:")

for index, feature in enumerate(
    student_feature_names,
    start=1
):

    print(
        f"{index:02d}. {feature}"
    )


# ============================================================
# FAULT NET ENCODING
# ============================================================

print_header("FAULT NET ENCODING")


def create_fault_net_encoding(
    dataframe,
    mapping=None
):
    """
    Create fault_net_encoded.

    Priority:

        1. Existing valid mapping from metadata
        2. Deterministic mapping from sorted unique values

    Returns:

        dataframe
        mapping
    """

    data = dataframe.copy()

    if "fault_net" not in data.columns:

        raise ValueError(
            "Dataset does not contain 'fault_net' column."
        )


    # --------------------------------------------------------
    # Case 1: Mapping is a dictionary
    # --------------------------------------------------------

    if isinstance(mapping, dict):

        # Determine whether mapping is:
        #
        #   {"A": 0, "B": 1}
        #
        # or:
        #
        #   {0: "A", 1: "B"}

        mapping_keys = list(mapping.keys())

        if (
            len(mapping_keys) > 0
            and all(
                isinstance(k, str)
                for k in mapping_keys
            )
        ):

            forward_mapping = mapping

        else:

            forward_mapping = {
                str(value): int(key)
                for key, value in mapping.items()
            }


        data["fault_net_encoded"] = (
            data["fault_net"]
            .astype(str)
            .map(forward_mapping)
        )


        # Unknown fault names
        unknown_mask = (
            data["fault_net_encoded"]
            .isna()
        )

        if unknown_mask.any():

            print()
            print(
                "WARNING: Unknown fault_net values "
                "were found."
            )

            unknown_values = sorted(
                data.loc[
                    unknown_mask,
                    "fault_net"
                ]
                .astype(str)
                .unique()
                .tolist()
            )

            print(
                "Unknown values:",
                unknown_values
            )

            # Assign deterministic new IDs
            current_values = list(
                forward_mapping.values()
            )

            next_id = (
                max(current_values) + 1
                if current_values
                else 0
            )

            for value in unknown_values:

                forward_mapping[value] = next_id
                next_id += 1

            data["fault_net_encoded"] = (
                data["fault_net"]
                .astype(str)
                .map(forward_mapping)
            )


        data["fault_net_encoded"] = (
            data["fault_net_encoded"]
            .astype(int)
        )

        return data, forward_mapping


    # --------------------------------------------------------
    # Case 2: sklearn LabelEncoder
    # --------------------------------------------------------

    if hasattr(mapping, "transform"):

        values = (
            data["fault_net"]
            .astype(str)
            .values
        )

        try:

            encoded = mapping.transform(
                values
            )

            data["fault_net_encoded"] = (
                encoded.astype(int)
            )

            return data, mapping

        except Exception:

            print(
                "WARNING: Saved fault encoder "
                "could not encode all values."
            )


    # --------------------------------------------------------
    # Case 3: Create deterministic mapping
    # --------------------------------------------------------

    unique_fault_nets = sorted(
        data["fault_net"]
        .astype(str)
        .unique()
        .tolist()
    )

    forward_mapping = {
        name: index
        for index, name in enumerate(
            unique_fault_nets
        )
    }

    data["fault_net_encoded"] = (
        data["fault_net"]
        .astype(str)
        .map(forward_mapping)
        .astype(int)
    )

    return data, forward_mapping


# ------------------------------------------------------------
# Obtain mapping
# ------------------------------------------------------------

student_fault_mapping = extract_fault_mapping(
    student_features_metadata
)

teacher_fault_mapping = extract_fault_mapping(
    teacher_features_metadata
)


# Prefer student mapping because the student model is
# the model being evaluated.

if student_fault_mapping is not None:

    df, active_fault_mapping = (
        create_fault_net_encoding(
            df,
            student_fault_mapping
        )
    )

elif teacher_fault_mapping is not None:

    df, active_fault_mapping = (
        create_fault_net_encoding(
            df,
            teacher_fault_mapping
        )
    )

else:

    print()
    print(
        "No saved fault-net mapping found."
    )

    print(
        "Creating deterministic mapping "
        "from the dataset..."
    )

    df, active_fault_mapping = (
        create_fault_net_encoding(
            df,
            None
        )
    )


print()
print("Fault-net encoding:")

for name, value in active_fault_mapping.items():

    print(
        f"  {name} -> {value}"
    )


# ============================================================
# FEATURE VALIDATION
# ============================================================

print_header("FEATURE VALIDATION")


all_required_features = sorted(
    set(
        teacher_feature_names
        + student_feature_names
    )
)


missing_features = [
    feature
    for feature in all_required_features
    if feature not in df.columns
]


if missing_features:

    print()
    print(
        "ERROR: Missing features in dataset:"
    )

    print(
        missing_features
    )

    print()
    print("Available columns:")
    print(
        list(df.columns)
    )

    raise ValueError(
        f"Missing features in dataset: "
        f"{missing_features}"
    )


print()
print("All required features are available.")

print()
print("Available feature columns:")

for feature in all_required_features:

    print(
        f"  OK  {feature}"
    )


# ============================================================
# PREPARE TARGET
# ============================================================

print_header("TARGET PREPARATION")


y = (
    pd.to_numeric(
        df["detected"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)


print()
print("Target column: detected")

print(
    f"Detected = 1 : {int((y == 1).sum())}"
)

print(
    f"Detected = 0 : {int((y == 0).sum())}"
)


# ============================================================
# PREPARE TEACHER FEATURES
# ============================================================

X_teacher = df[
    teacher_feature_names
].copy()


# ============================================================
# PREPARE STUDENT FEATURES
# ============================================================

X_student = df[
    student_feature_names
].copy()


# ============================================================
# NUMERIC CONVERSION
# ============================================================

def convert_features_to_numeric(
    dataframe,
    feature_names
):
    """
    Convert all model input features to numeric values.
    """

    result = dataframe.copy()

    for feature in feature_names:

        result[feature] = pd.to_numeric(
            result[feature],
            errors="coerce"
        )

    if result.isna().any().any():

        bad_columns = (
            result.columns[
                result.isna().any()
            ]
            .tolist()
        )

        print()
        print(
            "ERROR: Non-numeric or missing "
            "feature values detected."
        )

        print(
            "Problem columns:",
            bad_columns
        )

        raise ValueError(
            "Feature matrix contains NaN values."
        )

    return result


X_teacher = convert_features_to_numeric(
    X_teacher,
    teacher_feature_names
)

X_student = convert_features_to_numeric(
    X_student,
    student_feature_names
)


print()
print(
    "Teacher feature matrix:",
    X_teacher.shape
)

print(
    "Student feature matrix:",
    X_student.shape
)


# ============================================================
# MODEL PREDICTION FUNCTION
# ============================================================

def get_model_predictions(
    model,
    X
):
    """
    Obtain:

        hard predictions
        detection probabilities

    from an sklearn-style model.
    """

    # --------------------------------------------------------
    # Hard prediction
    # --------------------------------------------------------

    predictions = model.predict(X)

    predictions = np.asarray(
        predictions
    ).reshape(-1)

    predictions = (
        predictions
        .astype(int)
    )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = None


    if hasattr(model, "predict_proba"):

        try:

            proba = model.predict_proba(X)

            proba = np.asarray(proba)


            if proba.ndim == 2:

                # Binary classifier:
                #
                # column 0 = P(class 0)
                # column 1 = P(class 1)

                if proba.shape[1] >= 2:

                    probabilities = (
                        proba[:, 1]
                    )

                else:

                    probabilities = (
                        proba[:, 0]
                    )

            else:

                probabilities = proba.reshape(-1)


        except Exception:

            probabilities = None


    # --------------------------------------------------------
    # If probability unavailable
    # --------------------------------------------------------

    if probabilities is None:

        probabilities = (
            predictions
            .astype(float)
        )


    probabilities = np.clip(
        probabilities,
        0.0,
        1.0
    )

    return (
        predictions,
        probabilities
    )


# ============================================================
# TEACHER PREDICTION
# ============================================================

print_header("TEACHER PREDICTION")

try:

    teacher_predictions, teacher_probabilities = (
        get_model_predictions(
            teacher_model,
            X_teacher
        )
    )

except Exception as exc:

    print()
    print(
        "ERROR during teacher prediction:"
    )

    print(exc)

    raise


print()
print(
    "Teacher predictions generated:",
    len(teacher_predictions)
)


# ============================================================
# STUDENT PREDICTION
# ============================================================

print_header("STUDENT PREDICTION")

try:

    student_predictions, student_probabilities = (
        get_model_predictions(
            student_model,
            X_student
        )
    )

except Exception as exc:

    print()
    print(
        "ERROR during student prediction:"
    )

    print(exc)

    raise


print()
print(
    "Student predictions generated:",
    len(student_predictions)
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    probabilities
):
    """
    Calculate classification and regression-style
    probability error metrics.
    """

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            probabilities
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "rmse": rmse,
    }


# ============================================================
# CALCULATE TEACHER METRICS
# ============================================================

teacher_metrics = calculate_metrics(
    y,
    teacher_predictions,
    teacher_probabilities
)


# ============================================================
# CALCULATE STUDENT METRICS
# ============================================================

student_metrics = calculate_metrics(
    y,
    student_predictions,
    student_probabilities
)


# ============================================================
# PRINT TEACHER RESULTS
# ============================================================

print_header("TEACHER MODEL RESULTS")

print()
print(
    f"Accuracy  : "
    f"{teacher_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Precision : "
    f"{teacher_metrics['precision'] * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{teacher_metrics['recall'] * 100:.2f}%"
)

print(
    f"F1 Score  : "
    f"{teacher_metrics['f1'] * 100:.2f}%"
)

print(
    f"RMSE      : "
    f"{teacher_metrics['rmse']:.6f}"
)


# ============================================================
# PRINT STUDENT RESULTS
# ============================================================

print_header("STUDENT MODEL RESULTS")

print()
print(
    f"Accuracy  : "
    f"{student_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Precision : "
    f"{student_metrics['precision'] * 100:.2f}%"
)

print(
    f"Recall    : "
    f"{student_metrics['recall'] * 100:.2f}%"
)

print(
    f"F1 Score  : "
    f"{student_metrics['f1'] * 100:.2f}%"
)

print(
    f"RMSE      : "
    f"{student_metrics['rmse']:.6f}"
)


# ============================================================
# TEACHER VS STUDENT
# ============================================================

print_header("TEACHER VS STUDENT")

print()

print(
    f"{'Metric':<15}"
    f"{'Teacher':>15}"
    f"{'Student':>15}"
)

print(
    "-" * 45
)

print(
    f"{'Accuracy':<15}"
    f"{teacher_metrics['accuracy'] * 100:>14.2f}%"
    f"{student_metrics['accuracy'] * 100:>14.2f}%"
)

print(
    f"{'Precision':<15}"
    f"{teacher_metrics['precision'] * 100:>14.2f}%"
    f"{student_metrics['precision'] * 100:>14.2f}%"
)

print(
    f"{'Recall':<15}"
    f"{teacher_metrics['recall'] * 100:>14.2f}%"
    f"{student_metrics['recall'] * 100:>14.2f}%"
)

print(
    f"{'F1 Score':<15}"
    f"{teacher_metrics['f1'] * 100:>14.2f}%"
    f"{student_metrics['f1'] * 100:>14.2f}%"
)

print(
    f"{'RMSE':<15}"
    f"{teacher_metrics['rmse']:>15.6f}"
    f"{student_metrics['rmse']:>15.6f}"
)


# ============================================================
# STUDENT-TEACHER AGREEMENT
# ============================================================

print_header("TEACHER-STUDENT AGREEMENT")

agreement = np.mean(
    teacher_predictions
    == student_predictions
)

disagreement = 1.0 - agreement


print()
print(
    f"Prediction agreement : "
    f"{agreement * 100:.2f}%"
)

print(
    f"Prediction disagreement: "
    f"{disagreement * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRICES
# ============================================================

print_header("TEACHER CONFUSION MATRIX")

teacher_cm = confusion_matrix(
    y,
    teacher_predictions,
    labels=[0, 1]
)

print()
print(
    "                 Predicted"
)

print(
    "                 0       1"
)

print(
    f"Actual 0       "
    f"{teacher_cm[0, 0]:>5}   "
    f"{teacher_cm[0, 1]:>5}"
)

print(
    f"Actual 1       "
    f"{teacher_cm[1, 0]:>5}   "
    f"{teacher_cm[1, 1]:>5}"
)


print_header("STUDENT CONFUSION MATRIX")

student_cm = confusion_matrix(
    y,
    student_predictions,
    labels=[0, 1]
)

print()
print(
    "                 Predicted"
)

print(
    "                 0       1"
)

print(
    f"Actual 0       "
    f"{student_cm[0, 0]:>5}   "
    f"{student_cm[0, 1]:>5}"
)

print(
    f"Actual 1       "
    f"{student_cm[1, 0]:>5}   "
    f"{student_cm[1, 1]:>5}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print_header("STUDENT CLASSIFICATION REPORT")

print()

print(
    classification_report(
        y,
        student_predictions,
        target_names=[
            "Undetected",
            "Detected"
        ],
        zero_division=0
    )
)


# ============================================================
# SAMPLE PREDICTIONS
# ============================================================

print_header("SAMPLE PREDICTIONS")

print()

sample_count = min(
    10,
    len(df)
)

print(
    f"{'Sample':<10}"
    f"{'Actual':<10}"
    f"{'Teacher':<10}"
    f"{'Student':<10}"
    f"{'P(Student)':<15}"
    f"{'Agreement':<12}"
)

print(
    "-" * 67
)


for i in range(sample_count):

    actual = int(
        y.iloc[i]
    )

    teacher_pred = int(
        teacher_predictions[i]
    )

    student_pred = int(
        student_predictions[i]
    )

    student_probability = float(
        student_probabilities[i]
    )

    same = (
        "YES"
        if teacher_pred == student_pred
        else "NO"
    )

    print(
        f"{i + 1:<10}"
        f"{actual:<10}"
        f"{teacher_pred:<10}"
        f"{student_pred:<10}"
        f"{student_probability:<15.4f}"
        f"{same:<12}"
    )


# ============================================================
# CREATE DETAILED RESULTS DATAFRAME
# ============================================================

print_header("CREATING EVALUATION RESULTS")

results = df.copy()


results["teacher_probability"] = (
    teacher_probabilities
)

results["teacher_prediction"] = (
    teacher_predictions
)

results["student_probability"] = (
    student_probabilities
)

results["student_prediction"] = (
    student_predictions
)

results["teacher_correct"] = (
    results["teacher_prediction"]
    == results["detected"]
)

results["student_correct"] = (
    results["student_prediction"]
    == results["detected"]
)

results["teacher_student_agree"] = (
    results["teacher_prediction"]
    == results["student_prediction"]
)


# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

results.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print(
    "Evaluation results saved:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print_header("FINAL EVALUATION SUMMARY")

print()

print(
    f"Dataset samples      : {len(df)}"
)

print(
    f"Teacher accuracy     : "
    f"{teacher_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Student accuracy     : "
    f"{student_metrics['accuracy'] * 100:.2f}%"
)

print(
    f"Teacher F1 score     : "
    f"{teacher_metrics['f1'] * 100:.2f}%"
)

print(
    f"Student F1 score     : "
    f"{student_metrics['f1'] * 100:.2f}%"
)

print(
    f"Teacher RMSE         : "
    f"{teacher_metrics['rmse']:.6f}"
)

print(
    f"Student RMSE         : "
    f"{student_metrics['rmse']:.6f}"
)

print(
    f"Teacher-student "
    f"agreement           : "
    f"{agreement * 100:.2f}%"
)

print()
print(
    f"Results file         : "
    f"{OUTPUT_PATH}"
)

print()

print("=" * 70)
print("STUDENT MODEL EVALUATION COMPLETE")
print("=" * 70)
print()