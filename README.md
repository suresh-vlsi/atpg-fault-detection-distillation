# ATPG Fault Detection using Knowledge Distillation

A student-level VLSI CAD / Design-for-Test (DFT) project combining **Automatic Test Pattern Generation (ATPG)**, **fault simulation**, and **knowledge distillation**.

The project uses a teacher model to learn fault-detection behavior and transfers that knowledge to a smaller student model. The student model is evaluated on generated fault-detection samples and demonstrated through a final inference flow.

## Project Overview

Traditional ATPG techniques can become computationally expensive as circuit complexity increases. This project explores a machine-learning-assisted approach:

```text
Circuit / Fault Model
        │
        ▼
   ATPG / Simulation
        │
        ▼
 Fault Detection Dataset
        │
        ▼
   Teacher Model
        │
        │ Knowledge Distillation
        ▼
   Student Model
        │
        ▼
Fault Detection Prediction
```

### Main objectives

- Generate and process fault-detection data.
- Model stuck-at fault behavior.
- Use a teacher neural network as a reference model.
- Distill the teacher's learned representation into a smaller student model.
- Evaluate student predictions against actual fault-detection labels.
- Provide a reproducible Python-based demonstration.
- Maintain generated datasets, trained models, and evaluation results for analysis.

## Repository Structure

```text
atpg-fault-detection-distillation/
│
├── atpg.py
├── circuit.py
├── d_algorithm.py
├── dataset_generator.py
├── fault.py
├── fault_collapsing.py
├── final_demo.py
├── main.py
├── parser.py
├── simulator.py
├── student_evaluation.py
├── student_model.py
├── teacher_model.py
│
├── dataset/
│   ├── fault_detection_dataset.csv
│   ├── final_demo_results.csv
│   ├── student_distillation_results.csv
│   └── student_evaluation_results.csv
│
├── examples/
│   └── simple.bench
│
├── models/
│   ├── student_features.pkl
│   ├── student_model.pkl
│   ├── teacher_features.pkl
│   └── teacher_model.pkl
│
├── tests/
├── .gitignore
└── README.md
```

## Key Components

### Circuit Parser

`parser.py` reads the circuit description and converts the benchmark representation into an internal circuit structure.

Example benchmark:

```text
examples/simple.bench
```

### Circuit Representation

`circuit.py` defines the circuit data structures used by the ATPG and simulation stages.

### Fault Model

`fault.py` provides the fault representation used by the experiments. The project focuses on **stuck-at faults**, including `stuck-at-0` and `stuck-at-1`.

### D-Algorithm

`d_algorithm.py` contains the ATPG-oriented logic used to reason about test-pattern generation and fault detection.

### Fault Collapsing

`fault_collapsing.py` provides fault-reduction functionality so equivalent or redundant fault cases can be reduced before subsequent processing.

### Fault Simulation

`simulator.py` simulates circuit behavior for generated test patterns and determines whether a target fault can be detected.

### Dataset Generation

`dataset_generator.py` creates the machine-learning dataset used by the teacher/student models. Generated data is stored under `dataset/`.

### Teacher Model

`teacher_model.py` implements the reference/teacher model used for learning fault-detection behavior. The trained model and extracted features are stored under `models/`.

### Student Model

`student_model.py` implements the smaller student model used for knowledge distillation.

### Student Evaluation

`student_evaluation.py` evaluates the student model against the available evaluation dataset and records the resulting metrics.

### Final Demonstration

`final_demo.py` provides the final end-to-end demonstration:

```text
Dataset
   ↓
Load student model
   ↓
Inference
   ↓
Fault detection predictions
   ↓
Accuracy / result table
   ↓
CSV output
```

The final results are written to `dataset/final_demo_results.csv`.

# Installation

## Requirements

Recommended environment:

- Python 3.10+
- Windows / Linux / WSL
- Git

Check Python:

```powershell
python --version
```

Check Git:

```powershell
git --version
```

## Clone the Repository

```bash
git clone https://github.com/suresh-vlsi/atpg-fault-detection-distillation.git
cd atpg-fault-detection-distillation
```

## Optional Virtual Environment

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Linux / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
```

# Running the Project

Run the stages as needed:

```bash
python dataset_generator.py
python teacher_model.py
python student_model.py
python student_evaluation.py
python final_demo.py
```

For a complete demonstration, the final command is:

```bash
python final_demo.py
```

The final demo generates:

```text
dataset/final_demo_results.csv
```

# Final Demonstration

The final demonstration verifies that the complete flow is operational.

A successful run reports:

```text
Student model : LOADED
Inference     : SUCCESSFUL
Predictions   : GENERATED
Results CSV   : SAVED
```

The current demonstration produced a student-model accuracy of:

```text
75.00%
```

This value is specific to the current demonstration dataset/model and should not be interpreted as a general benchmark for ATPG fault detection.

## Example Result

A typical result table contains:

```text
sample   actual_detected   student_prediction   student_probability   prediction_label
1        0                  0                    0.0                   FAULT NOT DETECTED
2        0                  0                    0.0                   FAULT NOT DETECTED
...
10       1                  1                    1.0                   FAULT DETECTED
...
```

The complete output is stored in:

```text
dataset/final_demo_results.csv
```

# Knowledge Distillation Concept

The central ML idea is **knowledge distillation**.

The teacher model provides a learned representation of the fault-detection problem. Instead of requiring the student model to learn the complete problem independently, the student is trained to reproduce useful information from the teacher.

```text
                 Training
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
   Ground Truth             Teacher Model
                                │
                                ▼
                         Teacher Knowledge
                                │
                                ▼
                         Student Training
                                │
                                ▼
                         Student Model
```

The resulting student model is intended to provide a compact inference model while retaining useful fault-detection behavior.

# VLSI / DFT Relevance

This project connects machine learning with classical VLSI Design-for-Test concepts.

### Classical ATPG flow

```text
Circuit
  ↓
Fault List
  ↓
Fault Reduction / Collapsing
  ↓
ATPG
  ↓
Test Pattern
  ↓
Fault Simulation
  ↓
Fault Coverage
```

### ML-assisted flow explored in this project

```text
Circuit
  ↓
Fault / ATPG Data
  ↓
Dataset
  ↓
Teacher Model
  ↓
Knowledge Distillation
  ↓
Student Model
  ↓
Fault Detection Prediction
```

Potential applications include:

- ATPG acceleration
- Fault classification
- Test-pattern prioritization
- Fault simulation assistance
- Test coverage prediction
- ML-assisted DFT workflows
- Hardware-aware model compression

# Technologies

```text
Python
Machine Learning
Knowledge Distillation
ATPG
D-Algorithm
Fault Simulation
Stuck-at Fault Modeling
Fault Collapsing
CSV Dataset Processing
Git / GitHub
```

# Output Files

| File | Purpose |
|---|---|
| `dataset/fault_detection_dataset.csv` | Fault-detection dataset |
| `dataset/student_evaluation_results.csv` | Student evaluation results |
| `dataset/student_distillation_results.csv` | Distillation results |
| `dataset/final_demo_results.csv` | Final demonstration predictions |
| `models/teacher_model.pkl` | Saved teacher model |
| `models/teacher_features.pkl` | Teacher feature representation |
| `models/student_model.pkl` | Saved student model |
| `models/student_features.pkl` | Student feature representation |

# Project Status

**Status: Completed functional demonstration**

- [x] Circuit representation
- [x] Fault modeling
- [x] ATPG / D-Algorithm component
- [x] Fault collapsing
- [x] Fault simulation
- [x] Dataset generation
- [x] Teacher model
- [x] Student model
- [x] Knowledge-distillation workflow
- [x] Student evaluation
- [x] Final inference demonstration
- [x] CSV result generation
- [x] Git/GitHub repository setup

# Future Improvements

1. Support larger ISCAS benchmark circuits.
2. Add additional fault models.
3. Implement transition-delay faults.
4. Compare D-Algorithm, PODEM and FAN.
5. Add conventional ATPG fault coverage as a baseline.
6. Compare teacher and student inference time.
7. Measure model size reduction.
8. Add precision, recall, F1-score and confusion matrix.
9. Perform cross-circuit generalization experiments.
10. Explore graph neural networks for circuit representation.
11. Investigate hardware-aware knowledge distillation.
12. Use the student model to prioritize ATPG search.
13. Integrate Verilog simulation into the ML pipeline.
14. Evaluate scalability on larger benchmark circuits.

# Research Direction

A stronger research-oriented version of this project can investigate:

> **Knowledge Distillation for ML-Assisted ATPG and Fault Detection**

The student model can eventually be used not only as a binary fault detector, but as an intelligent component inside the ATPG loop to prioritize promising test patterns or faults.

```text
             RTL / Gate-Level Netlist
                       │
                       ▼
                 Fault Generation
                       │
                       ▼
                 Fault Collapsing
                       │
                       ▼
                 ATPG Candidate Set
                       │
                       ▼
              ┌──────────────────┐
              │ Student ML Model │
              └──────────────────┘
                       │
             Fault / Pattern Score
                       │
                       ▼
                ATPG Prioritization
                       │
                       ▼
                Fault Simulation
                       │
                       ▼
                 Coverage Update
                       │
                       └──────► Iterate
```

This provides a bridge between **classical VLSI CAD algorithms** and **machine-learning-assisted EDA**.

# Author

**Suresh Kumar**

M.Tech — Systems & Control Engineering, IIT Bombay

Interests:

- VLSI Design
- Design-for-Test
- ATPG
- Formal Verification
- RTL Design
- Digital VLSI
- Machine Learning for EDA
- Hardware-Aware AI

GitHub: https://github.com/suresh-vlsi

Project: https://github.com/suresh-vlsi/atpg-fault-detection-distillation

# License

This project is intended for academic, educational, and research purposes.
