# Methodology

## 1. Fault Dataset Generation

A combinational circuit is analyzed using ATPG concepts.
The D-algorithm is used to generate test patterns for stuck-at faults.

Supported faults:

- Stuck-at-0 (SA0)
- Stuck-at-1 (SA1)

---

## 2. Fault Simulation

Generated test patterns are simulated on:

- Good circuit
- Faulty circuit

The output difference determines fault detection.

---

## 3. Teacher Model

A high-capacity machine learning model learns:

- Circuit inputs
- Fault location
- Fault type
- Faulty output behavior

The teacher model provides:

- Hard labels
- Soft probability outputs

---

## 4. Knowledge Distillation

The student model learns from the teacher model.

Loss:

\[
L = \alpha L_{CE} + (1-\alpha)L_{KD}
\]

where:

- \(L_{CE}\) = classification loss
- \(L_{KD}\) = knowledge distillation loss

---

## 5. Student Model

The compact model performs:

- Fault detection
- Fault classification
- Efficient inference

This reduces computational complexity while preserving accuracy.