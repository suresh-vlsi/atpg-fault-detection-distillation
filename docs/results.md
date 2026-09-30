# Experimental Results

## Dataset

The ATPG fault dataset was generated using stuck-at fault simulation.

| Parameter | Value |
|---|---:|
| Total samples | 80 |
| Features | 7 |
| Fault types | SA0 / SA1 |
| Circuit type | Combinational logic |

---

# Teacher Model Performance

The teacher model is trained using the generated ATPG fault dataset.

| Metric | Value |
|---|---:|
| Accuracy | 100% |
| Precision | 100% |
| Recall | 100% |
| F1 Score | 100% |

The teacher model acts as the high-accuracy reference model.

---

# Student Model Performance

The student model is trained using knowledge transferred from the teacher.

| Metric | Value |
|---|---:|
| Accuracy | 75% |
| Precision | 40% |
| Recall | 66.67% |
| F1 Score | 50% |

---

# Knowledge Distillation

The student model learns from:

- Teacher predictions
- Soft probability outputs
- ATPG fault features

The distilled model provides a compact alternative for fault detection.

---

# Generated Outputs
