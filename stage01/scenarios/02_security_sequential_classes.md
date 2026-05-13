# Scenario 2: Security — sequence of threat classes (C1→C2→C3→C4)

## Objective

Transfer the CL analysis to **classification**: sequentially introduce new threat / anomaly classes and demonstrate forgetting vs retention under fine-tuning vs EWC / replay.

## Description

- **Task:** Classification of events / telemetry: normal vs type_1, type_2, type_3 (e.g. sensor damage, spoofed command, temperature anomaly).
- **Sequence:** the model first learns C1 (e.g. normal + threat_1), then adds C2 without retraining on C1, then C3, C4. In each phase, training is on the new class only (plus, optionally, replay from earlier classes).

## Variants to compare

| Variant | Description | Expectation |
|---------|-------------|-------------|
| **Fine-tune** | After each phase, fine-tune on the new class | Drop in accuracy / F1 on C1, C2 after adding further classes |
| **EWC** | Fisher after each phase; regularization in the loss | Better retention on earlier classes |
| **Replay** | Buffer of samples (obs, label) from earlier classes; mixed into training | Retention depends on buffer size |
| **EWC+replay** | Hybrid | Best retention at a constrained budget |
| **Joint** | Training on all classes from the start | Upper bound as a reference |

## Metrics

- **Accuracy / macro F1** on every class after every phase (matrix: phase × class).
- **Confusion matrix** after the last phase (fine-tune vs the best CL method).

## What it demonstrates (CL strengths and weaknesses)

- **Strength:** In the supervised domain CL also limits forgetting; the same scheme (EWC, replay) works in RL and in classification — consistency of the analysis.
- **Weakness:** Requires storing replay (labels must be correct) or a good Fisher estimate; with many classes the buffer size grows.
