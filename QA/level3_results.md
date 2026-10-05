# Question A - Level 3: Results

Prediction was committed before running (see `level3_prediction.md`).
Script: `level3_threshold.py`. Test set: 61 patients, 28 with heart disease.

## Threshold sweep (Logistic Regression, seed 3025)

| threshold | TP | FN | FP | TN | accuracy | precision | recall |
|---|---|---|---|---|---|---|---|
| 0.50 | 20 | 8 | 3 | 30 | 0.820 | 0.870 | 0.714 |
| 0.40 | 21 | 7 | 4 | 29 | 0.820 | 0.840 | 0.750 |
| 0.30 | 22 | 6 | 7 | 26 | 0.787 | 0.759 | 0.786 |
| 0.25 | 23 | 5 | 8 | 25 | 0.787 | 0.742 | 0.821 |
| 0.20 | 25 | 3 | 13 | 20 | 0.738 | 0.658 | 0.893 |
| **0.18** | **26** | **2** | **14** | **19** | **0.738** | **0.650** | **0.929** |
| 0.10 | 27 | 1 | 18 | 15 | 0.689 | 0.600 | 0.964 |

## Prediction vs actual

| | Predicted | Actual |
|---|---|---|
| Precision direction | down | down (0.870 -> 0.650) |
| Precision value | ~0.70 | 0.650 |
| Threshold needed | ~0.30 | 0.18 |
| Extra false positives | 5 to 7 | 11 (3 -> 14) |

I got the direction right but underestimated the cost. At 0.30 recall was only 0.786.
The last few sick patients get low probabilities (0.18 to 0.25), so the threshold had to go
much lower than I expected, and healthy patients with similar scores got flagged too.
Going from 0.25 to 0.20 alone added 5 false positives.

## Threshold I would use for a real screening tool

**0.18.** Missing a sick patient (FN) is worse than a false alarm (FP) for screening,
because a false alarm only leads to a follow-up test, while a missed patient goes home
untreated. At 0.18 the model misses 2 of 28 sick patients instead of 8. The cost is
14 false alarms out of 33 healthy patients, which is acceptable for a first-stage screen.

## Why accuracy alone would mislead

- A model that always predicts "no disease" gets 54.1% accuracy and catches nobody.
- My chosen threshold (0.18) has lower accuracy than the default (0.738 vs 0.820),
  but it is clearly the better screening model because it misses 6 fewer sick patients.
- Accuracy treats a missed patient and a false alarm as equally bad. In screening they are not.
