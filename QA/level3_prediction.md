# Question A - Level 3: Prediction

Model: Logistic Regression, seed 3025

At the default threshold of 0.5 my model gets precision 0.870 and recall 0.714.
On the 61 test patients: TP = 20, FN = 8, FP = 3, TN = 30.

## What happens to precision if I lower the threshold until recall reaches 0.9?

I think precision will go **down**, to around **0.70**.

Lowering the decision threshold will make the model classify more patients as positive.
This increases the number of true positives but also increases the number of false
positives, which reduces precision.

## Threshold needed

Around **0.30**.

## Extra false alarms

I expect **5 to 7** more false positives than now.
