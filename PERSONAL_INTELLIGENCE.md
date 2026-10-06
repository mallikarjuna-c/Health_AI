# Personal Intelligence Note

Seed S = 3025. All numbers below are from my own runs.

## Decision log

### Question A - Predict a health risk

**Decision 1: Logistic Regression as the final model, not Random Forest.**
Rejected: Random Forest. On my test set LR got accuracy 0.820 and recall 0.714, while RF got
0.754 and 0.607. With only 303 patients, the simpler model generalised better, and LR also
gives weights I can explain (top features: `ca`, `cp_4`, `slope_2`).

**Decision 2: Screening threshold 0.18 instead of the default 0.5.**
Rejected: 0.5. At 0.5 the model missed 8 of 28 sick patients. At 0.18 it misses only 2
(recall 0.929), at the cost of 14 false alarms instead of 3. For screening, a missed patient is
worse than a follow-up test, so I accept lower precision (0.650) and lower accuracy (0.738).

### Question B - Turn the model into an app

**Decision 1: Serve the exact model from Question A instead of retraining in the app folder.**
Rejected: a separate training script for the app on all 303 rows. The brief asks to serve a
model from Question A, and using the same model means the app's behaviour is described by the
numbers I measured in Question A (82% accuracy). The 0.18 threshold from Question A Level 3 is
used as the "Moderate risk - see a doctor" cut-off.

**Decision 2: Keep the server running when the model file is missing.**
Rejected: letting the app crash at startup. In my Break 1 test the server did not start at all
and the user only saw "This site can't be reached". After the fix the page still opens,
`/health` reports "model missing" and `/predict` returns a clear 503 message.

## AI usage declaration

**Tools used:** Claude (Anthropic), in the Claude Code desktop app.

**What for:** reading the assignment brief, planning the steps, writing a first version of the
code for each level, explaining concepts (gradient of the log loss, L2 penalty, `joblib`,
`uvicorn`), and drafting the README and results write-ups. I ran every script myself, edited the
code and comments, wrote the Level 3 predictions myself and committed them before running the
tests, and reviewed the write-ups against my own numbers.

**Where the AI was wrong or weak, and how I fixed it:**

1. **Front-end bug found in my own Level 3 test.** The AI-written form converted every box with
   `Number(v)`. In JavaScript `Number("")` is `0`, so an empty ST depression box was silently sent
   as 0, passed validation and changed a patient from Moderate risk (0.335) to Low risk (0.097).
   I found this when I broke the app on purpose. Fixed by sending `null` for empty boxes and
   returning "oldpeak is required." from the API.
2. **Hard-coded relative path.** The first version used `"QA/heart_disease.csv"`, which crashed
   with FileNotFoundError when I ran the script from inside the `QA` folder. Fixed by building the
   path from the script's own location.
3. **Command that did not work on my machine.** `uvicorn app:app --reload` was not recognised in
   PowerShell because the Scripts folder is not on my PATH. I used `python -m uvicorn` instead.
