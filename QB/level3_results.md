# Question B - Level 3: Results

Prediction was committed before testing (see `level3_prediction.md`, commit at 11:39).

## Break 1: model file missing

**Test:** renamed `QA/model.joblib`, then started the server.

**Before the fix:** the server did not start at all. The terminal showed:

```
File "...\QB\app.py", line 24, in <module>
    bundle = joblib.load(MODEL_PATH)
FileNotFoundError: [Errno 2] No such file or directory: '...\QA\model.joblib'
```

The model is loaded once at the top of `app.py`, when the server starts, so a missing file
stops the whole app. In the browser the page could not be opened ("This site can't be reached").

**Fix:** wrapped `joblib.load` in `try / except FileNotFoundError`. The server now starts,
prints a warning, and:

| Request | Before | After |
|---|---|---|
| `GET /` (form) | site can't be reached | 200, form opens |
| `GET /health` | - | `{"status": "model missing"}` |
| `POST /predict` | - | 503 "The prediction model is not available right now. Please try again later." |

**Prediction vs actual:** I predicted both that the server would start and that it would fail
to start. It failed to start. I also predicted the form would open and then show "Could not
reach the server", but the form did not open at all, because no server was running.

## Break 2: empty "ST depression" box

**Test:** cleared the oldpeak box and clicked "Check my risk".

**Before the fix:** the app gave a risk result with no error. The form converts every box with
`Number(v)`, and in JavaScript `Number("")` is `0`. Oldpeak 0 is inside the valid range (0 to 7),
so my Level 2 validation accepted it and the model predicted as if the patient had no ST depression.

Same patient (50, male, chest pain type 3), only oldpeak changed:

| oldpeak sent | risk | level shown |
|---|---|---|
| 3.5 (real value) | 0.335 | Moderate risk - "see a doctor" |
| empty -> 0 | 0.097 | **Low risk** - "no strong signs" |

The empty box silently turned a patient who should see a doctor into "Low risk", and the wrong
result was also saved in the database.

**Fix:**
1. Front end: an empty box is sent as `null` instead of `Number("")`.
2. Back end: a `null` value gets the message "oldpeak is required." (status 422), and nothing is saved.

**After the fix:** `422 {"detail": ["oldpeak is required."]}`, and `/stats` still shows 0 requests.

**Prediction vs actual:** I predicted the result would be correct because validation would stop
bad input. This was wrong. Validation only checks that a value is in range; it cannot tell that
0 came from an empty box. The lesson is that the front end must not invent values, and the API
must treat missing data as missing.

## Making the app safe for 100 users at once

1. **More workers:** run `uvicorn app:app --workers 4` (no `--reload`) so several requests are
   handled in parallel. The model is small and loaded once per worker, so memory is not a problem.
2. **Database:** SQLite allows only one writer at a time. For 100 users I would switch to
   PostgreSQL with a connection pool, or at least enable SQLite WAL mode
   (`PRAGMA journal_mode=WAL`) so reads do not block writes.
3. **Rate limiting:** limit requests per user (for example with `slowapi`) so one client cannot
   flood the server.
4. **Load testing:** use a tool like `locust` to simulate 100 users and check response time
   and error rate before going live.
5. **Monitoring:** the `/health` endpoint lets a load balancer or monitoring tool check that the
   model is loaded and stop sending traffic to a broken instance.
