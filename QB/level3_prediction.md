# Question B - Level 3: Prediction

## Break 1: the model file is missing

I rename `QA/model.joblib` and start the server with `python -m uvicorn app:app`.

What I think will happen:
The server starts normally and the error only appears when someone clicks the button and The server fails to start and shows an error in the terminal

What the user will see in the browser:
The page initially opens normally and displays the form, but after clicking the Predict button, an error appears saying “Could not reach the server. Is it running?”

## Break 2: a field in the form is left empty

I clear the "ST depression (oldpeak)" box in the form and click "Check my risk".

What I think will happen:
When I leave the field empty, the app shows an error asking me to fill it in. When I enter invalid input, it still gives a risk result.

Will the result be correct? Why:
Yes,the result will be correct, because the validation will stop any bad or missing input before it reaches the model, so no wrong prediction will be made.
