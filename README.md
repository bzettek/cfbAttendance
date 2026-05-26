# CFB Attendance Predictor

A machine learning web app that estimates college football game attendance based on five inputs: stadium capacity, historical fill rate, team record, and game-day precipitation.

## How it works

1. **Model** — A `RandomForestRegressor` (400 trees) is trained on ~6,700 historical game records. Raw inputs are expanded into engineered features (win %, expected attendance, rain pressure, etc.) before fitting. Holdout R² > 0.999, MAE ~341 seats.

2. **API** — A Flask app (`app.py`) loads the trained model and exposes:
   - `GET /api/teams` — autocomplete list of all teams
   - `GET /api/team?name=...` — fuzzy team lookup returning capacity and fill rate defaults
   - `POST /api/predict` — accepts `[capacity, fill, wins, losses, prcp]` and returns a predicted attendance count

3. **Post-processing** — Before returning a prediction, the API applies light domain rules: a rain dampening factor (up to −12%), a win/loss record multiplier (up to ±30–45%), and a hard cap at stadium capacity.

4. **Frontend** — A responsive single-page UI (`templates/index.html`) lets users type a team name, auto-fills known capacity/fill data, and calls the prediction API.

## Running locally

```bash
pip install -r requirements.txt
flask --app app.py --debug run
```

Open [http://localhost:5000](http://localhost:5000).

## Retraining

```bash
python train_new_model.py
```

Reads `merged_attendance_dataset.csv`, writes `new_regressor_model.joblib` and `model_metrics.json`.

## Key files

| File | Purpose |
|---|---|
| `app.py` | Flask API + team lookup |
| `train_new_model.py` | Model training script |
| `model_utils.py` | Feature engineering shared between training and serving |
| `notebooks/model_walkthrough.ipynb` | Detailed analysis and documentation |
| `merged_attendance_dataset.csv` | Training data |
| `new_regressor_model.joblib` | Serialized model pipeline |
