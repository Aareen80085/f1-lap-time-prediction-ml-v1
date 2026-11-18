# F1 Lap Time Prediction Using Machine Learning

A beginner-friendly machine learning project that predicts Formula 1 lap time in seconds from lap, tyre, speed-trap, and weather information.

## Project type

- Supervised machine learning
- Regression
- Domain: Formula 1 sports analytics
- Target variable: `LapTimeSeconds`

## Dataset

- **Name:** 2026 Australian Grand Prix Practice 1 Laps CSV
- **Source:** [TracingInsights Raw F1 Data](https://app.tracinginsights.com/analysis/download-raw-data/)
- **Raw size:** 487 rows and 49 columns
- **One row represents:** one driver lap

The project uses the following input features:

| Feature | Meaning |
| --- | --- |
| `LapNumber` | Number of the lap |
| `Stint` | Current tyre stint number |
| `TyreLife` | Laps completed on the current tyre set |
| `Compound` | Tyre compound, such as Soft, Medium, or Hard |
| `SpeedI1` | Speed at the first intermediate timing point |
| `SpeedI2` | Speed at the second intermediate timing point |
| `SpeedST` | Speed at the main speed trap |
| `AirTemp` | Air temperature |
| `TrackTemp` | Track temperature |
| `Humidity` | Air humidity |
| `WindSpeed` | Wind speed |

## Models used

The project trains and compares three regression models:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Random Forest Regressor was selected as the best model for this run because it achieved the lowest Mean Absolute Error (MAE) on the held-out test data.

## Workflow

1. Load the dataset
2. Check and understand the dataset
3. Split the data into training and testing sets
4. Clean and prepare the data
5. Explore the data with graphs
6. Train three regression models
7. Compare the model results
8. Use the best model to predict lap time
9. Write the conclusion

## Data preprocessing

- Checked data types, missing values, and duplicates
- Removed rows with missing or invalid `LapTimeSeconds`
- Removed duplicate records
- Converted `Compound` into numeric columns using one-hot encoding
- Scaled numeric features using `StandardScaler`
- Used an 80/20 train-test split with `random_state=42`

## Evaluation metrics

This is a regression project, so it uses:

- MAE
- MSE
- RMSE
- R² score

Classification metrics such as accuracy, recall, F1-score, and the confusion matrix are not used because the project predicts a continuous numeric lap time.

## Exploratory data analysis

The notebook includes:

- Histogram of lap times
- Count plot of tyre compounds
- Bar chart of average lap time by tyre compound
- Box plot of lap times by compound
- Scatter plot of tyre life versus lap time
- Correlation heatmap
- Pair plot
- Line chart of median lap time by lap number

## Run the notebook in Google Colab

1. Upload `F1_Lap_Time_Prediction_Simple.ipynb` to Google Colab.
2. Run the cells in order.
3. When prompted, upload the downloaded TracingInsights Laps CSV.
4. Run the model comparison cell to see the evaluation metrics and selected model.

## Run the optional Streamlit dashboard locally

Create and activate a virtual environment first:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies and start the dashboard:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Open the local URL printed by Streamlit and upload the same Laps CSV.

## Project files

```text
F1_Lap_Time_Prediction_Simple.ipynb  # Main Google Colab notebook
streamlit_app.py                     # Optional interactive dashboard
requirements.txt                     # Python dependencies
F1_Lap_Time_Prediction_Submission_Report.docx  # Project report
```

## Limitations

- The dataset is from one Practice 1 session.
- Some laps may be incomplete, in-laps, or out-laps and do not have valid lap times.
- The model does not include all race factors, such as fuel load, traffic, strategy, or detailed car setup.
- Results should not be generalised to all circuits and races without further testing.

## Future improvements

- Train on data from multiple races and seasons
- Add driver, team, circuit, and fuel-related features
- Add detailed telemetry summaries
- Tune model hyperparameters
- Deploy the model as an interactive application

## References

- [TracingInsights Raw F1 Data](https://app.tracinginsights.com/analysis/download-raw-data/)
- [scikit-learn documentation](https://scikit-learn.org/)
- [Pandas documentation](https://pandas.pydata.org/docs/)
