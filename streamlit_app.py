"""Interactive F1 lap-time prediction dashboard.

Run locally with: streamlit run streamlit_app.py
"""

import os
from pathlib import Path

# Speed up matplotlib import by using local font cache
_mpl_dir = Path(__file__).resolve().parent / ".matplotlib"
_mpl_dir.mkdir(exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(_mpl_dir)

import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Authentic F1 Pirelli Tyre Color Palette
COMPOUND_COLOR_SCALE = alt.Scale(
    domain=["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"],
    range=["#ff1801", "#ffd100", "#ffffff", "#39b54a", "#00a0de"]
)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


st.set_page_config(page_title="F1 Lap Time Studio", page_icon="🏎️", layout="wide", initial_sidebar_state="expanded")
sns.set_theme(style="darkgrid")
plt.rcParams.update({"figure.facecolor": "#10131a", "axes.facecolor": "#10131a", "axes.labelcolor": "#e8edf5", "xtick.color": "#e8edf5", "ytick.color": "#e8edf5", "text.color": "#e8edf5"})

st.markdown("""
<style>
    .stApp { background: radial-gradient(circle at 15% 0%, #262b36 0%, #10131a 42%, #07090d 100%); color: #eef2f7; }
    [data-testid="stSidebar"] { background: #0b0e14; border-right: 1px solid #303744; }
    .block-container { padding-top: 1.5rem; }
    h1, h2, h3 { color: #f5f7fa !important; letter-spacing: .025em; }
    h1 { border-left: 7px solid #e10600; padding-left: .7rem; }
    [data-testid="stMetric"] { background: #171c25; border-top: 3px solid #e10600; border-radius: 4px; padding: .8rem 1rem; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #eef2f7 !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { background: #1a202b; border-radius: 4px 4px 0 0; color: #d4d9e2; padding: .6rem 1rem; }
    .stTabs [aria-selected="true"] { background: #e10600; color: white; }
    div[data-testid="stDataFrame"] { border: 1px solid #343b49; }
</style>
""", unsafe_allow_html=True)


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename common Laps CSV columns to one simple, consistent format and remove duplicate columns."""
    # Remove any existing duplicate column names in the raw CSV
    frame = frame.loc[:, ~frame.columns.duplicated(keep="first")].copy()

    names = {
        "LapNumber": "lap_number",
        "Stint": "stint",
        "TyreLife": "tyre_life",
        "Compound": "compound",
        "Position": "position",
        "TrackStatus": "track_status",
        "LapTimeSeconds": "lap_time_seconds",
        "Driver": "driver",
        "Team": "team",
        "SpeedI1": "speed_i1",
        "SpeedI2": "speed_i2",
        "SpeedST": "speed_st",
        "AirTemp": "air_temp",
        "TrackTemp": "track_temp",
        "Humidity": "humidity",
        "WindSpeed": "wind_speed",
    }
    renamed = frame.rename(columns=names).copy()
    # Remove duplicate columns again in case renaming created a collision (e.g. 'Team' and 'team')
    renamed = renamed.loc[:, ~renamed.columns.duplicated(keep="first")].copy()
    return renamed


def get_lap_time_seconds(series: pd.Series) -> pd.Series:
    """Handle both numeric seconds and time strings such as 1:24.500."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.8:
        return numeric
    return pd.to_timedelta(series.astype(str), errors="coerce").dt.total_seconds()


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    data = normalise_columns(raw)
    if "lap_time_seconds" not in data.columns:
        if "LapTime_in_seconds" in data.columns:
            data["lap_time_seconds"] = get_lap_time_seconds(data["LapTime_in_seconds"])
        elif "LapTime" in data.columns:
            data["lap_time_seconds"] = get_lap_time_seconds(data["LapTime"])
        elif "lap_time" in data.columns:
            data["lap_time_seconds"] = get_lap_time_seconds(data["lap_time"])
        else:
            raise ValueError("No lap-time column was found. Expected lap_time_seconds, LapTimeSeconds, LapTime, or lap_time.")

    feature_columns = [
        column
        for column in ["lap_number", "stint", "tyre_life", "compound", "speed_i1", "speed_i2", "speed_st", "air_temp", "track_temp", "humidity", "wind_speed"]
        if column in data.columns
    ]
    if not feature_columns:
        raise ValueError("None of the expected lap-level feature columns were found.")

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["lap_time_seconds"]).copy()
    data = data[data["lap_time_seconds"] > 0].copy()
    # Ensure all column names are completely unique before handing to pyarrow/streamlit
    data = data.loc[:, ~data.columns.duplicated(keep="first")].copy()
    data = data.drop_duplicates().copy()
    return data, feature_columns


@st.cache_resource
def train_models(data: pd.DataFrame, features: list[str]):
    """Train the three required regression models and return metrics and the best pipeline."""
    X = data[features]
    y = data["lap_time_seconds"]

    # Same simple classroom split used in the notebook.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    numeric_features = X_train.select_dtypes(include="number").columns.tolist()
    categorical_features = [column for column in features if column not in numeric_features]

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numbers",
                Pipeline([
                    ("fill_missing", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categories",
                Pipeline([
                    ("fill_missing", SimpleImputer(strategy="most_frequent")),
                    ("encode", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers)
    algorithms = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
    }

    results, trained = [], {}
    for name, algorithm in algorithms.items():
        pipeline = Pipeline([("prepare_data", preprocessor), ("algorithm", algorithm)])
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, predicted)
        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_test, predicted),
            "MSE": mse,
            "RMSE": math.sqrt(mse),
            "R²": r2_score(y_test, predicted),
        })
        trained[name] = pipeline

    comparison = pd.DataFrame(results).sort_values("MAE").reset_index(drop=True)
    best_name = comparison.loc[0, "Model"]
    return comparison, trained[best_name], best_name, X_train, X_test, y_test, trained[best_name].predict(X_test)


st.title("F1 LAP TIME COMMAND CENTRE")
st.caption("Australian Grand Prix • Practice 1 • Lap-time prediction and race-engineer dashboard")

with st.sidebar:
    st.header("Data Source")
    data_source = st.radio(
        "Choose telemetry source:",
        ["🏎️ Demo Dataset (Australian GP FP1)", "📁 Upload my own CSV"],
        index=0
    )

sample_csv_path = Path(__file__).resolve().parent / "sample_f1_laps.csv"

if data_source == "📁 Upload my own CSV":
    with st.sidebar:
        uploaded_file = st.file_uploader("Upload Laps CSV", type="csv")
        st.caption("Use a Laps CSV downloaded from TracingInsights.")
    if uploaded_file is None:
        st.info("👈 Please upload a Laps CSV in the sidebar to begin, or switch to the Demo Dataset.")
        st.stop()
    raw_df = pd.read_csv(uploaded_file)
else:
    if not sample_csv_path.exists():
        st.error(f"Sample data file not found at {sample_csv_path}")
        st.stop()
    raw_df = pd.read_csv(sample_csv_path)
    with st.sidebar:
        st.success("✅ Loaded 2026 Australian GP Practice 1 telemetry!")
df, features = prepare_data(raw_df)

comparison, best_model, best_name, X_train, X_test, y_test, test_predictions = train_models(df, features)

with st.sidebar:
    st.header("Exploration filters")
    filtered_df = df.copy()
    for column, label in [("driver", "Driver"), ("compound", "Tyre compound"), ("stint", "Stint")]:
        if column in filtered_df.columns:
            values = sorted(filtered_df[column].dropna().astype(str).unique())
            chosen = st.multiselect(label, values, default=values)
            filtered_df = filtered_df[filtered_df[column].astype(str).isin(chosen)]

    if "lap_number" in filtered_df.columns:
        maximum = int(filtered_df["lap_number"].max())
        selected_limit = st.slider("Maximum lap number", 1, maximum, maximum)
        filtered_df = filtered_df[filtered_df["lap_number"] <= selected_limit]

metric_columns = st.columns(4)
metric_columns[0].metric("Lap records", f"{len(filtered_df):,}")
metric_columns[1].metric("Average lap time", f"{filtered_df['lap_time_seconds'].mean():.2f} s")
metric_columns[2].metric("Fastest lap", f"{filtered_df['lap_time_seconds'].min():.2f} s")
metric_columns[3].metric("Best model", best_name)

eda_tab, models_tab, prediction_tab = st.tabs(["📊 Paddock data wall & graphs", "📈 Model garage & accuracy", "🔴 Race engineer prediction"])

with eda_tab:
    st.subheader("🏎️ Interactive Telemetry & Race Visualisations")
    if filtered_df.empty:
        st.warning("No records match the selected filters.")
    else:
        # Row 1: Distribution & Tyre Count
        row1_left, row1_right = st.columns(2)
        with row1_left:
            st.markdown("##### ⏱️ Lap Time Distribution")
            hist_counts, bin_edges = np.histogram(filtered_df["lap_time_seconds"].dropna(), bins=20)
            bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}s" for i in range(len(hist_counts))]
            dist_df = pd.DataFrame({"Lap Time Interval": bin_labels, "Lap Count": hist_counts})
            st.bar_chart(dist_df, x="Lap Time Interval", y="Lap Count", color="#00d2be")

        with row1_right:
            st.markdown("##### 🍩 Laps Completed by Tyre Compound")
            if "compound" in filtered_df.columns:
                counts_df = filtered_df["compound"].value_counts().reset_index()
                counts_df.columns = ["Compound", "Laps Completed"]
                counts_chart = (
                    alt.Chart(counts_df)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("Compound:N", title="Tyre Compound", sort=None),
                        y=alt.Y("Laps Completed:Q", title="Laps Completed"),
                        color=alt.Color("Compound:N", scale=COMPOUND_COLOR_SCALE, legend=None),
                        tooltip=["Compound", "Laps Completed"]
                    )
                )
                st.altair_chart(counts_chart, use_container_width=True)
            else:
                st.info("No tyre-compound data available.")

        # Row 2: Average Time by Compound & Tyre Degradation
        row2_left, row2_right = st.columns(2)
        with row2_left:
            st.markdown("##### ⚡ Average Lap Time by Tyre Compound")
            if "compound" in filtered_df.columns:
                avg_compound = filtered_df.groupby("compound")["lap_time_seconds"].mean().round(2).reset_index()
                avg_compound.columns = ["Compound", "Average Lap Time (s)"]
                avg_chart = (
                    alt.Chart(avg_compound)
                    .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                    .encode(
                        x=alt.X("Compound:N", title="Tyre Compound", sort=None),
                        y=alt.Y("Average Lap Time (s):Q", title="Average Lap Time (s)", scale=alt.Scale(zero=False)),
                        color=alt.Color("Compound:N", scale=COMPOUND_COLOR_SCALE, legend=None),
                        tooltip=["Compound", "Average Lap Time (s)"]
                    )
                )
                st.altair_chart(avg_chart, use_container_width=True)
            else:
                st.info("No tyre-compound data available.")

        with row2_right:
            st.markdown("##### 📉 Tyre Degradation (Tyre Age vs Lap Time)")
            if "tyre_life" in filtered_df.columns:
                tooltips = [c for c in ["driver", "compound", "tyre_life", "lap_number", "lap_time_seconds"] if c in filtered_df.columns]
                deg_chart = (
                    alt.Chart(filtered_df)
                    .mark_circle(size=90, opacity=0.85)
                    .encode(
                        x=alt.X("tyre_life:Q", title="Tyre Age (laps)"),
                        y=alt.Y("lap_time_seconds:Q", title="Lap Time (seconds)", scale=alt.Scale(zero=False)),
                        color=alt.Color("compound:N", scale=COMPOUND_COLOR_SCALE, title="Compound"),
                        tooltip=tooltips
                    )
                    .interactive()
                )
                st.altair_chart(deg_chart, use_container_width=True)
            else:
                st.info("No tyre-life data available.")

        # Row 3: Session Pace Progression
        if "lap_number" in filtered_df.columns:
            st.markdown("##### 🏁 Session Pace Progression (Median Pace across Session)")
            trend = filtered_df.groupby("lap_number")["lap_time_seconds"].median().round(2).reset_index()
            trend.columns = ["Lap Number", "Median Lap Time (s)"]
            st.line_chart(trend, x="Lap Number", y="Median Lap Time (s)", color="#e10600")

        # Row 4: Telemetry Correlation Matrix
        numeric = filtered_df.select_dtypes(include="number")
        if numeric.shape[1] >= 2:
            st.markdown("##### 🌡️ Telemetry Correlation Matrix")
            st.caption("Values close to +1.0 indicate features that increase together; -1.0 indicates inverse relationships.")
            st.dataframe(
                numeric.corr().style.background_gradient(cmap="coolwarm", axis=None).format("{:.2f}"),
                use_container_width=True
            )

        st.subheader("📋 Filtered Session Data")
        clean_display_df = filtered_df.loc[:, ~filtered_df.columns.duplicated(keep="first")]
        st.dataframe(clean_display_df, use_container_width=True, height=240)

with models_tab:
    st.subheader("Machine Learning Model Leaderboard")
    st.dataframe(comparison.style.format({"MAE": "{:.3f}", "MSE": "{:.3f}", "RMSE": "{:.3f}", "R²": "{:.3f}"}), use_container_width=True)
    st.success(f"🏆 Winning Model: **{best_name}** (Lowest Mean Absolute Error: ±{comparison.loc[0, 'MAE']:.2f} seconds)")

    st.markdown(f"##### 🎯 Actual vs Predicted Lap Times ({best_name})")
    pred_df = pd.DataFrame({
        "Actual Lap Time (s)": np.round(y_test, 2),
        "Predicted Lap Time (s)": np.round(test_predictions, 2)
    })
    st.scatter_chart(
        pred_df,
        x="Actual Lap Time (s)",
        y="Predicted Lap Time (s)",
        color="#00d2be"
    )

with prediction_tab:
    st.subheader("Race Engineer: Lap Time Simulator")
    st.caption(f"Predict lap time using the trained **{best_name}** model based on live session conditions.")
    values = {}
    form_columns = st.columns(2)
    for index, column in enumerate(X_train.columns):
        with form_columns[index % 2]:
            if pd.api.types.is_numeric_dtype(X_train[column]):
                lower = float(X_train[column].quantile(.05))
                upper = float(X_train[column].quantile(.95))
                upper = upper if upper > lower else lower + 1
                values[column] = st.slider(column.replace("_", " ").title(), lower, upper, float(X_train[column].median()))
            else:
                options = sorted(X_train[column].dropna().astype(str).unique())
                values[column] = st.selectbox(column.replace("_", " ").title(), options)

    if st.button("🏁 Predict Lap Time", type="primary"):
        input_df = pd.DataFrame([values], columns=X_train.columns)
        prediction = float(best_model.predict(input_df)[0])
        st.metric("Predicted Lap Time", f"{prediction:.2f} s")
        st.info("Estimated based on machine learning weights fitted to the session telemetry data.")
