import io, base64
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for
from sklearn.linear_model import LinearRegression
from datetime import timedelta

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

app = Flask(__name__)

# ---------- Data Load & Prep ----------
df = pd.read_csv("weather_clean.csv")
df["last_updated"] = pd.to_datetime(df["last_updated"], errors="coerce")

# Defensive: derive parts (safe even if already exist)
df["date"]       = df["last_updated"].dt.date
df["dayofweek"]  = df["last_updated"].dt.dayofweek
df["hour"]       = df["last_updated"].dt.hour
df["month"]      = df["last_updated"].dt.month
df["year_month"] = df["last_updated"].dt.to_period("M").astype(str)

# Pick available numeric columns
POSSIBLE_NUM = [
    "temperature_celsius","humidity","wind_kph","pressure_mb","precip_mm","uv_index",
    "air_quality_PM2.5","air_quality_PM10"
]
NUM_COLS = [c for c in POSSIBLE_NUM if c in df.columns]
DEFAULT_METRIC = "temperature_celsius" if "temperature_celsius" in df.columns else (NUM_COLS[0] if NUM_COLS else None)


def build_city_forecast(df_city, horizon=3):
    """
    Trains two simple LinearRegression models (temperature & humidity)
    with lag/rolling features, then produces 'horizon' days of recursive forecasts.
    Returns: future_df (DataFrame), temp_img (base64), hum_img (base64)
    """
    if df_city.empty:
        return None, None, None

    city = df_city["location_name"].iloc[0] if "location_name" in df_city.columns else "City"

    c = df_city.sort_values("last_updated").copy()
    c["dayofweek"] = c["last_updated"].dt.dayofweek

    # build lags
    for lag in [1, 2, 3]:
        if "temperature_celsius" in c.columns:
            c[f"temp_lag{lag}"] = c["temperature_celsius"].shift(lag)
        if "humidity" in c.columns:
            c[f"hum_lag{lag}"]  = c["humidity"].shift(lag)

    # rolling (use only past values; shift to avoid leakage)
    if "temperature_celsius" in c.columns:
        c["temp_roll3"] = c["temperature_celsius"].shift(1).rolling(3).mean()
    if "humidity" in c.columns:
        c["hum_roll3"]  = c["humidity"].shift(1).rolling(3).mean()

    c = c.dropna().reset_index(drop=True)
    if len(c) < 10:
        # not enough history to train
        return None, None, None

    # Feature sets
    temp_feats = ["dayofweek","temp_lag1","temp_lag2","temp_lag3","temp_roll3"]
    hum_feats  = ["dayofweek","hum_lag1","hum_lag2","hum_lag3","hum_roll3"]

    # Train models on all available history (demo-style)
    model_temp = None
    model_hum  = None
    if all(f in c.columns for f in temp_feats) and "temperature_celsius" in c.columns:
        model_temp = LinearRegression().fit(c[temp_feats], c["temperature_celsius"])
    if all(f in c.columns for f in hum_feats) and "humidity" in c.columns:
        model_hum  = LinearRegression().fit(c[hum_feats],  c["humidity"])

    # Seed state with last row
    last = c.iloc[-1]
    base_date = last["last_updated"]

    # lag state
    t1 = last.get("temp_lag1"); t2 = last.get("temp_lag2"); t3 = last.get("temp_lag3")
    h1 = last.get("hum_lag1");  h2 = last.get("hum_lag2");  h3 = last.get("hum_lag3")

    future = []
    for step in range(1, horizon+1):
        next_date = base_date + timedelta(days=step)
        dow = next_date.weekday()

        # compute roll3 from lags
        temp_roll3 = None if t1 is None else float(np.mean([t for t in [t1,t2,t3] if t is not None]))
        hum_roll3  = None if h1 is None else float(np.mean([h for h in [h1,h2,h3] if h is not None]))

        # build feature rows
        pred_temp = None
        if model_temp is not None and temp_roll3 is not None:
            feats_temp = [[dow, t1, t2, t3, temp_roll3]]
            pred_temp = float(model_temp.predict(feats_temp)[0])

        pred_hum = None
        if model_hum is not None and hum_roll3 is not None:
            feats_hum  = [[dow, h1, h2, h3, hum_roll3]]
            pred_hum   = float(model_hum.predict(feats_hum)[0])

        future.append({
            "city": city,
            "date": next_date.date(),
            "pred_temperature": None if pred_temp is None else round(pred_temp, 2),
            "pred_humidity":    None if pred_hum  is None else round(pred_hum, 2),
        })

        # shift lags (recursive)
        if pred_temp is not None:
            t3, t2, t1 = t2, t1, pred_temp
        if pred_hum is not None:
            h3, h2, h1 = h2, h1, pred_hum

    future_df = pd.DataFrame(future)

    # Build quick visuals: last 14 days + forecast
    temp_img = hum_img = None
    # Temperature chart
    if "temperature_celsius" in c.columns and future_df["pred_temperature"].notna().any():
        hist = c[["last_updated","temperature_celsius"]].tail(14).rename(
            columns={"last_updated":"date","temperature_celsius":"Temperature (°C)"})
        pred = future_df[["date","pred_temperature"]].rename(columns={"pred_temperature":"Temperature (°C)"})
        fig, ax = plt.subplots(figsize=(11,4))
        ax.plot(hist["date"], hist["Temperature (°C)"], label="History", color="steelblue")
        ax.plot(pred["date"], pred["Temperature (°C)"], "o--", label="Forecast", color="tomato")
        ax.set_title(f"{city}: Temperature — last 14 days + {len(future_df)}-day forecast")
        ax.set_ylabel("°C"); ax.grid(True, alpha=0.3); ax.legend()
        temp_img = fig_to_base64(fig)

    # Humidity chart
    if "humidity" in c.columns and future_df["pred_humidity"].notna().any():
        hist = c[["last_updated","humidity"]].tail(14).rename(
            columns={"last_updated":"date","humidity":"Humidity (%)"})
        pred = future_df[["date","pred_humidity"]].rename(columns={"pred_humidity":"Humidity (%)"})
        fig, ax = plt.subplots(figsize=(11,4))
        ax.plot(hist["date"], hist["Humidity (%)"], label="History", color="seagreen")
        ax.plot(pred["date"], pred["Humidity (%)"], "o--", label="Forecast", color="darkorange")
        ax.set_title(f"{city}: Humidity — last 14 days + {len(future_df)}-day forecast")
        ax.set_ylabel("%"); ax.grid(True, alpha=0.3); ax.legend()
        hum_img = fig_to_base64(fig)

    return future_df, temp_img, hum_img


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img

def get_city():
    # Default to city with max rows
    if "location_name" not in df.columns or df["location_name"].empty:
        return None
    return df["location_name"].value_counts().idxmax()

# ---------- Routes ----------
@app.route("/", methods=["GET", "POST"])
def index():
    cities = sorted(df["location_name"].unique()) if "location_name" in df.columns else []
    city = request.values.get("city") or (cities[0] if cities else None)
    metric = request.values.get("metric") or DEFAULT_METRIC

    # Quick overview chart for home
    imgs = []
    title = "Welcome"
    if city and metric and metric in df.columns:
        sub = df[df["location_name"] == city].copy()
        daily = (sub
                 .set_index("last_updated")
                 .resample("D")[[metric]]
                 .mean()
                 .reset_index())
        fig, ax = plt.subplots(figsize=(11,4))
        ax.plot(daily["last_updated"], daily[metric], color="tomato")
        ax.set_title(f"{city}: Daily {metric.replace('_',' ').title()}")
        ax.set_xlabel("Date"); ax.set_ylabel(metric.replace("_"," ").title())
        ax.grid(True, alpha=0.3)
        imgs.append(fig_to_base64(fig))
        title = f"Dashboard — {city}"

    return render_template("index.html",
                           title=title,
                           cities=cities,
                           selected_city=city,
                           metric=metric,
                           images=imgs)

# ---------- Distributions ----------
@app.route("/distributions")
def distributions():
    city = request.args.get("city") or get_city()
    cols = NUM_COLS
    if not cols:
        return render_template("plot.html", title="Distributions", images=[], note="No numeric columns found.")
    sub = df[df["location_name"] == city] if city and "location_name" in df.columns else df

    n = len(cols)
    ncols = 2                        # fewer per row = more space
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4*nrows), squeeze=False)

    for i, col in enumerate(cols):
        ax = axes[i // ncols, i % ncols]
        sns.histplot(data=sub, x=col, kde=True, ax=ax, color="teal")
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col.replace("_"," ").title())
        ax.set_ylabel("Count")

    # turn off empty slots if any
    for j in range(i+1, nrows*ncols):
        axes[j // ncols, j % ncols].axis("off")

    plt.tight_layout(pad=3.0)        # <-- adds spacing between subplots
    img = fig_to_base64(fig)

    return render_template("plot.html", title=f"Distributions — {city}", images=[img])


# ---------- Correlation Heatmap ----------
@app.route("/correlation")
def correlation():
    city = request.args.get("city") or get_city()
    cols = NUM_COLS
    if len(cols) < 2:
        return render_template("plot.html", title="Correlation Heatmap", images=[], note="Need ≥ 2 numeric columns.")
    sub = df[df["location_name"] == city] if city and "location_name" in df.columns else df
    corr = sub[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True, ax=ax)
    ax.set_title(f"Correlation Heatmap — {city}")
    img = fig_to_base64(fig)
    return render_template("plot.html", title=f"Correlation — {city}", images=[img])

# ---------- Time-Series Trend (Top Cities) ----------
@app.route("/trend")
def trend():
    metric = request.args.get("metric") or DEFAULT_METRIC
    if not metric or metric not in df.columns:
        return render_template("plot.html", title="Trends", images=[], note="Metric not found.")
    if "location_name" not in df.columns:
        return render_template("plot.html", title="Trends", images=[], note="No location_name column.")

    top_cities = df["location_name"].value_counts().head(4).index.tolist()
    sub = df[df["location_name"].isin(top_cities)].copy()
    daily = (sub
             .set_index("last_updated")
             .groupby("location_name")
             .resample("D")[[metric]]
             .mean()
             .reset_index())

    fig, ax = plt.subplots(figsize=(12,5))
    for city in top_cities:
        d = daily[daily["location_name"] == city]
        ax.plot(d["last_updated"], d[metric], marker="o", label=city)
    ax.set_title(f"Daily {metric.replace('_',' ').title()} — Top Cities")
    ax.set_xlabel("Date"); ax.set_ylabel(metric.replace("_"," ").title())
    ax.grid(True, alpha=0.3); ax.legend(title="City")
    img = fig_to_base64(fig)
    return render_template("plot.html", title=f"Trend — {metric}", images=[img])

# ---------- Weekday Boxplot ----------
@app.route("/weekday")
def weekday():
    city = request.args.get("city") or get_city()
    metric = request.args.get("metric") or DEFAULT_METRIC
    if not metric or metric not in df.columns:
        return render_template("plot.html", title="Weekday Pattern", images=[], note="Metric not found.")
    sub = df[df["location_name"] == city] if city and "location_name" in df.columns else df
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(data=sub, x="dayofweek", y=metric, palette="Set2", ax=ax)
    ax.set_title(f"{metric.replace('_',' ').title()} by Day of Week — {city} (0=Mon)")
    ax.set_xlabel("Day of Week"); ax.set_ylabel(metric.replace("_"," ").title())
    img = fig_to_base64(fig)
    return render_template("plot.html", title=f"Weekday — {city}", images=[img])


@app.route("/predict", methods=["GET"])
def predict():
    # inputs via querystring: ?city=Bikaner&horizon=3
    city = request.args.get("city") or get_city()
    try:
        horizon = int(request.args.get("horizon", 3))
    except:
        horizon = 3
    horizon = max(1, min(horizon, 14))  # cap to 14 for safety

    if city is None or "location_name" not in df.columns:
        return render_template("predict.html", title="Predictions", table_html=None, images=[], note="No city data found.")

    city_df = df[df["location_name"] == city].copy()
    future_df, temp_img, hum_img = build_city_forecast(city_df, horizon=horizon)

    if future_df is None:
        return render_template("predict.html", title="Predictions", table_html=None, images=[], note="Not enough history to forecast.")

    # build a small HTML table to display predictions
    show_cols = ["city","date"]
    if "pred_temperature" in future_df.columns:
        show_cols.append("pred_temperature")
    if "pred_humidity" in future_df.columns:
        show_cols.append("pred_humidity")

    table_html = future_df[show_cols].to_html(index=False, classes="table", border=0)

    imgs = [img for img in [temp_img, hum_img] if img is not None]
    return render_template("predict.html",
                           title=f"Predictions — {city} (next {horizon} days)",
                           table_html=table_html,
                           images=imgs,
                           note=None)

# ---------- Hourly Pattern ----------
@app.route("/hourly")
def hourly():
    city = request.args.get("city") or get_city()
    metric = request.args.get("metric") or DEFAULT_METRIC
    if not metric or metric not in df.columns:
        return render_template("plot.html", title="Hourly Pattern", images=[], note="Metric not found.")
    sub = df[df["location_name"] == city] if city and "location_name" in df.columns else df
    if sub["hour"].nunique() <= 1:
        return render_template("plot.html", title="Hourly Pattern", images=[], note="No intra-day data found.")
    hourly_mean = sub.groupby("hour", as_index=False)[metric].mean()
    fig, ax = plt.subplots(figsize=(10,4))
    sns.lineplot(data=hourly_mean, x="hour", y=metric, marker="o", color="crimson", ax=ax)
    ax.set_title(f"Average {metric.replace('_',' ').title()} by Hour — {city}")
    ax.set_xlabel("Hour"); ax.set_ylabel(metric.replace("_"," ").title())
    ax.grid(True, alpha=0.3); ax.set_xticks(range(0,24,2))
    img = fig_to_base64(fig)
    return render_template("plot.html", title=f"Hourly — {city}", images=[img])

# ---------- Monthly Heatmap (Year–Month) ----------
@app.route("/heatmap")
def heatmap():
    metric = request.args.get("metric") or DEFAULT_METRIC
    if not metric or metric not in df.columns:
        return render_template("plot.html", title="Monthly Heatmap", images=[], note="Metric not found.")
    if "location_name" not in df.columns:
        return render_template("plot.html", title="Monthly Heatmap", images=[], note="No location_name column.")

    monthly_ym = df.groupby(["location_name","year_month"], as_index=False)[metric].mean()
    pivot_ym = monthly_ym.pivot(index="location_name", columns="year_month", values=metric).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(14, max(4, 0.4*len(pivot_ym))))
    sns.heatmap(pivot_ym, cmap="YlOrRd", cbar_kws={"label": metric.replace("_"," ").title()}, ax=ax)
    ax.set_title(f"Monthly Avg {metric.replace('_',' ').title()} by City (Year–Month)")
    ax.set_xlabel("Year–Month"); ax.set_ylabel("City")
    img = fig_to_base64(fig)
    return render_template("plot.html", title="Monthly Heatmap", images=[img])

# ---------- Scatter Temp vs Humidity ----------
@app.route("/scatter")
def scatter():
    if not all(c in df.columns for c in ["temperature_celsius","humidity"]):
        return render_template("plot.html", title="Scatter", images=[], note="Need temperature_celsius and humidity.")
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(data=sample, x="temperature_celsius", y="humidity", alpha=0.5, ax=ax)
    ax.set_title("Temperature vs Humidity")
    ax.set_xlabel("Temperature (°C)"); ax.set_ylabel("Humidity (%)")
    img = fig_to_base64(fig)
    return render_template("plot.html", title="Scatter — Temp vs Humidity", images=[img])

# ---------- Geo Scatter (optional) ----------
@app.route("/geo")
def geo():
    if not all(c in df.columns for c in ["latitude","longitude"]):
        return render_template("plot.html", title="Geo", images=[], note="No latitude/longitude columns found.")
    sample = df.sample(min(5000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(6,7))
    hue = "temperature_celsius" if "temperature_celsius" in df.columns else None
    sns.scatterplot(data=sample, x="longitude", y="latitude", hue=hue,
                    palette="coolwarm" if hue else None, alpha=0.6, edgecolor="none", ax=ax)
    ax.set_title("Geospatial Spread" + (" — colored by Temperature" if hue else ""))
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    img = fig_to_base64(fig)
    return render_template("plot.html", title="Geo", images=[img])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

