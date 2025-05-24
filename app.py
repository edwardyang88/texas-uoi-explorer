import streamlit as st
import pandas as pd
import json
import plotly.express as px

# ─── page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Texas UOI Explorer",
    layout="wide"
)
st.title("🔍 Texas Urban Opportunity Index (UOI) Explorer")

# ─── load your data & geojson ────────────────────────────────────────
# ensure your CSV has a 3-digit county FIPS column named 'fips'
df = pd.read_csv("texas_counties_full.csv", dtype={"fips": str})

# prefix "48" (Texas) onto every county FIPS, zero-pad to 5 chars
df["fips"] = ("48" + df["fips"].str.zfill(3)).str.zfill(5)
df = df[df["County"].str.upper() != "LOVING"]
with open("geojson-counties-fips.json") as f:
    counties = json.load(f)

# keep only Texas (STATE FIPS == "48")
counties["features"] = [
    feat for feat in counties["features"]
    if feat["properties"]["STATE"] == "48"
]

# Optionally assign feature.id so px knows to match on `locations="fips"`
for feat in counties["features"]:
    feat["id"] = feat["properties"]["GEO_ID"] if "GEO_ID" in feat["properties"] else feat["properties"]["GEOID"]

# ─── recompute Z-scores for each indicator ────────────────────────────
raw = [
    "Median_Household_Income",
    "Bachelors_Degree_Pct",
    "No_Vehicle_Pct",
    "No_Health_Insurance_Pct"
]
for var in raw:
    zcol = f"Z_{var}"
    vals = df[var]
    if var == "No_Health_Insurance_Pct":
        # invert: lower uninsured = positive opportunity
        df[zcol] = - (vals - vals.mean()) / vals.std()
    else:
        df[zcol] =       (vals - vals.mean()) / vals.std()

# ─── sidebar: presets & custom sliders ────────────────────────────────
st.sidebar.header("⚙️ UOI Weights")
preset = st.sidebar.selectbox(
    "Choose preset…",
    ["Even", "Income-heavy", "Education-heavy", "Custom"],
    key="preset"
)

if preset == "Even":
    w_inc, w_bach, w_auto, w_health = 0.25, 0.25, 0.25, 0.25
elif preset == "Income-heavy":
    w_inc, w_bach, w_auto, w_health = 0.40, 0.20, 0.20, 0.20
elif preset == "Education-heavy":
    w_inc, w_bach, w_auto, w_health = 0.20, 0.40, 0.20, 0.20
else:
    w_inc    = st.sidebar.slider("Income",       0.0, 1.0, 0.25, key="s_inc")
    w_bach   = st.sidebar.slider("Bachelor’s %",  0.0, 1.0, 0.25, key="s_bach")
    w_auto   = st.sidebar.slider("No-Vehicle %",  0.0, 1.0, 0.25, key="s_auto")
    w_health = st.sidebar.slider("Uninsured %",   0.0, 1.0, 0.25, key="s_health")

# normalize to sum = 1
_total = w_inc + w_bach + w_auto + w_health
w_inc, w_bach, w_auto, w_health = (
    w_inc/_total, w_bach/_total, w_auto/_total, w_health/_total
)
st.sidebar.markdown(
    f"**Normalized:** Income {w_inc:.2f}, Edu {w_bach:.2f}, "
    f"Auto {w_auto:.2f}, Health {w_health:.2f}"
)

# ─── compute custom UOI ──────────────────────────────────────────────
df["UOI_custom"] = (
      df["Z_Median_Household_Income"] * w_inc
    + df["Z_Bachelors_Degree_Pct"]    * w_bach
    + df["Z_No_Vehicle_Pct"]          * w_auto
    + df["Z_No_Health_Insurance_Pct"] * w_health
)

# 2) Load the US counties GeoJSON
with open("geojson-counties-fips.json") as f:
    counties = json.load(f)

# 3) Keep Texas only (state FIPS = "48")
counties["features"] = [
    feat for feat in counties["features"]
    if feat["properties"]["STATE"] == "48"
]

# 4) Compute df["fips"] by matching your 'County' names to each feature's id
def fips_lookup(county_name):
    name = county_name.strip().upper()
    for feat in counties["features"]:
        if feat["properties"]["NAME"].upper() == name:
            return feat["id"]
    return None

df["fips"] = df["County"].apply(fips_lookup)

# 5) Plot the choropleth map
fig = px.choropleth(
    df,
    geojson=counties,
    locations="fips",
    color="UOI_custom",
    hover_name="County",
    color_continuous_scale="Viridis",
    scope="usa",
    labels={"UOI": "Urban Opportunity Index"},
    title="Texas Urban Opportunity Index by County",
)
fig.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig, use_container_width=True)

# 6) Optional: drop-down for details
selected = st.selectbox("Highlight a county:", df["County"].sort_values())
if selected:
    st.write(df[df["County"] == selected])

