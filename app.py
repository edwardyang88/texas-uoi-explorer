import streamlit as st
import pandas as pd
import plotly.express as px
import json

st.set_page_config(layout="wide")
st.title("Texas Urban Opportunity Index (UOI) by County")

# Load up the main CSV
df = pd.read_csv("texas_counties_full.csv")

# Load the GeoJSON with all US counties (just filter TX below)
with open("geojson-counties-fips.json") as f:
    counties = json.load(f)

# Only want Texas counties (state FIPS = 48)
counties["features"] = [
    feat for feat in counties["features"]
    if feat["properties"]["STATE"] == "48"
]

# If 'fips' not in CSV for some reason, try to match county names to FIPS from geojson
if "fips" not in df.columns:
    def fips_lookup(county_name):
        name = county_name.strip().upper()
        for feat in counties["features"]:
            if feat["properties"]["NAME"].upper() == name:
                return feat["id"]
        return None
    df["fips"] = df["County"].apply(fips_lookup)

# Sidebar stuff — lets you set how much weight each variable has in the index
st.sidebar.header("⚙️ UOI Weights")
preset = st.sidebar.selectbox("Choose preset…",
    ["Even", "Income-heavy", "Education-heavy", "Custom"]
)

# Set the weights for each factor based on user pick
if preset == "Even":
    w_inc, w_bach, w_auto, w_health, w_rent, w_bb = [1/6]*6
elif preset == "Income-heavy":
    w_inc, w_bach, w_auto, w_health, w_rent, w_bb = 0.4, 0.1, 0.1, 0.1, 0.15, 0.15
elif preset == "Education-heavy":
    w_inc, w_bach, w_auto, w_health, w_rent, w_bb = 0.1, 0.4, 0.1, 0.1, 0.15, 0.15
else:
    w_inc    = st.sidebar.slider("Income",      0.0, 1.0, 1/6)
    w_bach   = st.sidebar.slider("Bachelor’s %", 0.0, 1.0, 1/6)
    w_auto   = st.sidebar.slider("No-Vehicle %", 0.0, 1.0, 1/6)
    w_health = st.sidebar.slider("Uninsured %",  0.0, 1.0, 1/6)
    w_rent   = st.sidebar.slider("Median Rent",  0.0, 1.0, 1/6)
    w_bb     = st.sidebar.slider("Broadband %",  0.0, 1.0, 1/6)

# Normalize so the weights add up to 1
_total = w_inc + w_bach + w_auto + w_health + w_rent + w_bb
w_inc, w_bach, w_auto, w_health, w_rent, w_bb = (
    w_inc/_total, w_bach/_total, w_auto/_total, w_health/_total, w_rent/_total, w_bb/_total
)
st.sidebar.markdown(
    f"**Normalized:** Income {w_inc:.2f}, Edu {w_bach:.2f}, "
    f"Auto {w_auto:.2f}, Health {w_health:.2f}, Rent {w_rent:.2f}, Broadband {w_bb:.2f}"
)

# Add Z-score columns if missing (just in case)
for col in ["Median_Household_Income", "Bachelors_Degree_Pct", "No_Vehicle_Pct", "No_Health_Insurance_Pct", "Median_Gross_Rent", "Broadband_Pct"]:
    zcol = "Z_" + col
    if zcol not in df.columns and col in df.columns:
        if col == "No_Health_Insurance_Pct":  # Lower uninsured = better
            df[zcol] = -(df[col] - df[col].mean()) / df[col].std()
        elif col == "No_Vehicle_Pct":         # Lower = better (more cars)
            df[zcol] = -(df[col] - df[col].mean()) / df[col].std()
        elif col == "Median_Gross_Rent":      # Lower rent = better
            df[zcol] = -(df[col] - df[col].mean()) / df[col].std()
        else:                                 # Higher is better for these
            df[zcol] = (df[col] - df[col].mean()) / df[col].std()

# Calculate the UOI using all six factors and the weights from above
df["UOI_custom"] = (
      df["Z_Median_Household_Income"] * w_inc
    + df["Z_Bachelors_Degree_Pct"]    * w_bach
    + df["Z_No_Vehicle_Pct"]          * w_auto
    + df["Z_No_Health_Insurance_Pct"] * w_health
    + df["Z_Median_Gross_Rent"]       * w_rent
    + df["Z_Broadband_Pct"]           * w_bb
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
# County details dropdown — show the actual numbers for the selected county
selected = st.selectbox("Highlight a county:", df["County"].sort_values())
if selected:
    st.write(df[df["County"] == selected][[
        "County",
        "UOI_custom",
        "Median_Household_Income",
        "Bachelors_Degree_Pct",
        "No_Vehicle_Pct",
        "No_Health_Insurance_Pct",
        "Median_Gross_Rent",
        "Broadband_Pct",
        "Voter_Turnout"
    ]].rename(columns={
        "UOI_custom":"UOI (custom)"
    }))

# Raw table — if you want to see everything at once, expand this
with st.expander("Show raw data table"):
    st.dataframe(df[[
        "County",
        "UOI_custom",
        "Median_Household_Income",
        "Bachelors_Degree_Pct",
        "No_Vehicle_Pct",
        "No_Health_Insurance_Pct",
        "Median_Gross_Rent",
        "Broadband_Pct",
        "Voter_Turnout"
    ]])
