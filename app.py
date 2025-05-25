import os
import streamlit as st
import pandas as pd
import plotly.express as px
import json
import requests

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="Texas UOI Dashboard",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- LOAD DATA ----
DATA_CSV = "./src/texas_counties_full.csv"
GEOJSON_F = "./src/geojson-counties-fips.json"

# ---- LOAD DATA ----
df = pd.read_csv("./src/texas_counties_full.csv")

# Fetch official FIPS-based US counties GeoJSON
counties = requests.get(
    "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
).json()

# ---- FIPS COLUMN CHECK ----
if "fips" not in df.columns:
    def fips_lookup(county_name):
        name = county_name.strip().upper()
        for feat in counties["features"]:
            if feat["properties"]["NAME"].upper() == name:
                return feat["id"]
        return None
    df["fips"] = df["County"].apply(fips_lookup)
df["fips"] = df["fips"].astype(str).str.zfill(5)
# Ensure full 5-digit Texas FIPS codes by prefixing '48'
df["fips"] = df["fips"].apply(lambda x: x if x.startswith("48") else "48" + x[-3:])

# ---- UOI WEIGHTS SIDEBAR ----
st.sidebar.header("⚙️ UOI Weights")
preset = st.sidebar.selectbox("Choose preset…", ["Even", "Income-heavy", "Education-heavy", "Custom"])

if preset == "Even":
    w_inc, w_bach, w_auto, w_health, w_rent, w_bb = [1/6]*6
elif preset == "Income-heavy":
    w_inc, w_bach, w_auto, w_health, w_rent, w_bb = 0.4, 0.1, 0.1, 0.1, 0.15, 0.15
elif preset == "Education-heavy":
    w_inc, w_bach, w_auto, w_health, w_rent, w_bb = 0.1, 0.4, 0.1, 0.1, 0.15, 0.15
else:  # Custom
    w_inc    = st.sidebar.slider("Income weight",            0.0, 1.0, 1/6)
    w_bach   = st.sidebar.slider("Bachelor’s % weight",      0.0, 1.0, 1/6)
    w_auto   = st.sidebar.slider("No Vehicle % weight",      0.0, 1.0, 1/6)
    w_health = st.sidebar.slider("Uninsured % weight",       0.0, 1.0, 1/6)
    w_rent   = st.sidebar.slider("Median Rent weight",       0.0, 1.0, 1/6)
    w_bb     = st.sidebar.slider("Broadband % weight",       0.0, 1.0, 1/6)

# Normalize weights so they sum to 1
_total = w_inc + w_bach + w_auto + w_health + w_rent + w_bb
w_inc, w_bach, w_auto, w_health, w_rent, w_bb = (
    w_inc/_total, w_bach/_total, w_auto/_total,
    w_health/_total, w_rent/_total, w_bb/_total
)
st.sidebar.markdown(
    "**Normalized weights:**  " +
    f"Income {w_inc:.2f},  " +
    f"Edu {w_bach:.2f},  " +
    f"No Vehicle {w_auto:.2f},  " +
    f"Health {w_health:.2f},  " +
    f"Rent {w_rent:.2f},  " +
    f"Broadband {w_bb:.2f}"
)
# ---- ENSURE UOI_CUSTOM EXISTS ----
# Create Z-score columns if missing
for col in [
    "Median_Household_Income",
    "Bachelors_Degree_Pct",
    "No_Vehicle_Pct",
    "No_Health_Insurance_Pct",
    "Median_Gross_Rent",
    "Broadband_Pct"
]:
    zcol = "Z_" + col
    if col in df.columns and zcol not in df.columns:
        if col in ["No_Vehicle_Pct", "No_Health_Insurance_Pct", "Median_Gross_Rent"]:
            df[zcol] = -(df[col] - df[col].mean()) / df[col].std()
        else:
            df[zcol] = (df[col] - df[col].mean()) / df[col].std()

# Compute UOI_custom on the fly if missing
if "UOI_custom" not in df.columns:
    df["UOI_custom"] = (
        df["Z_Median_Household_Income"] * w_inc
      + df["Z_Bachelors_Degree_Pct"]    * w_bach
      + df["Z_No_Vehicle_Pct"]          * w_auto
      + df["Z_No_Health_Insurance_Pct"] * w_health
      + df["Z_Median_Gross_Rent"]       * w_rent
      + df["Z_Broadband_Pct"]           * w_bb
    )

# ---- MAP ----
st.title("Texas Urban Opportunity Index (UOI) by County")

# Ensure FIPS is string and padded
df["fips"] = df["fips"].astype(str).str.zfill(5)

# Plot the choropleth
fig = px.choropleth(
    df,
    geojson=counties,
    locations="fips",
    featureidkey="id",
    color="UOI_custom",
    hover_name="County",
    color_continuous_scale=px.colors.sequential.Viridis,
    color_continuous_midpoint=df["UOI_custom"].mean(),
    scope="usa",               # ensures map focuses on USA
    labels={"UOI_custom": "Urban Opportunity Index"},
    title="Texas Urban Opportunity Index by County",
)
fig.update_geos(fitbounds="locations", visible=False)
fig.update_layout(margin=dict(l=0, r=0, t=40, b=0), height=520)
st.plotly_chart(fig, use_container_width=True)


# ---- COUNTY DETAILS SELECTOR ----
st.header("📋 Select a County for Details")
county_choice = st.selectbox("Choose a county:", df["County"].sort_values())
details = df[df["County"] == county_choice].iloc[0]
# Convert Series to one-row DataFrame and rename columns
details_df = details.to_frame().T.rename(columns={
    "UOI_custom": "UOI",
    "Median_Household_Income": "Income",
    "Bachelors_Degree_Pct": "Bachelor's %",
    "No_Health_Insurance_Pct": "No Insurance %",
    "Median_Gross_Rent": "Rent",
    "Broadband_Pct": "Broadband %",
    "No_Vehicle_Pct": "No Vehicle %"
})
st.table(details_df)


# ---- REGIONAL COMPARISON SETUP ----
region_options = {
    "Austin Metro": ["TRAVIS", "WILLIAMSON", "HAYS"],
    "Dallas-Fort Worth Metro": ["DALLAS", "TARRANT", "COLLIN", "DENTON"],
    "Houston Metro": ["HARRIS", "FORT BEND", "MONTGOMERY", "BRAZORIA", "GALVESTON"],
    "San Antonio Metro": ["BEXAR", "COMAL", "GUADALUPE", "MEDINA", "WILSON", "ATASCOSA", "KENDALL"],
    "Rio Grande Valley": ["CAMERON", "HIDALGO", "STARR", "WILLACY"],
    "West Texas": ["EL PASO", "MIDLAND", "ECTOR", "LUBBOCK", "TOM GREEN"],
}
indicator_options = {
    "Urban Opportunity Index": "UOI_custom",
    "Median Household Income": "Median_Household_Income",
    "Bachelor's Degree %": "Bachelors_Degree_Pct",
    "No Health Insurance %": "No_Health_Insurance_Pct",
    "Median Gross Rent": "Median_Gross_Rent",
    "Broadband %": "Broadband_Pct",
    "No Vehicle %": "No_Vehicle_Pct"
}

# ---- USER INPUTS ----
st.header("🔍 Regional Indicator Comparison")
col1, col2, col3 = st.columns([2,2,3])
with col1:
    region1 = st.selectbox("Region 1", list(region_options.keys()), index=0)
with col2:
    region2 = st.selectbox("Region 2", list(region_options.keys()), index=1)
with col3:
    indicator = st.selectbox("Indicator to compare", list(indicator_options.keys()), index=0)

indicator_col = indicator_options[indicator]

# ---- CALCULATE AVERAGES ----
def regional_avg(counties_list, col):
    subset = df[df["County"].isin(counties_list)]
    return round(subset[col].mean(), 2) if not subset.empty else None

comp_df = pd.DataFrame({
    "Region": [region1, region2],
    indicator: [
        regional_avg(region_options[region1], indicator_col),
        regional_avg(region_options[region2], indicator_col)
    ]
})

# ---- INTERACTIVE BAR CHART ----
st.subheader(f"Average {indicator} by Region")
bar_fig = px.bar(
    comp_df,
    x="Region",
    y=indicator,
    color="Region",
    text=indicator,
    color_discrete_sequence=px.colors.qualitative.Pastel,
    title=f"{indicator} Comparison"
)
bar_fig.update_traces(textposition='outside')
bar_fig.update_layout(showlegend=False, yaxis_title="Average Value")


st.plotly_chart(bar_fig, use_container_width=True)


# ---- COUNTY-TO-COUNTY INDICATOR COMPARISON ----
st.header("🏙️ County‑to‑County Indicator Comparison")
col1, col2, col3 = st.columns([2,2,3])
with col1:
    county1 = st.selectbox("County A", df["County"].sort_values(), index=0, key="countyA")
with col2:
    county2 = st.selectbox("County B", df["County"].sort_values(), index=1, key="countyB")
with col3:
    county_indicator = st.selectbox("Indicator", list(indicator_options.keys()), index=0, key="countyIndicator")
col = indicator_options[county_indicator]
comp_county_df = pd.DataFrame({
    "County": [county1, county2],
    county_indicator: [
        df.loc[df["County"]==county1, col].iloc[0],
        df.loc[df["County"]==county2, col].iloc[0]
    ]
})
st.subheader(f"{county_indicator} Comparison")
county_fig = px.bar(
    comp_county_df,
    x="County",
    y=county_indicator,
    color="County",
    text=county_indicator,
    color_discrete_sequence=px.colors.qualitative.Vivid,
)
county_fig.update_traces(textposition='outside')
county_fig.update_layout(showlegend=False, yaxis_title=f"Average {county_indicator}")
st.plotly_chart(county_fig, use_container_width=True)


# ---- RAW DATA TABLE ----
with st.expander("Show raw data table"):
    st.dataframe(df[[
        "County", "UOI_custom", "Median_Household_Income",
        "Bachelors_Degree_Pct", "No_Health_Insurance_Pct",
        "Median_Gross_Rent", "Broadband_Pct", "No_Vehicle_Pct"
    ]])