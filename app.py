import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go  # NEW
import json
import requests
import numpy as np
from scipy.stats import pearsonr
from scipy.stats import spearmanr  # NEW: Spearman correlation
import statsmodels.api as sm       # NEW: regression analysis
from sklearn.utils import resample

# For heteroskedasticity and robust SE
from statsmodels.stats.diagnostic import het_breuschpagan

# NEW: custom display labels for indicators
DISPLAY_LABELS = {
    "Upward_Mobility": "Income Growth"
}

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

df = pd.read_csv(DATA_CSV)
# Normalize county names for merging
df['County'] = df['County'].astype(str).str.strip().str.title()

# ---- LOAD PARTISANSHIP DATA ----
part_df = pd.read_csv('./src/merged_partisan.csv')
# Standardize county names
part_df['County'] = part_df['County'].astype(str).str.strip().str.title()
# Robustly clean vote counts and margin columns using regex (allow digits and minus)
for col in ['Trump_Votes', 'Biden_Votes', 'Other_Votes', 'Margin', 'Total']:
    part_df[col] = part_df[col].astype(str).str.replace(r'[^\d-]', '', regex=True).astype(int)
# Robustly clean percent columns, removing non-numeric/non-dot chars
for col in ['Trump_Pct', 'Biden_Pct', 'Other_Pct', 'Margin_Pct']:
    part_df[col] = part_df[col].astype(str).str.replace(r'[^\d\.]', '', regex=True).astype(float)
# Define a Partisanship metric as Trump's vote share
part_df['Partisanship'] = part_df['Trump_Pct']
# Merge into main DataFrame
df = df.merge(
    part_df[['County', 'Trump_Votes', 'Trump_Pct', 'Biden_Votes', 'Biden_Pct', 'Partisanship']],
    on='County',
    how='left'
)

# ---- DATA CLEANING: remove invalid placeholders and specific counties ----
BAD_VALUES = {-666666666, -66666, -22222, -9999}
for col in ["Median_Household_Income","Bachelors_Degree_Pct","Unemployment_Rate","No_Health_Insurance_Pct","Median_Gross_Rent","Broadband_Pct","High_School_Grad_Pct","Upward_Mobility","Gini_Index","No_Vehicle_Pct"]:
    df = df[~df[col].isin(BAD_VALUES)]
df = df[df["County"].str.upper() != "LOVING"].reset_index(drop=True)

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
df["fips"] = df["fips"].apply(lambda x: x if x.startswith("48") else "48" + x[-3:])

# ---- INDICATOR LIST (modify as needed) ----
UOI_INDICATORS = [
    "Median_Household_Income",
    "Bachelors_Degree_Pct",
    "Unemployment_Rate",
    "No_Health_Insurance_Pct",
    "Median_Gross_Rent",
    "Broadband_Pct",
    "High_School_Grad_Pct",
    "Upward_Mobility",
    "Gini_Index",
    "No_Vehicle_Pct"
]
# Only keep indicators that exist in data
UOI_INDICATORS = [col for col in UOI_INDICATORS if col in df.columns]

# ---- UOI WEIGHTS SIDEBAR ----
st.sidebar.header("⚙️ UOI Weights")
preset = st.sidebar.selectbox("Choose preset…", ["Even", "Income-heavy", "Education-heavy", "Growth-heavy", "Custom"])

# Default weights evenly distributed (auto-adjust for number of indicators)
default_weight = 1 / len(UOI_INDICATORS)
weights = [default_weight] * len(UOI_INDICATORS)

# Allow for weighting presets or custom sliders
if preset == "Even":
    pass  # already set
elif preset == "Income-heavy":
    weights = [0.3 if name == "Median_Household_Income" else (0.7/(len(UOI_INDICATORS)-1)) for name in UOI_INDICATORS]
elif preset == "Education-heavy":
    # Bump Bachelor's and High School Grad weights
    weights = [
        0.3 if name == "Bachelors_Degree_Pct"
        else 0.2 if name == "High_School_Grad_Pct"
        else (0.5 / (len(UOI_INDICATORS) - 2))
        for name in UOI_INDICATORS
    ]
elif preset == "Growth-heavy":
    # Emphasize Income Growth (Upward Mobility)
    weights = [
        0.3 if name == "Upward_Mobility"
        else (0.7 / (len(UOI_INDICATORS) - 1))
        for name in UOI_INDICATORS
    ]
else:  # Custom
    for i, col in enumerate(UOI_INDICATORS):
        label = DISPLAY_LABELS.get(col, col.replace('_', ' '))
        weights[i] = st.sidebar.slider(f"{label} weight", 0.0, 1.0, default_weight)
# Normalize
weights = [w/sum(weights) for w in weights]
# Display normalized weights in a table
norm_weights_df = pd.DataFrame({
    "Indicator": [DISPLAY_LABELS.get(col, col.replace('_', ' ')) for col in UOI_INDICATORS],
    "Weight": [round(w, 2) for w in weights]
})
st.sidebar.header("🔢 Normalized Weights")
st.sidebar.table(norm_weights_df)

# NEW: filter by UOI quintile
st.sidebar.header("🔢 Filter by UOI Quintile")
quintile_options = ["All", "1st", "2nd", "3rd", "4th", "5th"]
selected_quintile = st.sidebar.selectbox("Select Quintile", quintile_options, index=0)

# ---- ENSURE Z-SCORES EXIST ----
for col in UOI_INDICATORS:
    zcol = "Z_" + col
    if col in df.columns and zcol not in df.columns:
        # For "bad" indicators, reverse Z-score so higher = better
        if col in ["Unemployment_Rate", "No_Health_Insurance_Pct", "Median_Gross_Rent", "Gini_Index", "No_Vehicle_Pct"]:
            df[zcol] = -(df[col] - df[col].mean()) / df[col].std()
        else:
            df[zcol] = (df[col] - df[col].mean()) / df[col].std()

# ---- COMPUTE UOI (Custom) ----
df["UOI_custom"] = sum(
    df[f"Z_{col}"] * w for col, w in zip(UOI_INDICATORS, weights)
)

# NEW: compute UOI quintiles for filtering, gracefully handling duplicate edges
quantiles = pd.qcut(df["UOI_custom"], 5, duplicates="drop")
# convert categorical codes into ordinal labels
codes = quantiles.cat.codes + 1
def suffix(n):
    return "th" if 11 <= n <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
df["UOI_quintile"] = [f"{c}{suffix(c)}" for c in codes]

# NEW: apply quintile filter to map data
if selected_quintile != "All":
    map_df = df[df["UOI_quintile"] == selected_quintile]
else:
    map_df = df

# ---- VIEW TABS ----
tabs = st.tabs(["Dashboard", "Research Insights"])
dashboard_tab, insights_tab = tabs

with dashboard_tab:
    # ---- MAP ----
    st.title("Texas Urban Opportunity Index (UOI) by County")
    df["fips"] = df["fips"].astype(str).str.zfill(5)

    # Base layer: all counties in grey
    fig = go.Figure()
    fig.add_trace(go.Choropleth(
        geojson=counties,
        locations=df["fips"],
        z=[0]*len(df),
        colorscale=[[0, "lightgrey"], [1, "lightgrey"]],
        showscale=False,
        marker_line_width=0,
        hoverinfo="skip",
    ))

    # Overlay: selected quintile counties colored by UOI_custom
    fig.add_trace(go.Choropleth(
        geojson=counties,
        locations=map_df["fips"],
        z=map_df["UOI_custom"],
        colorscale=px.colors.sequential.Viridis,
        zmin=df["UOI_custom"].min(),
        zmax=df["UOI_custom"].max(),
        marker_line_width=0.5,
        marker_line_color="black",
        text=map_df["County"],  # county names for tooltip
        hovertemplate="County: %{text}<br>UOI: %{z:.2f}<extra></extra>",
        colorbar_title="Urban Opportunity Index"
    ))

    fig.update_geos(
        fitbounds="locations",
        scope="usa",                   # NEW: ensure albers projection centers on USA bounds
        visible=False,
        projection={'type': 'albers usa'}
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=650,
        title="Texas Urban Opportunity Index by County"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- COUNTY DETAILS SELECTOR ----
    st.header("📋 Select a County for Details")
    county_choice = st.selectbox("Choose a county:", df["County"].sort_values())
    # Only display details if matching row exists
    matching = df[df["County"] == county_choice]
    if not matching.empty:
        details = matching.iloc[0]
        # Convert Series to DataFrame for display, include all UOI indicators and Partisanship
        details_df = details[["County", "UOI_custom"] + UOI_INDICATORS + ["Partisanship"]].to_frame().T
        # Apply display labels to details table, and map Partisanship to "Trump Vote %"
        details_df = details_df.rename(columns={**DISPLAY_LABELS, "Partisanship": "Trump Vote %"})
        st.table(details_df)
    else:
        st.warning(f"No data available for selected county: {county_choice}")

    # ---- REGIONAL COMPARISON ----
    region_options = {
        "Austin Metro": ["TRAVIS", "WILLIAMSON", "HAYS"],
        "Dallas-Fort Worth Metro": ["DALLAS", "TARRANT", "COLLIN", "DENTON"],
        "Houston Metro": ["HARRIS", "FORT BEND", "MONTGOMERY", "BRAZORIA", "GALVESTON"],
        "San Antonio Metro": ["BEXAR", "COMAL", "GUADALUPE", "MEDINA", "WILSON", "ATASCOSA", "KENDALL"],
        "Brazos Valley": ["BRAZOS", "BURLESON", "GRIMES", "LEON", "MADISON", "ROBERTSON", "WASHINGTON"],
        "Rio Grande Valley": ["CAMERON", "HIDALGO", "STARR", "WILLACY"],
        "West Texas": ["EL PASO", "MIDLAND", "ECTOR", "LUBBOCK", "TOM GREEN"],
    }
    # convert region county names to title-case to match df["County"]
    region_options = {
        region: [name.title() for name in counties_list]
        for region, counties_list in region_options.items()
    }
    # Build display-to-column mapping, applying custom labels
    indicator_options = {}
    for col in ["UOI_custom"] + UOI_INDICATORS:
        if col == "UOI_custom":
            key = "Urban Opportunity Index"
        else:
            key = DISPLAY_LABELS.get(col, col.replace('_', ' '))
        indicator_options[key] = col

    st.header("🔍 Regional Indicator Comparison")
    col1, col2, col3 = st.columns([2,2,3])
    with col1:
        region1 = st.selectbox("Region 1", list(region_options.keys()), index=0)
    with col2:
        region2 = st.selectbox("Region 2", list(region_options.keys()), index=1)
    with col3:
        indicator = st.selectbox("Indicator to compare", list(indicator_options.keys()), index=0)
    indicator_col = indicator_options[indicator]
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
    def safe_get(county, col):
        subset = df.loc[df["County"] == county, col]
        return subset.iloc[0] if not subset.empty else None
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
            safe_get(county1, col),
            safe_get(county2, col)
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
        raw_df = df[["County", "UOI_custom"] + UOI_INDICATORS].rename(columns=DISPLAY_LABELS)
        st.dataframe(raw_df)

with insights_tab:
    st.header("🧪 Research Insights")
    st.markdown("_Click and drag on graph to zoom in._")
    # Prepare clean data
    df_clean = df.dropna(subset=["UOI_custom", "Voter_Turnout"])

    # ---- PARTISANSHIP ANALYSIS ----
    if "Partisanship" in df_clean.columns:
        st.write("### Partisanship Analysis")
        # ---- ANALYSIS SCOPE ----
        # region_options is defined in dashboard_tab; we redefine here for scope filtering
        region_options = {
            "Austin Metro": ["Travis", "Williamson", "Hays"],
            "Dallas-Fort Worth Metro": ["Dallas", "Tarrant", "Collin", "Denton"],
            "Houston Metro": ["Harris", "Fort Bend", "Montgomery", "Brazoria", "Galveston"],
            "San Antonio Metro": ["Bexar", "Comal", "Guadalupe", "Medina", "Wilson", "Atascosa", "Kendall"],
            "Brazos Valley": ["Brazos", "Burleson", "Grimes", "Leon", "Madison", "Robertson", "Washington"],
            "Rio Grande Valley": ["Cameron", "Hidalgo", "Starr", "Willacy"],
            "West Texas": ["El Paso", "Midland", "Ector", "Lubbock", "Tom Green"],
        }
        # build a flat list of all metro counties
        metro = [c for region in region_options.values() for c in region]
        df_party = df.dropna(subset=["UOI_custom", "Partisanship"])
        scope = st.selectbox("Analysis scope", ["All counties", "Metro counties", "Non-metro counties"])
        if scope == "Metro counties":
            df_party = df_party[df_party["County"].isin(metro)]
        elif scope == "Non-metro counties":
            df_party = df_party[~df_party["County"].isin(metro)]
        if len(df_party) < 2:
            st.write("Not enough data available to perform Partisanship Analysis.")
        else:
            # Let user choose indicator for X-axis
            axis_options = {"Urban Opportunity Index": "UOI_custom"}
            for col in UOI_INDICATORS:
                label = DISPLAY_LABELS.get(col, col.replace('_', ' '))
                axis_options[label] = col
            x_label = st.selectbox("Select X-axis indicator for Partisanship Analysis", list(axis_options.keys()), index=0)
            x_col = axis_options[x_label]
            x_p = df_party[x_col]
            p_p = df_party["Partisanship"]

            # Pearson correlation
            corr_p, pval_p = pearsonr(x_p, p_p)
            # Spearman correlation
            spear_p_corr, spear_p_p = spearmanr(x_p, p_p)
            # Linear regression summary via statsmodels for partisanship
            Xp = sm.add_constant(x_p)
            model_p = sm.OLS(p_p, Xp).fit()
            # Heteroskedasticity test (Breusch-Pagan)
            bp_test = het_breuschpagan(model_p.resid, Xp)
            # Robust standard errors (HC3)
            robust_p = model_p.get_robustcov_results(cov_type='HC3')
            rp_df = pd.DataFrame({
                "coef": robust_p.params,
                "std err (HC3)": robust_p.bse,
                "p-value": robust_p.pvalues
            })

            # ---- BOOTSTRAP CI FOR UNIVARIATE CORR ----
            def bootstrap_ci(x, y, n_boot=1000):
                bs = []
                for _ in range(n_boot):
                    sample = resample(pd.DataFrame({"x": x, "y": y}))
                    bs.append(pearsonr(sample["x"], sample["y"])[0])
                return np.percentile(bs, [2.5, 97.5])
            ci_low, ci_high = bootstrap_ci(x_p, p_p)

            # ---- MULTIVARIATE CONTROLS ----
            ctrl_vars = ["UOI_custom", "Broadband_Pct", "Unemployment_Rate"]
            ctrl_df = df_party.dropna(subset=ctrl_vars + ["Partisanship"])
            X_ctrl = sm.add_constant(ctrl_df[ctrl_vars])
            model_ctrl = sm.OLS(ctrl_df["Partisanship"], X_ctrl).fit()

            # --- METRICS ROW ---
            st.subheader("Partisanship Analysis Overview")
            cols = st.columns(5)
            # Pearson r
            cols[0].markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-family:\"Courier New\", monospace; font-size:24px; margin:0;'>{corr_p:.3f}</div>"
                f"<div style='font-size:12px; color:gray; margin:0;'>Pearson r</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            # Pearson p‑value
            cols[1].markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-family:\"Courier New\", monospace; font-size:24px; margin:0;'>{pval_p:.3f}</div>"
                f"<div style='font-size:12px; color:gray; margin:0;'>Pearson p-value</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            # Spearman r
            cols[2].markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-family:\"Courier New\", monospace; font-size:24px; margin:0;'>{spear_p_corr:.3f}</div>"
                f"<div style='font-size:12px; color:gray; margin:0;'>Spearman r</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            # Spearman p‑value
            cols[3].markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-family:\"Courier New\", monospace; font-size:24px; margin:0;'>{spear_p_p:.3f}</div>"
                f"<div style='font-size:12px; color:gray; margin:0;'>Spearman p-value</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            # R²
            cols[4].markdown(
                f"<div style='text-align:center;'>"
                f"<div style='font-family:\"Courier New\", monospace; font-size:24px; margin:0;'>{model_p.rsquared:.3f}</div>"
                f"<div style='font-size:12px; color:gray; margin:0;'>R²</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.caption(f"95% CI for Pearson r: [{ci_low:.3f}, {ci_high:.3f}]")
            st.caption(f"Breusch-Pagan p-value: {bp_test[1]:.3f}")
            with st.expander("Regression coefficients (HC3 robust SE)"):
                st.dataframe(rp_df.style.format({"coef": "{:.3f}", "std err (HC3)": "{:.3f}", "p-value": "{:.3g}"}))
            st.caption(f"Controlled regression R²: {model_ctrl.rsquared:.3f}")

            # Scatter plot with regression, SPLIT BY PARTY
            fig_part = go.Figure()
            # split by party for partisanship
            trump_party = df_party[df_party["Partisanship"] >= df_party["Biden_Pct"]]
            harris_party = df_party[df_party["Partisanship"] < df_party["Biden_Pct"]]
            fig_part.add_trace(go.Scatter(
                x=trump_party[x_col], y=trump_party["Partisanship"],
                mode="markers", name="Trump counties",
                marker=dict(color="red"),
                text=trump_party["County"],
                customdata=np.stack((trump_party["Trump_Pct"], trump_party["Biden_Pct"]), axis=-1),
                hovertemplate=(
                    "County: %{text}<br>"
                    "<span style='color:red; font-family:\"Courier New\", monospace;'>Trump: %{customdata[0]:.2f}%</span><br>"
                    "<span style='color:lightblue; font-family:\"Courier New\", monospace;'>Harris: %{customdata[1]:.2f}%</span><br>"
                    f"{x_label}: "+"%{x:.2f}<extra></extra>"
                )
            ))
            fig_part.add_trace(go.Scatter(
                x=harris_party[x_col], y=harris_party["Partisanship"],
                mode="markers", name="Harris counties",
                marker=dict(color="lightblue"),
                text=harris_party["County"],
                customdata=np.stack((harris_party["Trump_Pct"], harris_party["Biden_Pct"]), axis=-1),
                hovertemplate=(
                    "County: %{text}<br>"
                    "<span style='color:red; font-family:\"Courier New\", monospace;'>Trump: %{customdata[0]:.2f}%</span><br>"
                    "<span style='color:lightblue; font-family:\"Courier New\", monospace;'>Harris: %{customdata[1]:.2f}%</span><br>"
                    f"{x_label}: "+"%{x:.2f}<extra></extra>"
                )
            ))
            # Fit regression for partisanship
            m2, b2 = np.polyfit(x_p, p_p, 1)
            line_x = np.linspace(x_p.min(), x_p.max(), 100)
            line_p2_y = m2 * line_x + b2
            fig_part.add_trace(go.Scatter(x=line_x, y=line_p2_y, mode="lines", name="Fit Line"))
            fig_part.update_layout(
                title=f"{x_label} vs Trump 2024 Vote %",
                xaxis_title=x_label,
                yaxis_title="Trump 2024 Vote %",
                margin=dict(l=0, r=0, t=40, b=0)
            )
            st.plotly_chart(fig_part, use_container_width=True)

            # ---- DECOMPOSITION ----
            st.markdown("**Correlation by Indicator**")
            corrs = []
            for col in UOI_INDICATORS:
                label = DISPLAY_LABELS.get(col, col.replace('_',' '))
                z = df_party[f"Z_{col}"]
                r, p = pearsonr(z, df_party["Partisanship"])
                corrs.append((label, r, p))
            decomp_df = pd.DataFrame(corrs, columns=["Indicator","Pearson r","p-value"])
            fig_decomp = px.bar(
                decomp_df,
                x="Indicator",
                y="Pearson r",
                title="Indicator vs Trump 2024 Vote % (Pearson r with p-values)",
                hover_data={"p-value": ":.3f"},
                labels={"p-value": "p-value"}
            )
            fig_decomp.update_layout(showlegend=False)
            st.plotly_chart(fig_decomp, use_container_width=True)

    if len(df_clean) < 2:
        st.write("Not enough data available to perform Voter Turnout Analysis.")
    else:
        st.write("### Voter Turnout Analysis")
        x = df_clean["UOI_custom"]
        y = df_clean["Voter_Turnout"]

        # Compute stats
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_r, spearman_p = spearmanr(x, y)
        X_turn = sm.add_constant(x)
        model_turn = sm.OLS(y, X_turn).fit()
        # Heteroskedasticity test (Breusch-Pagan)
        bp_turn = het_breuschpagan(model_turn.resid, X_turn)
        st.write(f"_Breusch-Pagan test p-value_: {bp_turn[1]:.3g}")
        # Robust standard errors (HC3)
        robust_t = model_turn.get_robustcov_results(cov_type='HC3')
        st.markdown("**Turnout regression coefficients with HC3 robust SE**")
        rt_df = pd.DataFrame({
            "coef": robust_t.params,
            "std err (HC3)": robust_t.bse,
            "p-value": robust_t.pvalues
        })
        st.dataframe(rt_df.style.format({"coef": "{:.3f}", "std err (HC3)": "{:.3f}", "p-value": "{:.3g}"}))

        # Display metrics
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-family:\"Courier New\", monospace; font-size:24px; margin:0;'>{pearson_r:.3f}</div>"
            f"<div style='font-size:12px; color:gray; margin:0;'>Pearson r</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        t2.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-family:\"Courier New\", monospace; font-size:24px; margin:0;'>{pearson_p:.3g}</div>"
            f"<div style='font-size:12px; color:gray; margin:0;'>Pearson p-value</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        t3.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-family:\"Courier New\", monospace; font-size:24px; margin:0;'>{spearman_r:.3f}</div>"
            f"<div style='font-size:12px; color:gray; margin:0;'>Spearman r</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        t4.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-family:\"Courier New\", monospace; font-size:24px; margin:0;'>{spearman_p:.3g}</div>"
            f"<div style='font-size:12px; color:gray; margin:0;'>Spearman p-value</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        t5.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='font-family:\"Courier New\", monospace; font-size:24px; margin:0;'>{model_turn.rsquared:.3f}</div>"
            f"<div style='font-size:12px; color:gray; margin:0;'>R²</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        # Scatter plot with regression, split by party
        fig_insight = go.Figure()
        trump_df  = df_clean[df_clean["Partisanship"] >= df_clean["Biden_Pct"]]
        harris_df = df_clean[df_clean["Partisanship"] <  df_clean["Biden_Pct"]]

        def mk_turn_trace(subdf, name, color):
            return go.Scatter(
                x=subdf["UOI_custom"], y=subdf["Voter_Turnout"],
                mode="markers", name=name, marker=dict(color=color),
                text=subdf["County"],
                customdata=np.stack((subdf["Trump_Pct"], subdf["Biden_Pct"]),axis=-1),
                hovertemplate=(
                    "County: %{text}<br>"
                    "UOI: %{x:.2f}<br>"
                    "Turnout: %{y:.2f}%<br>"
                    "<span style='color:red; font-family:\"Courier New\", monospace;'>"
                      "Trump: %{customdata[0]:.2f}%</span><br>"
                    "<span style='color:lightblue; font-family:\"Courier New\", monospace;'>"
                      "Harris: %{customdata[1]:.2f}%</span><extra></extra>"
                )
            )

        fig_insight.add_trace(mk_turn_trace(trump_df, "Trump counties", "red"))
        fig_insight.add_trace(mk_turn_trace(harris_df, "Harris counties", "lightblue"))

        m, b = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 100)
        fig_insight.add_trace(go.Scatter(x=line_x, y=m*line_x+b, mode="lines", name="Fit Line"))

        fig_insight.update_layout(
            title="UOI vs 2024 Presidential Election Voter Turnout",
            xaxis_title="Urban Opportunity Index (UOI)",
            yaxis_title="2024 Presidential Election Turnout (%)",
            margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig_insight, use_container_width=True)

        # Average turnout by UOI quintile
        turnout_by_quintile = df_clean.groupby("UOI_quintile")["Voter_Turnout"].mean().reset_index()
        fig_quint = px.bar(
            turnout_by_quintile,
            x="UOI_quintile", y="Voter_Turnout",
            labels={"UOI_quintile":"UOI Quintile","Voter_Turnout":"Avg Turnout (%)"},
            title="Average Voter Turnout by UOI Quintile"
        )
        st.plotly_chart(fig_quint, use_container_width=True)