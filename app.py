"""San Diego County Literacy Dashboard (Streamlit)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="San Diego Literacy Dashboard", page_icon="📚", layout="wide")

DATA = Path(__file__).parent / "data"
MAP_CENTER = {"lat": 33.02, "lon": -116.85}
MAP_ZOOM = 8
MAP_STYLE = "carto-positron"

# Okabe-Ito colorblind-safe palette
BLUE, ORANGE, VERMILLION, GREEN, SKY_BLUE, PURPLE, GREY = (
    "#0072B2", "#E69F00", "#D55E00", "#009E73", "#56B4E9", "#CC79A7", "#999999")
AFFILIATION_COLORS = {"Council-affiliated": BLUE, "Independent": ORANGE}
PROGRAM_TYPES = {
    "Adult literacy": "adult literacy|adult education",
    "ESL": "esl",
    "Family literacy": "family literacy",
    "Early childhood": "early literacy",
}
YEARS = [str(year) for year in range(2017, 2024)]


# ---------------------------------------------------------------- data

@st.cache_data
def load_data():
    with open(DATA / "Literacy_Hotspot_ZIPs.geojson") as file:
        geojson = json.load(file)
    for feature in geojson["features"]:
        feature["properties"] = {"zip_code": feature["properties"]["zip_code"]}

    read = lambda name: pd.read_csv(DATA / name, dtype={"zip_code": str})
    scores = read("Literacy_MicroArea_Scores.csv")
    access = read("Literacy_Transit_Accessibility_Scores.csv")
    rankings = read("Literacy_ServiceGap_Rankings.csv")
    analysis = read("Literacy_Hotspot_Analysis.csv")
    level1 = read("Level1_Validated_2017_2023.csv")
    programs = pd.read_csv(DATA / "Programs_all_geocoded.csv", dtype=str)
    stops = pd.read_csv(DATA / "MTS_GTFS_processed.csv", dtype={"stop_id": str})

    # hotspot classes stored in the GeoJSON properties
    with open(DATA / "Literacy_Hotspot_ZIPs.geojson") as file:
        properties = pd.DataFrame([f["properties"] for f in json.load(file)["features"]])
    hotspot_cols = [c for c in ["zip_code", "need_tier", "hotspot_class", "gi_z", "gi_p"] if c in properties]
    zips = scores.merge(properties[hotspot_cols], on="zip_code", how="left")
    zips["need_tier"] = zips["need_tier"].fillna("Not rated")
    if "hotspot_class" not in zips:
        zips["hotspot_class"] = np.nan
    zips["hotspot_class"] = zips["hotspot_class"].fillna("Not rated")
    zips["label"] = zips["zip_code"] + " " + zips["city"].fillna("")

    # demographic segments: thirds of rated ZIP codes
    for col, name in [("pct_poverty", "poverty_group"), ("pct_limited_english", "english_group")]:
        rated_values = zips.loc[zips["eligible"], col]
        low, high = rated_values.quantile([1 / 3, 2 / 3])
        zips[name] = np.select([zips[col].isna(), zips[col] < low, zips[col] < high],
                               ["No data", "Low", "Medium"], default="High")

    # Level 1 history, cleaned like the EDA: missing, zero or repeated fill-in values removed
    level1 = level1[level1["geo_level"] == "zipcode"].copy()
    level1["zip_code"] = level1["zip_code"].str.zfill(5)
    counts = level1[YEARS].stack().round(12).value_counts()
    fill_values = counts[(counts >= 5) & (counts.index != 0)].index
    for year in YEARS:
        bad = level1[year].isna() | (level1[year] == 0) | level1[year].round(12).isin(fill_values)
        level1.loc[bad, year] = np.nan

    # programs
    programs["lat"] = pd.to_numeric(programs["Latitude"], errors="coerce")
    programs["lon"] = pd.to_numeric(programs["Longitude"], errors="coerce")
    programs["Affiliation"] = np.where(programs["Source File"] == "Affiliated", "Council-affiliated", "Independent")
    program_type = programs["Program Type"].fillna("").str.lower()
    for name, pattern in PROGRAM_TYPES.items():
        programs[name] = program_type.str.contains(pattern)
    programs["mappable"] = programs["lat"].between(32.53, 33.51) & programs["lon"].between(-117.61, -116.08)

    return geojson, zips, access, rankings, analysis, level1, programs, stops


geojson, zips, access, rankings, analysis, level1, programs, stops = load_data()


# ---------------------------------------------------------------- helpers

def choropleth(data, column, title, colors="viridis", categories=None, range_color=None, hover=None):
    hover = hover or {}
    common = dict(geojson=geojson, locations="zip_code", featureidkey="properties.zip_code",
                  center=MAP_CENTER, zoom=MAP_ZOOM, mapbox_style=MAP_STYLE, opacity=0.7,
                  hover_name="label", hover_data=hover)
    if categories:
        fig = px.choropleth_mapbox(data, color=column, color_discrete_map=categories,
                                category_orders={column: list(categories)}, **common)
    else:
        fig = px.choropleth_mapbox(data, color=column, color_continuous_scale=colors, range_color=range_color, **common)
        fig.update_layout(coloraxis_colorbar=dict(title=title, thickness=12))
    fig.update_layout(height=600, margin=dict(l=0, r=0, t=0, b=0),
                      legend=dict(title=title, yanchor="top", y=0.99, xanchor="left", x=0.01,
                                  bgcolor="rgba(255,255,255,0.85)"))
    return fig


def add_programs(fig, shown):
    for affiliation, color in AFFILIATION_COLORS.items():
        part = shown[shown["Affiliation"] == affiliation]
        if len(part):
            fig.add_trace(go.Scattermapbox(
                lat=part["lat"], lon=part["lon"], mode="markers", name=f"{affiliation} program ({len(part)})",
                marker=dict(size=11, color=color), text=part["Program Name"],
                customdata=part[["Program Type", "City"]].fillna("").values,
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
    return fig


def filter_programs(affiliation, types):
    shown = programs[programs["mappable"]]
    if affiliation != "All programs":
        shown = shown[shown["Affiliation"] == affiliation]
    if types:
        shown = shown[shown[types].any(axis=1)]
    return shown


def download(data, file_name, label="Download this table (CSV)"):
    st.download_button(label, data.to_csv(index=False).encode("utf-8"), file_name=file_name, mime="text/csv")


def demographic_filter(data, key):
    with st.sidebar.expander("Demographic segments", expanded=False):
        poverty = st.multiselect("Poverty rate (thirds of ZIP codes)", ["High", "Medium", "Low"],
                                 default=["High", "Medium", "Low"], key=f"pov_{key}")
        english = st.multiselect("Limited-English households (thirds)", ["High", "Medium", "Low"],
                                 default=["High", "Medium", "Low"], key=f"eng_{key}")
        cities = st.multiselect("City", sorted(data["city"].dropna().unique()), key=f"city_{key}")
    keep = data["poverty_group"].isin(poverty + ["No data"]) & data["english_group"].isin(english + ["No data"])
    if cities:
        keep &= data["city"].isin(cities)
    return data[keep]


SOURCE = ("Sources: Council Level 1 rates (2017–2023); Level 2/3 model estimates; ACS 5-year; "
          "program inventory; MTS GTFS; IMLS Public Libraries Survey FY2024; Census 2020 ZCTA boundaries.")


# ---------------------------------------------------------------- pages

def page_hotspots():
    st.header("Literacy hotspots and literacy programs")
    side = st.sidebar
    layer = side.radio("Map layer", ["Need tier", "Gi* hotspot class", "Level 1 rate"])
    affiliation = side.radio("Programs shown", ["All programs", "Council-affiliated", "Independent"])
    types = side.multiselect("Program type", list(PROGRAM_TYPES))

    if layer == "Need tier":
        fig = choropleth(zips, "need_tier", "Need tier", categories={
            "High need": VERMILLION, "Medium need": ORANGE, "Low need": SKY_BLUE, "Not rated": GREY})
    elif layer == "Gi* hotspot class":
        fig = choropleth(zips, "hotspot_class", "Gi* class", categories={
            "Priority hotspot": VERMILLION, "Isolated high-need ZIP": ORANGE, "Cluster edge": PURPLE,
            "Cold spot": SKY_BLUE, "Not significant": "#DDDDDD", "Not rated": GREY})
    else:
        fig = choropleth(zips, "Lit_P1", "Level 1 rate", colors="viridis_r")

    shown = filter_programs(affiliation, types)
    st.plotly_chart(add_programs(fig, shown), use_container_width=True)

    left, right, third = st.columns(3)
    left.metric("Programs shown", len(shown))
    right.metric("High-need ZIP codes", int((zips["need_tier"] == "High need").sum()))
    third.metric("Priority hotspots (Gi*)", int((zips["hotspot_class"] == "Priority hotspot").sum()))
    st.caption("High need = top 25% of Level 1 rates. Priority hotspot = high need and part of a significant "
               "Getis-Ord Gi* cluster. " + SOURCE)
    download(shown[["Program Name", "Program Type", "Affiliation", "City", "ZIP", "lat", "lon"]],
             "programs_shown.csv", "Download programs shown (CSV)")


def page_levels():
    st.header("Literacy rates by ZIP code, all levels")
    levels = {"Level 1": "Lit_P1", "Level 2": "Lit_P2", "Level 3 or above": "Lit_P3"}
    level = st.sidebar.radio("Literacy level", list(levels))
    data = demographic_filter(zips, "levels")
    column = levels[level]

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(choropleth(data, column, f"{level} rate",
                                   colors="viridis_r" if level != "Level 3 or above" else "viridis"),
                        use_container_width=True)
    with right:
        long = data.melt(id_vars=["label"], value_vars=list(levels.values()), var_name="level", value_name="rate")
        long["level"] = long["level"].map({v: k for k, v in levels.items()})
        fig = px.box(long, x="level", y="rate", color="level", points="all", hover_name="label",
                     color_discrete_sequence=[VERMILLION, ORANGE, SKY_BLUE])
        fig.update_layout(showlegend=False, height=300, margin=dict(t=30, b=0), yaxis_tickformat=".0%",
                          title="Distribution across the ZIP codes shown")
        st.plotly_chart(fig, use_container_width=True)

        top = data.nlargest(10, column)[["label", column]]
        fig = px.bar(top.sort_values(column), x=column, y="label", orientation="h",
                     color_discrete_sequence=[BLUE], title=f"10 highest {level} rates")
        fig.update_layout(height=300, margin=dict(t=30, b=0), xaxis_tickformat=".0%", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    table = data[["zip_code", "city", "total_population", "Lit_P1", "Lit_P2", "Lit_P3", "pct_poverty",
                  "pct_limited_english"]].sort_values(column, ascending=(level == "Level 3 or above"))
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Level 1 = validated 2023 Council rates. Levels 2 and 3 = model estimates, scaled so the three "
               "levels add up to 100%. " + SOURCE)
    download(table, "literacy_rates_by_zip.csv")


def page_trends():
    st.header("Level 1 literacy rates over time, 2017–2023")
    year = st.select_slider("Year", options=YEARS, value="2023")
    start, end = st.sidebar.select_slider("Change period", options=YEARS, value=("2017", "2023"))
    trend_filter = st.sidebar.multiselect("Trend group", ["Persistent low literacy", "Improving", "Emerging problem"])

    data = zips[["zip_code", "label"]].merge(level1[["zip_code"] + YEARS], on="zip_code", how="left")
    data = data.merge(analysis[["zip_code", "persistent_low_literacy", "improving", "emerging_problem"]],
                      on="zip_code", how="left")
    data["change"] = data[end] - data[start]
    flags = {"Persistent low literacy": "persistent_low_literacy", "Improving": "improving",
             "Emerging problem": "emerging_problem"}
    if trend_filter:
        data = data[data[[flags[f] for f in trend_filter]].fillna(False).astype(bool).any(axis=1)]

    top = max(data[YEARS].max().max(), 0.01)
    left, right = st.columns(2)
    with left:
        st.subheader(f"Level 1 rate, {year}")
        st.plotly_chart(choropleth(data, year, "Level 1 rate", colors="viridis_r", range_color=(0, top)),
                        use_container_width=True)
    with right:
        st.subheader(f"Change, {start} to {end}")
        largest = max(data["change"].abs().max(), 0.01)
        st.plotly_chart(choropleth(data, "change", "Change", colors="PuOr_r", range_color=(-largest, largest)),
                        use_container_width=True)
    st.caption("Orange = Level 1 rate rose (worse); purple = fell (better). ZIP codes with a missing, zero "
               "or repeated fill-in value for a year are left blank for that year.")

    chosen = st.multiselect("Compare ZIP codes over time", data["label"].tolist(),
                            default=data.dropna(subset=[end]).nlargest(3, end)["label"].tolist())
    lines = data[data["label"].isin(chosen)].melt(id_vars="label", value_vars=YEARS, var_name="year", value_name="rate")
    fig = px.line(lines, x="year", y="rate", color="label", markers=True,
                  color_discrete_sequence=[BLUE, VERMILLION, GREEN, ORANGE, PURPLE, SKY_BLUE])
    fig.update_layout(height=350, yaxis_tickformat=".0%", yaxis_title="Level 1 rate", legend_title="")
    st.plotly_chart(fig, use_container_width=True)
    download(data.drop(columns=["label"]), "level1_trends.csv")


def page_accessibility():
    st.header("Access to literacy programs by walking and MTS transit")
    side = st.sidebar
    affiliation = side.radio("Programs counted", ["All programs", "Council-affiliated only"])
    threshold = side.slider("Accessibility threshold (minutes)", 15, 120, 45, step=5)
    show_stops = side.checkbox("Show MTS transit layer", value=True)
    mode_filter = side.multiselect("Transit modes", sorted(stops["modes"].dropna().unique()))

    version = "all" if affiliation == "All programs" else "affiliated"
    data = zips[["zip_code", "label"]].merge(access, on="zip_code", how="left")
    data["minutes"] = data[f"travel_min_{version}"].clip(upper=180)
    data["within"] = np.where(data[f"travel_min_{version}"] <= threshold, f"Within {threshold} min", f"Over {threshold} min")

    view = st.radio("Map shows", ["Travel time to nearest program", f"Within {threshold} minutes?", "Transit access score"],
                    horizontal=True)
    if view.startswith("Travel"):
        fig = choropleth(data, "minutes", "Minutes", colors="viridis_r", range_color=(0, 120),
                         hover={f"travel_min_{version}": ":.0f"})
    elif view.startswith("Within"):
        fig = choropleth(data, "within", "Nearest program",
                         categories={f"Within {threshold} min": GREEN, f"Over {threshold} min": VERMILLION})
    else:
        fig = choropleth(data, f"access_score_{version}", "Access score (0–100)", colors="viridis")

    if show_stops:
        shown_stops = stops if not mode_filter else stops[stops["modes"].isin(mode_filter)]
        fig.add_trace(go.Scattermapbox(lat=shown_stops["stop_lat"], lon=shown_stops["stop_lon"], mode="markers",
                                    name=f"MTS stops ({len(shown_stops):,})", marker=dict(size=4, color="#333333"),
                                    text=shown_stops["stop_name"], hovertemplate="%{text}<extra></extra>"))
    st.plotly_chart(fig, use_container_width=True)

    rated = data[zips["eligible"].values]
    a, b, c = st.columns(3)
    a.metric(f"ZIP codes within {threshold} min", int((rated[f"travel_min_{version}"] <= threshold).sum()))
    b.metric(f"Residents within {threshold} min",
             f"{rated.loc[rated[f'travel_min_{version}'] <= threshold, 'total_population'].sum():,.0f}")
    c.metric("Median minutes to nearest program", f"{rated[f'travel_min_{version}'].median():.0f}")
    st.caption("Door-to-door weekday midday travel: walking + MTS bus/trolley with up to one transfer and average "
               "waits. NCTD (North County) service is not included. " + SOURCE)
    table = data[["zip_code", "city", f"travel_min_{version}", f"programs_45min_{version}",
                  f"access_score_{version}", "in_mts_area"]]
    download(table, f"accessibility_{version}.csv")


def page_gaps():
    st.header("Service gap rankings")
    side = st.sidebar
    lens = side.radio("Rank by", ["Overall (literacy level 1 + 2)", "Literacy level", "Program type"])
    if lens == "Literacy level":
        level = side.radio("Level", ["Level 1", "Level 2", "Level 3"])
        key = {"Level 1": "level1", "Level 2": "level2", "Level 3": "level3"}[level]
    elif lens == "Program type":
        kind = side.radio("Program type", list(PROGRAM_TYPES))
        key = {"Adult literacy": "adult_literacy", "ESL": "esl", "Family literacy": "family_literacy",
               "Early childhood": "early_childhood"}[kind]
    else:
        key = None
    affiliation = side.radio("Programs counted", ["All programs", "Council-affiliated only"],
                             disabled=key is not None, help="Level and type rankings use all programs.")
    critical_only = side.checkbox("Only ZIP codes where independent programs are critical")

    if key is None:
        key = "all" if affiliation == "All programs" else "affiliated"
    data = rankings[rankings["eligible"]].merge(
        zips[["zip_code", "label", "poverty_group", "english_group"]], on="zip_code", how="left")
    data = demographic_filter(data, "gaps")
    if critical_only:
        data = data[data["independent_critical"]]
    severity, rank = f"gap_severity_{key}", f"gap_rank_{key}"
    data = data.sort_values(rank)

    top_n = st.slider("Number of ZIP codes in the chart", 5, 40, 15)
    chart = data.head(top_n).sort_values(severity).copy()
    chart["Independent programs"] = np.where(chart["independent_critical"], "Critical here", "Not critical")
    fig = px.bar(chart, x=severity, y="label", orientation="h", color="Independent programs",
                 color_discrete_map={"Critical here": ORANGE, "Not critical": BLUE},
                 labels={severity: "Gap severity (0–100)"})
    fig.update_layout(height=max(350, 22 * top_n), yaxis_title="", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig, use_container_width=True)

    columns = ["zip_code", "city", rank, severity, "adults_below_level3", "programs_10km_affiliated",
               "programs_10km_independent", "gap_severity_affiliated", "gap_severity_all", "severity_change",
               "independent_critical"]
    table = data[list(dict.fromkeys(columns))]
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.caption("Gap severity = 50% adults per program within 10 km (log scale) + 50% lack of access. "
               "Higher = larger gap; rank 1 = largest gap. Only rated ZIP codes (1,000+ residents, Census income data).")
    download(table, f"service_gap_rankings_{key}.csv")


def page_score():
    st.header("Micro-Area Literacy Score")
    side = st.sidebar
    measure = side.radio("Map shows", ["Literacy Score", "Need severity", "Service access gap", "Trend direction",
                                       "Score with affiliated programs only"])
    tiers = side.multiselect("Priority tier", ["High", "Medium", "Low", "Not rated"], default=["High", "Medium", "Low", "Not rated"])
    robust_only = side.checkbox("Only robust priorities (top 25% in 80%+ of sensitivity runs)")
    column = {"Literacy Score": "literacy_score", "Need severity": "need_severity",
              "Service access gap": "service_access_gap", "Trend direction": "trend_direction",
              "Score with affiliated programs only": "literacy_score_affiliated"}[measure]

    data = demographic_filter(zips[zips["priority_tier"].isin(tiers)], "score")
    if robust_only:
        data = data[data["robust_priority"]]

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(choropleth(data, column, measure, colors="viridis", range_color=(0, 100),
                                   hover={"score_rank": True, "priority_tier": True}), use_container_width=True)
    with right:
        options = data[data["eligible"]].sort_values("score_rank")["label"].tolist()
        if options:
            chosen = st.selectbox("Breakdown for ZIP code", options)
            row = data[data["label"] == chosen].iloc[0]
            st.metric("Literacy Score", f"{row['literacy_score']:.1f}",
                      help=f"Rank {row['score_rank']:.0f}; tier {row['priority_tier']}")
            subs = pd.DataFrame({"part": ["Need severity (50%)", "Service access gap (30%)", "Trend direction (20%)"],
                                 "value": [row["need_severity"], row["service_access_gap"], row["trend_direction"]]})
            fig = px.bar(subs, x="value", y="part", orientation="h", range_x=(0, 100),
                         color="part", color_discrete_sequence=[VERMILLION, ORANGE, GREEN])
            fig.update_layout(height=200, showlegend=False, margin=dict(t=10, b=0), yaxis_title="", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)

            components = {
                "Literacy level index": "c_literacy_level_index", "Demographic risk": "c_demographic_risk",
                "Program access": "c_program_access_all", "Program supply vs. demand": "c_program_supply_all",
                "Library proximity": "c_library_proximity", "Library usage": "c_library_usage",
                "Level 1 change": "c_level1_change", "Level 1 yearly trend": "c_level1_slope"}
            parts = pd.DataFrame({"component": list(components), "value": [row[c] for c in components.values()]})
            fig = px.bar(parts, x="value", y="component", orientation="h", range_x=(0, 100),
                         color_discrete_sequence=[BLUE], title="Components (0–100, higher = more need)")
            fig.update_layout(height=320, margin=dict(t=40, b=0), yaxis_title="", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Without independent programs the score would be {row['literacy_score_affiliated']:.1f} "
                       f"(+{row['independent_program_effect']:.1f}). Nearest library: {row['nearest_library']} "
                       f"({row['library_km']:.1f} km).")
        else:
            st.info("No rated ZIP codes match the filters.")

    st.caption("Score 0–100; higher = greater literacy need and priority. Affiliated and independent programs "
               "count equally in the main score. " + SOURCE)
    table = data.drop(columns=["label", "poverty_group", "english_group"], errors="ignore").sort_values("score_rank")
    st.dataframe(table, hide_index=True, use_container_width=True)
    download(table, "literacy_microarea_scores.csv")


def page_directory():
    st.header("Program directory")
    side = st.sidebar
    search = st.text_input("Search by name, organization, city or ZIP code", "")
    affiliation = side.radio("Affiliation", ["All programs", "Council-affiliated", "Independent"])
    types = side.multiselect("Program type", list(PROGRAM_TYPES))
    cities = side.multiselect("City", sorted(programs["City"].dropna().str.strip().unique()))

    shown = programs.copy()
    if affiliation != "All programs":
        shown = shown[shown["Affiliation"] == affiliation]
    if types:
        shown = shown[shown[types].any(axis=1)]
    if cities:
        shown = shown[shown["City"].str.strip().isin(cities)]
    if search:
        text = shown[["Program Name", "Parent Organisation", "City", "ZIP"]].fillna("").agg(" ".join, axis=1)
        shown = shown[text.str.contains(search, case=False, regex=False)]

    st.write(f"**{len(shown)}** programs match.")
    columns = ["Program Name", "Parent Organisation", "Affiliation", "Program Type", "Street or Mailing Address",
               "City", "ZIP", "Hours", "Phone", "Email", "Website", "Capacity"]
    table = shown[[c for c in columns if c in shown.columns]]
    st.dataframe(table, hide_index=True, use_container_width=True,
                 column_config={"Website": st.column_config.LinkColumn("Website")})
    mapped = shown[shown["mappable"]]
    if len(mapped):
        fig = go.Figure()
        add_programs(fig, mapped)
        fig.update_layout(mapbox=dict(style=MAP_STYLE, center=MAP_CENTER, zoom=MAP_ZOOM), height=450,
                          margin=dict(l=0, r=0, t=0, b=0), legend=dict(x=0.01, y=0.99))
        st.plotly_chart(fig, use_container_width=True)
    st.caption(f"{(~shown['mappable']).sum()} of the programs listed have no mappable address in the county "
               "(online, PO box, or outside San Diego County).")
    download(table, "program_directory.csv")


def page_about():
    st.header("About this dashboard")
    st.markdown("""
This dashboard brings together the San Diego County adult literacy analysis: where low literacy is concentrated,
how it has changed since 2017, and how well literacy programs, transit and libraries reach the people who need them.

**Geography.** ZIP codes (Census ZCTAs), 114 in San Diego County. ZIP codes with fewer than 1,000 residents or no
Census income data are shown but not rated, because their literacy estimates are unreliable.

**Literacy levels.** Level 1 rates are validated 2023 Council data. Level 2 and Level 3 rates are predicted from ACS
data with county-level PIAAC regression models, then scaled so the three levels add up to 100%.

**Hotspots.** High need = top 25% of Level 1 rates. Priority hotspots are high-need ZIP codes that are part of a
statistically significant Getis-Ord Gi* cluster (6 nearest neighbors, 9,999 permutations, p < 0.05).

**Accessibility.** Door-to-door weekday midday travel time from each ZIP code's center to the nearest program by
walking and MTS transit, with up to one transfer and average waiting times. NCTD service is not included.

**Service gaps.** Gap severity (0–100) combines adults below Level 3 per program within 10 km with an accessibility
score (40% transit time, 30% distance, 30% programs within 10 km). Calculated with all programs and with
Council-affiliated programs only.

**Micro-Area Literacy Score.** 0–100, higher = greater need and priority:
- Need severity (50%): literacy level index (1.0 × Level 1 + 0.5 × Level 2) and demographic risk
  (no high school diploma, limited English, poverty, unemployment).
- Service access gap (30%): program access, program supply vs. demand, library proximity and library usage.
- Trend direction (20%): 2017–2023 change and yearly trend in the Level 1 rate.

Affiliated and independent programs count equally in the main score; a second score uses Council-affiliated
programs only, to show where independent programs matter most.

**Limitations.** Level 2 and 3 rates and ZIP-level estimates are modeled and have no published margins of error.
Population counts include children. Transit covers MTS only. Distances are straight-line. Program capacity is not
known, so every program counts once.

**Methodology documents:** Literacy_Hotspot_Methodology, Literacy_ServiceGap_Methodology,
Literacy_Score_LiteratureReview and Literacy_Score_Methodology.

**Data sources:** San Diego Council on Literacy Level 1 data (2017–2023); U.S. Census Bureau ACS 5-year estimates
and 2020 ZCTA boundaries; PIAAC U.S. Skills Map county estimates; MTS GTFS feed; IMLS Public Libraries Survey FY2024;
literacy program inventory compiled for this project.
""")
    st.subheader("Download the data")
    for name, table in [("Micro-Area Literacy Scores", zips.drop(columns=["label", "poverty_group", "english_group"])),
                        ("Service gap rankings", rankings), ("Transit accessibility", access),
                        ("Level 1 rates 2017–2023", level1), ("Program directory", programs)]:
        download(table, name.lower().replace(" ", "_").replace("–", "_") + ".csv", f"Download {name} (CSV)")


# ---------------------------------------------------------------- navigation

PAGES = {
    "Hotspot map": page_hotspots,
    "Literacy rates by level": page_levels,
    "Trends over time": page_trends,
    "Accessibility and transit": page_accessibility,
    "Service gap rankings": page_gaps,
    "Micro-Area Literacy Score": page_score,
    "Program directory": page_directory,
    "About and methodology": page_about,
}

st.sidebar.title("📚 SD Literacy Dashboard")
page = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")
st.sidebar.divider()
PAGES[page]()
