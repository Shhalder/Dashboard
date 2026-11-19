# San Diego County Literacy Dashboard

Streamlit dashboard for the San Diego County adult literacy analysis: hotspots, literacy rates by level,
2017–2023 trends, transit accessibility, service gap rankings, the Micro-Area Literacy Score, a program
directory and a methodology page. Every page has a CSV download of the data shown.

## Data files (put these in the `data/` folder)

| File | Made in |
|---|---|
| Literacy_Hotspot_ZIPs.geojson | Literacy_Hotspot_Maps.ipynb (Literacy_Hotspot_Maps_Export/GIS/) |
| Literacy_MicroArea_Scores.csv | Micro-Area Literacy Score notebook |
| Literacy_ServiceGap_Rankings.csv | Literacy_ServiceGap_Visualizations.ipynb |
| Literacy_Hotspot_Analysis.csv | Hotspot analysis cells |
| Literacy_Transit_Accessibility_Scores.csv | Accessibility notebook |
| Level1_Validated_2017_2023_1.csv | Council data |
| Programs_final.csv | Program geocoding |
| MTS_GTFS_processed.csv | MTS GTFS notebook |

## Run locally
    pip install -r requirements.txt
    streamlit run app.py

## Deploy (Streamlit Community Cloud)
1. Create a GitHub repository and upload app.py, requirements.txt, README.md and the data/ folder.
2. Go to https://share.streamlit.io, sign in with GitHub, click "Create app", choose the repository,
   branch `main` and main file `app.py`, then "Deploy".
3. The app gets a public URL ending in `.streamlit.app`.
