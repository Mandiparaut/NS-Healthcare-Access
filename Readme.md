# NS Healthcare Access — NS Health Data Project 

## Healthcare Accessibility for Aging Communities in Nova Scotia

## Research Question
**Which Nova Scotian communities have older populations that live far from hospitals?**

## Project Overview
This project explores healthcare accessibility gaps for elderly Nova Scotians by combining geographic, health facility, and demographic data. It calculates distances from each community cluster to its nearest hospital and estimates how many residents aged 65+ live in each cluster.

---

## Repository Structure
```
NS_HealthCare_Access_v2/
├── raw_data/                         # Source datasets (not tracked)
│   ├── Hospitals_20260217.csv
│   ├── Nova_Scotia_Community_Clusters_20260217.csv
│   ├── 98-401-X2021018_English_CSV_data.csv
│   └── 98-401-X2021006_English_CSV_data_Atlantic.csv
├── cleaned_data/                     # Processed output
│   └── merged_clean.csv
├── lcsd000b21a_e/                    # Statistics Canada CSD boundary shapefile
├── lda_000b21a_e/                    # Statistics Canada DA boundary shapefile
├── data_cleaning_convertedl.py       # Main analysis script (auto-converted from notebook)
├── data_cleaning_merging_final.ipynb # Notebook version of the analysis
├── streamlit_app.py                  # Streamlit dashboard template
├── app.py                            # (Optional) supporting script / entry point
├── NS_Healthcare_Access_Report_2.0.docx
└── Readme.md
```

---

## Data Sources

### 1. Nova Scotia Open Data Portal
- **Community Clusters**: Geographic boundaries of NS communities
  - Link: https://data.novascotia.ca/Health-and-Wellness/Nova-Scotia-Community-Clusters/2rfn-4b6m/about_data
- **Hospitals**: Hospital locations across Nova Scotia
  - Link: https://data.novascotia.ca/Health-and-Wellness/Hospitals/tmfr-3h8a/about_data

### 2. Statistics Canada
- **2021 Census of Population (CSD)** — Census subdivisions (CSDs) for Nova Scotia
  - Link: https://www12.statcan.gc.ca/census-recensement/2021/dp-pd/prof/details/download-telecharger.cfm?Lang=E
  - File: `98-401-X2021018_English_CSV_data.csv`
- **2021 Census of Population (DA)** — Dissemination Areas for Atlantic Canada
  - File: `98-401-X2021006_English_CSV_data_Atlantic.csv`

---

## Merged Dataset Columns
The primary output is `cleaned_data/merged_clean.csv`.

| Column | Description |
|--------|-------------|
| Cluster | Community cluster name |
| ClusterID | Unique cluster ID |
| latitude | Cluster latitude coordinate |
| longitude | Cluster longitude coordinate |
| Census_CSD | Matched census subdivision name |
| Population_65_plus | Estimated population aged 65+ in the cluster |
| Total_pop_est | Estimated total population in the cluster |
| pct_65_plus | Share of population aged 65+ |
| Pop_source | How the population estimate was derived (e.g., `CSD_exact`, `DA_weighted`)
| Nearest_Hospital | Name of the closest hospital |
| Distance_km | Straight-line distance to nearest hospital (km) |

---

## Analysis Workflow
The analysis combines three main datasets:
1. **Hospitals** — point locations of Nova Scotia hospitals.
2. **Community clusters** — polygon boundaries for 54 health planning zones.
3. **Census population** — 65+ population from both Census Subdivision (CSD) and Dissemination Area (DA) sources.

The workflow:
- Load and clean raw inputs.
- Assign each cluster a representative point (from the cluster polygon).
- Calculate the straight-line distance from each cluster to the nearest hospital.
- Estimate 65+ population for each cluster using a hybrid approach:
  - Exact CSD counts where a cluster maps uniquely to one CSD.
  - DA area-weighted estimates where multiple clusters share a CSD.

---

## How to Run
### 1) Python script (recommended)
1. Activate the Python environment (example using this repo's venv):
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. Install required packages (if not already installed):
   ```powershell
   python -m pip install -r requirements.txt
   ```
   > If `requirements.txt` is not present, install at least: `geopandas pandas numpy scikit-learn folium matplotlib`
3. Run the analysis script:
   ```powershell
   python data_cleaning_convertedl.py
   ```
4. The output will be written to `cleaned_data/merged_clean.csv`.

### 2) Notebook
1. Open `data_cleaning_merging_final.ipynb` in VS Code or Jupyter.
2. Ensure the raw data files are present in `raw_data/`.
3. Run all cells.

### 3) Streamlit dashboard (optional)
1. Run:
   ```powershell
   streamlit run streamlit_app.py
   ```
2. Upload `cleaned_data/merged_clean.csv` to explore the results.

---

## Notes
- The analysis uses straight-line (Haversine) distance; actual travel distances will vary.
- Population estimates for clusters within a shared CSD are derived using area weighting with census Dissemination Areas.

---

## Team 6
MBAN 5510 — March 2026
