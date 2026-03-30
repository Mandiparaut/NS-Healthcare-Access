"""
NS Healthcare Access — Elderly Population & Hospital Distance Analysis
MBAN 5510 — Team 6 — March 2026

Research Question:
  Which Nova Scotian communities have older populations that live far from hospitals?

Pipeline:
  1. Load raw data (hospitals, clusters, CSD census, DA census)
  2. Establish provincial 65+ benchmark
  3. Clean hospitals to acute care only
  4. Build cluster polygon GeoDataFrame and compute centroids
  5. CSD spatial join — assign each cluster to its CSD
  6. Identify shared vs unique CSDs
  7. Method A — exact CSD population for 26 unique clusters
  8. Method B — DA area-weighted population for 28 shared-CSD clusters
  9. Combine into one 54-row dataset
  10. Nearest-hospital distance (BallTree Haversine)
  11. Validate and save merged_clean.csv
  12. Descriptive statistics
  13. Vulnerability scoring
  14. Interactive map (HTML)
  15. Summary table for report

Files required (place in the paths shown):
  raw_data/Hospitals_20260217.csv
  raw_data/Nova_Scotia_Community_Clusters_20260217.csv
  raw_data/98-401-X2021018_English_CSV_data.csv
  raw_data/98-401-X2021006_English_CSV_data_Atlantic.csv   <- 2.2 GB, download from StatCan GEONO=006
  lcsd000b21a_e/lcsd000b21a_e.shp
  lda_000b21a_e/lda_000b21a_e.shp
"""

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — CONSTANTS
# All "magic numbers" that appear throughout the pipeline are defined here
# so they are visible, documented, and easy to update in one place.
# ══════════════════════════════════════════════════════════════════════════════

# ── File paths ────────────────────────────────────────────────────────────────
HOSPITALS_FILE   = 'raw_data/Hospitals_20260217.csv'
CLUSTERS_FILE    = 'raw_data/Nova_Scotia_Community_Clusters_20260217.csv'
CSD_CENSUS_FILE  = 'raw_data/98-401-X2021018_English_CSV_data.csv'
DA_CENSUS_FILE   = 'raw_data/98-401-X2021006_English_CSV_data_Atlantic.csv'
CSD_SHP_PATH     = 'lcsd000b21a_e/lcsd000b21a_e.shp'
DA_SHP_PATH      = 'lda_000b21a_e/lda_000b21a_e.shp'
OUTPUT_DIR       = 'cleaned_data'

# ── Statistics Canada geography codes ────────────────────────────────────────
NS_PRUID         = '12'          # Nova Scotia province code in StatCan shapefiles
NS_DA_DGUID      = '2021S051212' # DGUID prefix for all NS Dissemination Areas
                                 #   2021  = census year
                                 #   S0512 = DA geographic level code
                                 #   12    = NS province code

# ── DA census file location ───────────────────────────────────────────────────
# NS DA rows begin at line 4,975,223 in the Atlantic file (from the geo index).
# Everything before this line is Canada, province, NL, and PEI data we do not need.
NS_DA_START_LINE = 4_975_223

# ── Statistics Canada CHARACTERISTIC_IDs used ────────────────────────────────
CHAR_TOTAL_POP   = 1    # Total population, all ages
CHAR_POP_65_PLUS = 24   # Population aged 65 and over (unconditional count)
CHAR_PCT_65_PLUS = 37   # 65+ as a percentage of total population

# ── Geographic projection ─────────────────────────────────────────────────────
CRS_LATLON       = 'EPSG:4326'  # Standard lat/lon (WGS 84)
CRS_STATSCAN     = 'EPSG:3347'  # Statistics Canada Lambert Conformal Conic
                                 # — accurate for Canadian distance and area

# ── Hospital type filter ──────────────────────────────────────────────────────
ACUTE_TYPES = {'Community', 'Regional', 'Community Health Centre', 'Tertiary'}

# ── Vulnerability scoring ─────────────────────────────────────────────────────
WEIGHT_DISTANCE  = 0.5   # weight given to normalised distance in composite score
WEIGHT_AGE_SHARE = 0.5   # weight given to normalised 65+ share

# ── Earth radius for Haversine ────────────────────────────────────────────────
EARTH_RADIUS_KM  = 6371.0

# ── Checkpoint filenames (written after each expensive step) ──────────────────
# If the script is interrupted, restarting skips steps whose checkpoint exists.
CHECKPOINT_DA_CENSUS  = 'cleaned_data/_checkpoint_da_census.parquet'
CHECKPOINT_MERGED_POP = 'cleaned_data/_checkpoint_cluster_pop.parquet'

# ── Known problem CSD names that need explicit lookup-table correction ─────────
# These communities exist as BOTH a Town and a Municipal district in the census,
# so strip_admin_suffix alone cannot resolve them uniquely. The regex strips
# correctly for all other communities.
CSD_NAME_OVERRIDES = {
    'Digby, Town':              'Digby',
    'Lunenburg, Town':          'Lunenburg',
    'Shelburne, Town':          'Shelburne',
    'Yarmouth, Town':           'Yarmouth',
}


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import subprocess, sys

packages = ['geopandas', 'shapely', 'scikit-learn', 'pyogrio', 'folium']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

import os
import re
import time
import warnings
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import folium
import branca.colormap as cm
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
os.makedirs(OUTPUT_DIR, exist_ok=True)
print('All libraries loaded successfully.')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — LOAD RAW DATA
# ══════════════════════════════════════════════════════════════════════════════

# ── 2.1 Hospitals and clusters ────────────────────────────────────────────────
def _require_file(path):
    """Raise a clear FileNotFoundError if a required input file is missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Check the file paths in SECTION 0 — CONSTANTS."
        )

_require_file(HOSPITALS_FILE)
_require_file(CLUSTERS_FILE)
_require_file(CSD_CENSUS_FILE)
_require_file(DA_CENSUS_FILE)
_require_file(CSD_SHP_PATH)
_require_file(DA_SHP_PATH)

hospitals = pd.read_csv(HOSPITALS_FILE)
clusters  = pd.read_csv(CLUSTERS_FILE)

print(f'Hospitals loaded : {hospitals.shape[0]} rows')
print(f'Clusters loaded  : {clusters.shape[0]} rows')
print()
print('Hospital facility types in raw data:')
print(hospitals['TYPE'].value_counts().to_string())

# ── Intermediate assertion: confirm expected row counts ───────────────────────
assert hospitals.shape[0] == 48, \
    f"Expected 48 hospital rows, got {hospitals.shape[0]}. Has the source file changed?"
assert clusters.shape[0]  == 54, \
    f"Expected 54 cluster rows, got {clusters.shape[0]}. Has the source file changed?"


# ── 2.2 NS CSD census extract ─────────────────────────────────────────────────
# Contains every census characteristic for every NS Census Subdivision.
# Used in Section 7 (exact 65+ counts for 26 unique clusters) and Section 2b
# (provincial 65+ benchmark).

census_csd = pd.read_csv(CSD_CENSUS_FILE, encoding='latin1', low_memory=False)
print(f'\nNS CSD census loaded : {census_csd.shape[0]:,} rows, {census_csd.shape[1]} columns')

assert census_csd.shape[0] > 0, "CSD census file loaded but is empty — check the file."


# ── 2.3 Atlantic DA census file ───────────────────────────────────────────────
# This 2.2 GB file contains DA-level data for all of Atlantic Canada.
# IMPORTANT: Instead of building a list of ~5 million row indices to skip
# (which itself consumes significant memory and time), we use skiprows with
# a callable that rejects rows before the NS DA section, then filter by DGUID.
# This avoids materialising the full skip-list in memory.

if os.path.exists(CHECKPOINT_DA_CENSUS):
    print(f'\nCheckpoint found — loading DA census from {CHECKPOINT_DA_CENSUS}')
    print('(Delete this file to force a fresh load from the raw Atlantic CSV.)')
    da_census = pd.read_parquet(CHECKPOINT_DA_CENSUS)
    print(f'NS DA rows loaded from checkpoint : {len(da_census):,}')
else:
    print(f'\nLoading NS DA rows from Atlantic file...')
    print(f'Rows before line {NS_DA_START_LINE:,} will be skipped.')
    print('Expected time: 2-5 minutes on first run.')
    t0 = time.time()

    # skiprows as a callable: return True (skip) for rows before the NS DA section.
    # Row 0 is always kept (the header). Rows 1 .. NS_DA_START_LINE-2 are skipped.
    # This avoids building a list of 4.97 million integers in memory.
    skip_fn = lambda i: 0 < i < (NS_DA_START_LINE - 1)

    da_census = pd.read_csv(
        DA_CENSUS_FILE,
        encoding='latin1',
        low_memory=False,
        skiprows=skip_fn,
    )

    # Filter to confirmed NS DA rows by DGUID prefix
    da_census = da_census[
        da_census['DGUID'].astype(str).str.startswith(NS_DA_DGUID)
    ].reset_index(drop=True)

    elapsed = time.time() - t0
    print(f'NS DA rows loaded : {len(da_census):,}  ({elapsed:.0f}s)')
    print(f'Unique NS DAs     : {da_census["DGUID"].nunique():,}')

    # Save checkpoint so subsequent runs are instant
    da_census.to_parquet(CHECKPOINT_DA_CENSUS, index=False)
    print(f'Checkpoint saved  : {CHECKPOINT_DA_CENSUS}')

assert da_census['DGUID'].nunique() == 1670, (
    f"Expected 1,670 unique NS DAs, got {da_census['DGUID'].nunique()}. "
    "Delete the checkpoint and reload if the source file was updated."
)


# ── 2.4 Extract DA population characteristics ─────────────────────────────────

def extract_da_char(df, char_id, col_name):
    """
    Pull one census characteristic for all DAs into a DAUID → value table.

    Parameters
    ----------
    df       : DataFrame — full DA census rows (NS only)
    char_id  : int       — CHARACTERISTIC_ID to extract
    col_name : str       — name to give the value column in the output

    Returns
    -------
    DataFrame with columns ['DAUID', col_name]
    DAUID is the 8-digit DA identifier (last 8 characters of DGUID).
    Suppressed values ('x', blank) become NaN after pd.to_numeric.
    """
    out = df[df['CHARACTERISTIC_ID'] == char_id][['DGUID', 'C1_COUNT_TOTAL']].copy()
    out['value'] = pd.to_numeric(out['C1_COUNT_TOTAL'], errors='coerce')
    out['DAUID'] = out['DGUID'].astype(str).str[-8:]
    return out[['DAUID', 'value']].rename(columns={'value': col_name})

da_pop_total = extract_da_char(da_census, CHAR_TOTAL_POP,   'Total_pop')
da_pop_65    = extract_da_char(da_census, CHAR_POP_65_PLUS, 'Pop_65_plus')

da_65_valid      = da_pop_65['Pop_65_plus'].notna().sum()
da_65_suppressed = da_pop_65['Pop_65_plus'].isna().sum()

print()
print(f'NS DAs with valid 65+ count      : {da_65_valid:,}')
print(f'NS DAs with suppressed 65+ count : {da_65_suppressed:,} (treated as 0 downstream)')
print(f'NS 65+ population across all DAs : {da_pop_65["Pop_65_plus"].sum():,.0f}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2b — PROVINCIAL 65+ BENCHMARK VALIDATION
# Compare CSD extract total vs DA file total. Both come from the 2021 Census
# and should agree within a small suppression-related gap (< 10%).
# ══════════════════════════════════════════════════════════════════════════════

NS_65_BENCHMARK = (
    census_csd[
        (census_csd['CHARACTERISTIC_ID'] == CHAR_POP_65_PLUS) &
        (census_csd['GEO_LEVEL'] == 'Census subdivision')
    ]['C1_COUNT_TOTAL']
    .apply(pd.to_numeric, errors='coerce')
    .sum()
)

da_65_total = da_pop_65['Pop_65_plus'].sum()
diff        = abs(NS_65_BENCHMARK - da_65_total)
diff_pct    = diff / NS_65_BENCHMARK * 100

print('\n=== PROVINCIAL 65+ POPULATION: CSD vs DA ===')
print(f'  CSD extract total (NS_65_BENCHMARK) : {NS_65_BENCHMARK:,.0f}')
print(f'  DA file total (valid DAs only)       : {da_65_total:,.0f}')
print(f'  Difference                           : {diff:,.0f}  ({diff_pct:.1f}%)')

if diff_pct < 2:
    print('  RESULT: Files consistent (< 2% gap). Proceeding.')
elif diff_pct < 10:
    print(f'  RESULT: Small gap of {diff_pct:.1f}% — expected due to suppressed DA values.')
else:
    raise RuntimeError(
        f"CSD vs DA gap is {diff_pct:.1f}% — larger than the 10% threshold.\n"
        "Investigate before proceeding: check DA file loading and DGUID filter."
    )

# The gap at > 10% now raises an exception rather than printing a warning
# that could be scrolled past. A hard stop prevents silent downstream errors.

assert NS_65_BENCHMARK > 0, "NS_65_BENCHMARK is zero — CSD census file may not have loaded correctly."


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CLEAN HOSPITAL DATA
# ══════════════════════════════════════════════════════════════════════════════

hospitals['longitude'] = pd.to_numeric(
    hospitals['the_geom'].str.extract(r'POINT \((-?[\d.]+)')[0], errors='coerce')
hospitals['latitude']  = pd.to_numeric(
    hospitals['the_geom'].str.extract(r'POINT \(-?[\d.]+ (-?[\d.]+)')[0], errors='coerce')

hospitals_clean = (
    hospitals[['FACILITY', 'TOWN', 'COUNTY', 'TYPE', 'latitude', 'longitude']]
    .rename(columns={'FACILITY': 'Hospital', 'TOWN': 'Town', 'COUNTY': 'County', 'TYPE': 'Type'})
    .dropna(subset=['latitude', 'longitude'])
    .loc[lambda df: df['Type'].isin(ACUTE_TYPES)]
    .reset_index(drop=True)
)

print(f'Hospitals before filter : {len(hospitals)}')
print(f'Hospitals after filter  : {len(hospitals_clean)} acute care facilities')
print()
print('By type:')
print(hospitals_clean['Type'].value_counts().to_string())

assert len(hospitals_clean) == 43, (
    f"Expected 43 acute care hospitals after filtering, got {len(hospitals_clean)}. "
    "Check that ACUTE_TYPES matches the source file TYPE values."
)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — LOAD CLUSTER POLYGONS AND COMPUTE CENTROIDS
# ══════════════════════════════════════════════════════════════════════════════

clusters_gdf = gpd.GeoDataFrame(
    clusters[['Cluster', 'ClusterID']],
    geometry=gpd.GeoSeries.from_wkt(clusters['the_geom']),
    crs=CRS_LATLON
)
clusters_gdf = clusters_gdf.to_crs(CRS_STATSCAN)
clusters_gdf['centroid'] = clusters_gdf['geometry'].centroid

centroids_4326             = clusters_gdf['centroid'].to_crs(CRS_LATLON)
clusters_gdf['latitude']   = centroids_4326.y
clusters_gdf['longitude']  = centroids_4326.x
clusters_gdf = clusters_gdf.dropna(subset=['latitude', 'longitude']).reset_index(drop=True)

assert len(clusters_gdf) == 54, (
    f"Expected 54 cluster polygons, got {len(clusters_gdf)}. "
    "Check the cluster CSV for missing or malformed geometry."
)
print(f'Community clusters loaded: {len(clusters_gdf)}')
print(clusters_gdf[['Cluster', 'latitude', 'longitude']].head(8))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — CSD SPATIAL JOIN
# Match each cluster centroid to the CSD it falls inside.
# Three clusters whose centroids land offshore fall back to nearest-CSD.
# ══════════════════════════════════════════════════════════════════════════════

csd_gdf = gpd.read_file(CSD_SHP_PATH)
csd_gdf = csd_gdf[csd_gdf['PRUID'] == NS_PRUID].reset_index(drop=True)
csd_gdf = csd_gdf.to_crs(CRS_STATSCAN)

assert len(csd_gdf) == 95, (
    f"Expected 95 NS CSD polygons, got {len(csd_gdf)}. "
    "Check that the shapefile is the NS-only extract and PRUID filter is correct."
)
print(f'NS CSD polygons loaded: {len(csd_gdf)}')

clusters_pts             = clusters_gdf.copy()
clusters_pts['geometry'] = clusters_pts['centroid']

joined_csd = gpd.sjoin(
    clusters_pts[['Cluster', 'ClusterID', 'latitude', 'longitude', 'geometry']],
    csd_gdf[['CSDNAME', 'geometry']],
    how='left',
    predicate='within'
).drop(columns=['index_right'], errors='ignore')

n_matched = joined_csd['CSDNAME'].notna().sum()
n_missed  = joined_csd['CSDNAME'].isna().sum()
print(f'\nSpatial join: {n_matched} matched, {n_missed} unmatched (centroid offshore)')

if n_missed > 0:
    unmatched = joined_csd[joined_csd['CSDNAME'].isna()]['Cluster'].tolist()
    print(f'Applying nearest-CSD fallback for: {unmatched}')
    nearest = gpd.sjoin_nearest(
        clusters_pts[clusters_pts['Cluster'].isin(unmatched)][['Cluster', 'geometry']],
        csd_gdf[['CSDNAME', 'geometry']],
        how='left'
    )[['Cluster', 'CSDNAME']].drop_duplicates('Cluster')
    for _, row in nearest.iterrows():
        joined_csd.loc[joined_csd['Cluster'] == row['Cluster'], 'CSDNAME'] = row['CSDNAME']
        print(f'  {row["Cluster"]} -> {row["CSDNAME"]}')

assert joined_csd['CSDNAME'].notna().all(), (
    "Some clusters still have no CSD assignment after the nearest fallback. "
    f"Missing: {joined_csd[joined_csd['CSDNAME'].isna()]['Cluster'].tolist()}"
)
print(f'Final match rate: {joined_csd["CSDNAME"].notna().sum()} / {len(joined_csd)}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — IDENTIFY SHARED VS UNIQUE CSDS
# ══════════════════════════════════════════════════════════════════════════════

csd_cluster_counts = joined_csd.groupby('CSDNAME')['Cluster'].count()
shared_csds     = set(csd_cluster_counts[csd_cluster_counts > 1].index)
unique_csds     = set(csd_cluster_counts[csd_cluster_counts == 1].index)
shared_clusters = joined_csd[joined_csd['CSDNAME'].isin(shared_csds)]['Cluster'].tolist()
unique_clusters = joined_csd[joined_csd['CSDNAME'].isin(unique_csds)]['Cluster'].tolist()

assert len(shared_clusters) + len(unique_clusters) == 54, (
    f"Shared ({len(shared_clusters)}) + unique ({len(unique_clusters)}) != 54. "
    "Check the CSD spatial join for duplicate rows."
)

print('=== POPULATION METHOD ASSIGNMENT ===')
print(f'  Method A — CSD exact   : {len(unique_clusters)} clusters (one cluster per CSD)')
print(f'  Method B — DA weighted : {len(shared_clusters)} clusters (multiple clusters share a CSD)')
print()
print('CSDs shared by multiple clusters:')
for csd in sorted(shared_csds):
    clust_list = joined_csd[joined_csd['CSDNAME'] == csd]['Cluster'].tolist()
    print(f'  {csd} ({len(clust_list)} clusters):')
    for c in clust_list:
        print(f'    - {c}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — METHOD A: EXACT CSD POPULATION (26 CLUSTERS)
# ══════════════════════════════════════════════════════════════════════════════

def get_csd_char(char_id):
    """
    Extract one census characteristic for all NS CSDs.
    GEO_LEVEL filter keeps only Census subdivision rows,
    excluding the province-level summary row.
    """
    return (
        census_csd[
            (census_csd['CHARACTERISTIC_ID'] == char_id) &
            (census_csd['GEO_LEVEL'] == 'Census subdivision')
        ][['GEO_NAME', 'C1_COUNT_TOTAL']]
        .copy()
        .rename(columns={'C1_COUNT_TOTAL': f'char_{char_id}'})
    )

char1  = get_csd_char(CHAR_TOTAL_POP)
char24 = get_csd_char(CHAR_POP_65_PLUS)
char37 = get_csd_char(CHAR_PCT_65_PLUS)

csd_raw = (
    char1
    .merge(char24, on='GEO_NAME', how='left')
    .merge(char37, on='GEO_NAME', how='left')
)
for col in [f'char_{CHAR_TOTAL_POP}', f'char_{CHAR_POP_65_PLUS}', f'char_{CHAR_PCT_65_PLUS}']:
    csd_raw[col] = pd.to_numeric(csd_raw[col], errors='coerce')


def calc_pop_65_csd(row):
    """
    Return (population_65_plus, source_label) for one CSD row.
    Uses CHAR_POP_65_PLUS (ID 24) directly when available.
    Falls back to CHAR_TOTAL_POP × CHAR_PCT_65_PLUS / 100 when ID 24 is suppressed.
    Returns (None, 'unavailable') when neither is possible.
    """
    c1  = f'char_{CHAR_TOTAL_POP}'
    c24 = f'char_{CHAR_POP_65_PLUS}'
    c37 = f'char_{CHAR_PCT_65_PLUS}'
    if pd.notna(row[c24]):
        return row[c24], 'CSD_exact'
    if pd.notna(row[c1]) and pd.notna(row[c37]):
        return round(row[c1] * row[c37] / 100), 'CSD_reconstructed'
    return None, 'unavailable'

results = csd_raw.apply(calc_pop_65_csd, axis=1, result_type='expand')
csd_raw['Population_65_plus'] = results[0]
csd_raw['Pop_source']         = results[1]


def strip_admin_suffix(name):
    """
    Remove Statistics Canada administrative type labels from CSD names
    so they match the shapefile CSDNAME values.

    Examples:
      'Amherst, Town (T)'                    → 'Amherst'
      'Halifax (Regional municipality)'      → 'Halifax'
      'Cumberland, Subdivision B'            → 'Cumberland, Subd. B'  (unchanged — not a suffix)

    Known edge cases are handled by the CSD_NAME_OVERRIDES lookup table
    defined in SECTION 0, which takes priority over the regex passes.
    """
    # Apply lookup table overrides first
    if name in CSD_NAME_OVERRIDES:
        return CSD_NAME_OVERRIDES[name]

    # Pass 1: remove trailing parenthetical labels e.g. '(Regional municipality)'
    name = re.sub(r'\s*\(.*\)\s*$', '', name).strip()

    # Pass 2: remove ', Subdivision of ...' variants
    name = re.sub(
        r',\s*Subdivision of (county municipality|county|municipality)\s*$',
        '', name, flags=re.IGNORECASE
    ).strip()

    # Pass 3: remove trailing ', Town' / ', Village' / etc.
    name = re.sub(
        r',\s*(Town|Village|County municipality|Municipal district|'
        r'Regional municipality|Rural municipality|Indian reserve)\s*$',
        '', name, flags=re.IGNORECASE
    ).strip()

    return name

csd_raw['Community_clean'] = csd_raw['GEO_NAME'].apply(strip_admin_suffix)

# Aggregate — some communities appear as both Town and Municipal district
csd_pop = (
    csd_raw.groupby('Community_clean')
    .agg(
        Population_65_plus=('Population_65_plus', 'sum'),
        Total_pop         =(f'char_{CHAR_TOTAL_POP}', 'sum'),
        Pop_source        =('Pop_source', lambda x:
                            'CSD_exact' if (x == 'CSD_exact').all()
                            else 'CSD_reconstructed')
    )
    .reset_index()
    .rename(columns={'Community_clean': 'CSDNAME'})
)

unique_merged = (
    joined_csd[joined_csd['Cluster'].isin(unique_clusters)]
    .merge(csd_pop[['CSDNAME', 'Population_65_plus', 'Total_pop', 'Pop_source']],
           on='CSDNAME', how='left')
)
unique_merged['pct_65_plus'] = (
    unique_merged['Population_65_plus'] / unique_merged['Total_pop'] * 100
).round(1)

n_matched_csd = unique_merged['Population_65_plus'].notna().sum()
assert n_matched_csd == len(unique_merged), (
    f"Only {n_matched_csd} of {len(unique_merged)} unique clusters got a CSD population figure. "
    f"Missing: {unique_merged[unique_merged['Population_65_plus'].isna()]['Cluster'].tolist()}\n"
    "The CSD name cleaning may have failed to match these communities."
)
print(f'Unique clusters with CSD population: {n_matched_csd} / {len(unique_merged)}')
print('Population source breakdown:')
print(unique_merged['Pop_source'].value_counts().to_string())


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — METHOD B: DA AREA-WEIGHTED POPULATION (28 CLUSTERS)
# ══════════════════════════════════════════════════════════════════════════════

print('\nLoading DA boundary shapefile...')
da_gdf = gpd.read_file(DA_SHP_PATH)
da_gdf = da_gdf[da_gdf['PRUID'] == NS_PRUID].reset_index(drop=True)
da_gdf = da_gdf.to_crs(CRS_STATSCAN)

assert len(da_gdf) == 1670, (
    f"Expected 1,670 NS DA polygons, got {len(da_gdf)}. "
    "Check that the DA shapefile is the correct NS extract."
)
print(f'NS DA polygons loaded: {len(da_gdf):,}')

# Attach census figures. Suppressed NaN values become 0 (conservative).
da_gdf = da_gdf.merge(da_pop_65,    on='DAUID', how='left')
da_gdf = da_gdf.merge(da_pop_total, on='DAUID', how='left')
da_gdf['Pop_65_plus'] = da_gdf['Pop_65_plus'].fillna(0)
da_gdf['Total_pop']   = da_gdf['Total_pop'].fillna(0)
da_gdf['da_area_m2']  = da_gdf.geometry.area

# Performance optimisation: restrict to DAs within the five shared CSDs only.
# This reduces the intersection from 1,670 DAs to ~988 DAs (~40% faster).
shared_csd_polygons = csd_gdf[csd_gdf['CSDNAME'].isin(shared_csds)]
da_shared = (
    gpd.sjoin(da_gdf, shared_csd_polygons[['CSDNAME', 'geometry']],
              how='inner', predicate='intersects')
    .drop(columns=['index_right'], errors='ignore')
    .drop_duplicates('DAUID')
)

print(f'DAs in shared CSD areas : {len(da_shared)} (vs {len(da_gdf)} total NS DAs)')

assert len(da_shared) > 0, "No DAs found within shared CSD areas — check shared_csds set."

# ── Area-weighted polygon intersection ────────────────────────────────────────
shared_clusters_gdf = clusters_gdf[clusters_gdf['Cluster'].isin(shared_clusters)]

print(f'Running intersection: {len(shared_clusters_gdf)} clusters × {len(da_shared)} DAs')
print('Expected time: under 1 minute')

intersection = gpd.overlay(
    shared_clusters_gdf[['Cluster', 'ClusterID', 'geometry']],
    da_shared[['DAUID', 'Pop_65_plus', 'Total_pop', 'da_area_m2', 'geometry']],
    how='intersection',
    keep_geom_type=True
)

print(f'Fragments created: {len(intersection):,}')
assert len(intersection) > 0, "Intersection produced zero fragments — check CRS alignment."

intersection['frag_area_m2'] = intersection.geometry.area
intersection['weight']       = (
    intersection['frag_area_m2'] / intersection['da_area_m2']
).clip(0, 1)

intersection['pop_65_frag']    = intersection['Pop_65_plus'] * intersection['weight']
intersection['pop_total_frag'] = intersection['Total_pop']   * intersection['weight']

shared_pop = (
    intersection
    .groupby('Cluster')
    .agg(
        Population_65_plus=('pop_65_frag',    'sum'),
        Total_pop         =('pop_total_frag', 'sum')
    )
    .reset_index()
)

shared_pop['Population_65_plus'] = shared_pop['Population_65_plus'].round().astype(int)
shared_pop['Total_pop']          = shared_pop['Total_pop'].round().astype(int)
shared_pop['pct_65_plus']        = (
    shared_pop['Population_65_plus'] / shared_pop['Total_pop'] * 100
).round(1)
shared_pop['Pop_source'] = 'DA_area_weighted'
shared_pop['CSDNAME']    = shared_pop['Cluster'].map(
    joined_csd.set_index('Cluster')['CSDNAME']
)

assert len(shared_pop) == len(shared_clusters), (
    f"Expected {len(shared_clusters)} shared clusters after intersection, "
    f"got {len(shared_pop)}. Some clusters produced no fragments."
)

print('\nDA intersection complete.')
print(shared_pop[['Cluster', 'CSDNAME', 'Population_65_plus', 'pct_65_plus']]
      .sort_values('CSDNAME').to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — COMBINE BOTH METHODS INTO ONE DATASET
# ══════════════════════════════════════════════════════════════════════════════

keep_cols = ['Cluster', 'ClusterID', 'latitude', 'longitude',
             'CSDNAME', 'Population_65_plus', 'Total_pop', 'pct_65_plus', 'Pop_source']

unique_part = unique_merged[keep_cols].copy()

shared_part = (
    joined_csd[joined_csd['Cluster'].isin(shared_clusters)]
    [['Cluster', 'ClusterID', 'latitude', 'longitude']]
    .merge(
        shared_pop[['Cluster', 'Population_65_plus', 'Total_pop',
                    'pct_65_plus', 'Pop_source', 'CSDNAME']],
        on='Cluster'
    )
)[keep_cols]

cluster_pop = pd.concat([unique_part, shared_part], ignore_index=True)
cluster_pop = cluster_pop.rename(columns={'CSDNAME': 'Census_CSD', 'Total_pop': 'Total_pop_est'})

assert len(cluster_pop) == 54, (
    f"Combined dataset has {len(cluster_pop)} rows, expected 54."
)

total_65 = cluster_pop['Population_65_plus'].sum()
coverage = total_65 / NS_65_BENCHMARK

print('=== HYBRID POPULATION SUMMARY ===')
print(f'  Total clusters                  : {len(cluster_pop)}')
for src, grp in cluster_pop.groupby('Pop_source'):
    print(f'  {src:<25}    : {len(grp):>3} clusters')
print()
print(f'  Sum 65+ across all 54 clusters  : {total_65:,}')
print(f'  NS provincial benchmark (CSD)   : {NS_65_BENCHMARK:,.0f}')
print(f'  Coverage ratio                  : {coverage:.2f}x')
print(f'  Gap ({(1-coverage)*100:.1f}%) = communities outside cluster boundaries + suppressed DAs')

# Save checkpoint so distance + scoring steps can be re-run without repeating
# the expensive spatial intersection if only downstream code changes.
cluster_pop.to_parquet(CHECKPOINT_MERGED_POP, index=False)
print(f'\nCheckpoint saved: {CHECKPOINT_MERGED_POP}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — DISTANCE TO NEAREST HOSPITAL (BallTree Haversine)
# ══════════════════════════════════════════════════════════════════════════════

joined = clusters_gdf[['Cluster', 'ClusterID', 'latitude', 'longitude', 'geometry']].merge(
    cluster_pop, on=['Cluster', 'ClusterID', 'latitude', 'longitude'], how='left'
)

missing = joined['Population_65_plus'].isna().sum()
assert missing == 0, (
    f"{missing} clusters have no population data after merge: "
    f"{joined[joined['Population_65_plus'].isna()]['Cluster'].tolist()}"
)
print(f'All {len(joined)} clusters have population data.')

# BallTree requires coordinates in radians
hospital_coords = np.radians(hospitals_clean[['latitude', 'longitude']].astype(float).values)
cluster_coords  = np.radians(joined[['latitude', 'longitude']].astype(float).values)

tree = BallTree(hospital_coords, metric='haversine')
dist_rad, idx = tree.query(cluster_coords, k=1)

joined['Distance_km']      = np.round(dist_rad.flatten() * EARTH_RADIUS_KM, 2)
joined['Nearest_Hospital'] = hospitals_clean.iloc[idx.flatten()]['Hospital'].values

assert (joined['Distance_km'] >= 0).all(), "Negative distances produced — check coordinate order."
assert joined['Distance_km'].notna().all(), "Some clusters have NaN distance."

print('Distances calculated for all clusters.')
print(joined[['Cluster', 'Nearest_Hospital', 'Distance_km']].head(10).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — ASSEMBLE, VALIDATE, AND SAVE FINAL DATASET
# ══════════════════════════════════════════════════════════════════════════════

output_cols = [
    'Cluster', 'ClusterID', 'latitude', 'longitude',
    'Census_CSD', 'Population_65_plus', 'Total_pop_est', 'pct_65_plus', 'Pop_source',
    'Nearest_Hospital', 'Distance_km'
]
final_clean = joined[output_cols].copy()

# ── Hard assertions — any failure stops the script with a clear message ────────
assert len(final_clean)                          == 54,  f"Expected 54 rows, got {len(final_clean)}"
assert final_clean['Distance_km'].notna().all(),          "Some clusters are missing distances."
assert (final_clean['Distance_km'] >= 0).all(),           "Some distances are negative."
assert final_clean['Nearest_Hospital'].notna().all(),     "Some clusters have no hospital assigned."
assert final_clean['Population_65_plus'].notna().all(),   "Some clusters are missing population data."
assert final_clean['pct_65_plus'].between(0, 100).all(), "Some 65+ share values are outside [0, 100]."
assert (final_clean['Total_pop_est'] >= final_clean['Population_65_plus']).all(), \
    "Some clusters have more 65+ residents than total population."

total_65 = final_clean['Population_65_plus'].sum()

print('=== FINAL DATASET VALIDATION ===')
print(f'  Total clusters                  : {len(final_clean)}')
print()
print('  Population source breakdown:')
for src, grp in final_clean.groupby('Pop_source'):
    print(f'    {src:<25}: {len(grp):>3} clusters')
print()
print(f'  Sum 65+ across all clusters     : {total_65:,}')
print(f'  NS provincial benchmark (CSD)   : {NS_65_BENCHMARK:,.0f}')
print(f'  Coverage ratio                  : {total_65/NS_65_BENCHMARK:.2f}x')
print(f'  Min distance (km)               : {final_clean["Distance_km"].min():.2f}')
print(f'  Max distance (km)               : {final_clean["Distance_km"].max():.2f}')
print(f'  Median distance (km)            : {final_clean["Distance_km"].median():.2f}')
print()
print('All validation checks passed.')

final_clean.to_csv(f'{OUTPUT_DIR}/merged_clean.csv', index=False)
print(f'\nSaved: {OUTPUT_DIR}/merged_clean.csv')
print(f'Rows  : {len(final_clean)}')
print(f'Cols  : {list(final_clean.columns)}')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

print('\n=== DISTANCE TO NEAREST HOSPITAL (km) ===')
print(final_clean['Distance_km'].describe().round(2).to_string())
print()

print('=== COMMUNITIES BEYOND DISTANCE THRESHOLDS ===')
print(f'{"Threshold":<12} {"Clusters":>10} {"% of total":>12} {"65+ pop":>12}')
print('-' * 50)
for t in [15, 30, 45]:
    far = final_clean[final_clean['Distance_km'] > t]
    print(f'> {t} km{"": <8} {len(far):>10} '
          f'{len(far)/len(final_clean)*100:>11.1f}% '
          f'{far["Population_65_plus"].sum():>12,}')
print()

print('=== 65+ SHARE DISTRIBUTION ACROSS CLUSTERS ===')
bins   = [0, 15, 20, 25, 30, 100]
labels = ['<15%', '15-20%', '20-25%', '25-30%', '>30%']
final_clean['share_band'] = pd.cut(final_clean['pct_65_plus'], bins=bins, labels=labels)
for band, n in final_clean['share_band'].value_counts().sort_index().items():
    print(f'  {band:<8} {chr(9608)*n}  ({n} clusters)')
print()

print('=== TOP 10 FARTHEST COMMUNITIES ===')
print(final_clean[['Cluster', 'Nearest_Hospital', 'Distance_km', 'pct_65_plus', 'Pop_source']]
      .sort_values('Distance_km', ascending=False).head(10).to_string(index=False))
print()

print('=== TOP 10 HIGHEST 65+ SHARE ===')
print(final_clean[['Cluster', 'Distance_km', 'pct_65_plus', 'Population_65_plus', 'Pop_source']]
      .sort_values('pct_65_plus', ascending=False).head(10).to_string(index=False))


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — VULNERABILITY SCORING
# ══════════════════════════════════════════════════════════════════════════════
#
# vulnerability_score = WEIGHT_DISTANCE  × normalised_distance
#                     + WEIGHT_AGE_SHARE × normalised_65_share
#
# Both dimensions are normalised to [0, 1] via MinMaxScaler so they contribute
# equally regardless of their original units.
# Weights are defined as constants in SECTION 0 for easy sensitivity testing.

scaler = MinMaxScaler()

scored = final_clean[[
    'Cluster', 'Distance_km', 'pct_65_plus', 'Population_65_plus',
    'Total_pop_est', 'Nearest_Hospital', 'latitude', 'longitude', 'Pop_source'
]].copy()

scored['dist_norm']  = scaler.fit_transform(scored[['Distance_km']])
scored['share_norm'] = scaler.fit_transform(scored[['pct_65_plus']])
scored['vulnerability_score'] = (
    WEIGHT_DISTANCE  * scored['dist_norm'] +
    WEIGHT_AGE_SHARE * scored['share_norm']
).round(3)

scored = scored.sort_values('vulnerability_score', ascending=False).reset_index(drop=True)
scored.index += 1

dist_median  = final_clean['Distance_km'].median()
share_median = final_clean['pct_65_plus'].median()

vulnerable = final_clean[
    (final_clean['Distance_km'] > dist_median) &
    (final_clean['pct_65_plus'] > share_median)
].sort_values('Distance_km', ascending=False)

print('=== TOP 20 BY VULNERABILITY SCORE ===')
print(f'{"Rank":<5} {"Cluster":<42} {"Dist km":>8} {"65+%":>6} {"Score":>7} {"Source"}')
print('-' * 80)
for rank, row in scored.head(20).iterrows():
    print(f'{rank:<5} {row["Cluster"]:<42} {row["Distance_km"]:>7.1f} '
          f'{row["pct_65_plus"]:>5.1f}% {row["vulnerability_score"]:>7.3f}'
          f'  {row["Pop_source"]}')
print()
print('=== VULNERABLE COMMUNITIES ===')
print(f'Above-median distance (>{dist_median:.1f} km) AND above-median 65+ share (>{share_median:.1f}%)')
print(f'Total: {len(vulnerable)} of {len(final_clean)} clusters')
print()
print(vulnerable[['Cluster', 'Nearest_Hospital', 'Distance_km', 'pct_65_plus',
                   'Population_65_plus', 'Pop_source']].to_string(index=False))

scored.to_csv(f'{OUTPUT_DIR}/vulnerability_scored.csv', index_label='Rank')
print(f'\nSaved: {OUTPUT_DIR}/vulnerability_scored.csv')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 13b — SENSITIVITY ANALYSIS: VULNERABILITY SCORE WEIGHTING
#
# The 50/50 weighting is a methodological assumption. This section tests two
# alternative weightings and shows how the top 10 ranking changes.
#
# Scenarios:
#   Distance-led  (70/30) — prioritises geographic isolation
#   Balanced      (50/50) — equal weight (primary result)
#   Age-led       (30/70) — prioritises elderly share
# ══════════════════════════════════════════════════════════════════════════════

def run_scoring(df, w_dist, w_age, label):
    """
    Score all 54 clusters under a given weighting and return a ranked DataFrame.

    Parameters
    ----------
    df     : final_clean DataFrame
    w_dist : float — weight for normalised distance (0 to 1)
    w_age  : float — weight for normalised 65+ share (0 to 1)
    label  : str   — scenario name for display

    Returns
    -------
    DataFrame with columns: Cluster, Distance_km, pct_65_plus, score_{label}
    Indexed from 1 (rank 1 = most vulnerable).
    """
    assert abs(w_dist + w_age - 1.0) < 1e-9, "Weights must sum to 1.0"

    s = MinMaxScaler()
    temp = df[['Cluster', 'Distance_km', 'pct_65_plus']].copy()
    temp['dist_norm']  = s.fit_transform(temp[['Distance_km']])
    temp['share_norm'] = s.fit_transform(temp[['pct_65_plus']])
    temp['score']      = (w_dist * temp['dist_norm'] + w_age * temp['share_norm']).round(3)
    temp = temp.sort_values('score', ascending=False).reset_index(drop=True)
    temp.index += 1
    temp.index.name = 'Rank'
    return temp[['Cluster', 'score']].rename(columns={'score': f'score_{label}'})


# Run all three scenarios
scenarios = [
    (0.70, 0.30, '70_30'),
    (0.50, 0.50, '50_50'),
    (0.30, 0.70, '30_70'),
]

ranked = {}
for w_dist, w_age, label in scenarios:
    ranked[label] = run_scoring(final_clean, w_dist, w_age, label)

# Build comparison table — top 10 under each scenario
# Columns: Rank | Community (50/50) | Score | Rank under 70/30 | Rank under 30/70
base = ranked['50_50'].head(10).reset_index()   # baseline ranking
r7030 = ranked['70_30'].reset_index().rename(columns={'Rank': 'rank_70_30'})
r3070 = ranked['30_70'].reset_index().rename(columns={'Rank': 'rank_30_70'})

comparison = (
    base
    .merge(r7030[['Cluster', 'rank_70_30', 'score_70_30']], on='Cluster', how='left')
    .merge(r3070[['Cluster', 'rank_30_70', 'score_30_70']], on='Cluster', how='left')
)

# Movement column: how many places does the rank shift under each alternative?
comparison['move_70_30'] = comparison['Rank'] - comparison['rank_70_30']
comparison['move_30_70'] = comparison['Rank'] - comparison['rank_30_70']

def fmt_move(x):
    """Format rank movement with direction arrow."""
    if pd.isna(x) or x == 0:
        return '—'
    return f'▲{int(x)}' if x > 0 else f'▼{abs(int(x))}'

print('=== SENSITIVITY ANALYSIS — VULNERABILITY SCORE WEIGHTING ===')
print()
print(f'Baseline (50/50): distance weight = 0.50, elderly share weight = 0.50')
print()
print(f'{"Rank":<5} {"Community":<42} '
      f'{"Score":>7}  '
      f'{"70/30 rank":>10} {"Shift":>6}  '
      f'{"50/50 rank":>10}  '
      f'{"30/70 rank":>10} {"Shift":>6}')
print('-' * 100)

for _, row in comparison.iterrows():
    print(f'{int(row["Rank"]):<5} {row["Cluster"]:<42} '
          f'{row["score_50_50"]:>7.3f}  '
          f'{int(row["rank_70_30"]):>10} {fmt_move(row["move_70_30"]):>6}  '
          f'{int(row["Rank"]):>10}  '
          f'{int(row["rank_30_70"]):>10} {fmt_move(row["move_30_70"]):>6}')

print()
print('▲ = ranks higher (less vulnerable) under that weighting')
print('▼ = ranks lower (more vulnerable) under that weighting')

# ── Check stability of top 5 ──────────────────────────────────────────────────
top5_base = set(ranked['50_50'].head(5)['Cluster'])
top5_7030 = set(ranked['70_30'].head(5)['Cluster'])
top5_3070 = set(ranked['30_70'].head(5)['Cluster'])

stable_top5 = top5_base & top5_7030 & top5_3070
print()
print(f'Communities in top 5 under ALL three weightings ({len(stable_top5)}):')
for c in stable_top5:
    print(f'  {c}')

# ── Full ranked list under all three scenarios ────────────────────────────────
r70 = ranked['70_30'].reset_index().rename(
    columns={'Rank': 'rank_70_30', 'score_70_30': 'score_70_30'})
r50 = ranked['50_50'].reset_index().rename(
    columns={'Rank': 'rank_50_50', 'score_50_50': 'score_50_50'})
r30 = ranked['30_70'].reset_index().rename(
    columns={'Rank': 'rank_30_70', 'score_30_70': 'score_30_70'})

full_sensitivity = (
    r70[['Cluster', 'rank_70_30', 'score_70_30']]
    .merge(r50[['Cluster', 'rank_50_50', 'score_50_50']], on='Cluster', how='outer')
    .merge(r30[['Cluster', 'rank_30_70', 'score_30_70']], on='Cluster', how='outer')
    .sort_values('rank_50_50')
    .reset_index(drop=True)
)

full_sensitivity.to_csv(f'{OUTPUT_DIR}/sensitivity_analysis.csv', index=False)
print(f'\nFull sensitivity table saved: {OUTPUT_DIR}/sensitivity_analysis.csv')

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 14 — INTERACTIVE MAP (HTML)
# ══════════════════════════════════════════════════════════════════════════════

colormap_f = cm.LinearColormap(
    colors=['#1a9850', '#ffffbf', '#d73027'],
    vmin=scored['vulnerability_score'].min(),
    vmax=scored['vulnerability_score'].max(),
    caption='Vulnerability Score (red = most vulnerable)'
)

m = folium.Map(location=[44.8, -63.2], zoom_start=7, tiles='CartoDB positron')
colormap_f.add_to(m)

cluster_layer = folium.FeatureGroup(name='Community clusters', show=True)
for _, row in scored.iterrows():
    color  = colormap_f(row['vulnerability_score'])
    radius = max(6, row['pct_65_plus'] * 0.55)
    tip = (
        f"<b>{row['Cluster']}</b><br>"
        f"Vulnerability rank : #{int(row.name)}<br>"
        f"Score              : {row['vulnerability_score']:.3f}<br>"
        f"Distance           : {row['Distance_km']:.1f} km<br>"
        f"Nearest hospital   : {row['Nearest_Hospital']}<br>"
        f"65+ share          : {row['pct_65_plus']:.1f}%<br>"
        f"65+ population     : {int(row['Population_65_plus']):,}<br>"
        f"Population method  : {row['Pop_source']}"
    )
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=radius, color='white', weight=0.8,
        fill=True, fill_color=color, fill_opacity=0.82,
        tooltip=folium.Tooltip(tip, sticky=True)
    ).add_to(cluster_layer)
cluster_layer.add_to(m)

hosp_layer = folium.FeatureGroup(name='Acute hospitals', show=True)
for _, h in hospitals_clean.iterrows():
    folium.Marker(
        location=[h['latitude'], h['longitude']],
        icon=folium.Icon(color='darkblue', icon='plus-sign'),
        tooltip=f"<b>{h['Hospital']}</b><br>{h['Town']}<br>{h['Type']}"
    ).add_to(hosp_layer)
hosp_layer.add_to(m)

folium.LayerControl(collapsed=False).add_to(m)
m.save(f'{OUTPUT_DIR}/vulnerability_map.html')
print(f'Saved: {OUTPUT_DIR}/vulnerability_map.html')


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 15 — SUMMARY TABLE FOR REPORT
# ══════════════════════════════════════════════════════════════════════════════

summary = scored.copy()
summary.index.name = 'Rank'
vulnerable_names = set(vulnerable['Cluster'])
summary['Vulnerable'] = summary['Cluster'].isin(vulnerable_names).map({True: 'Yes', False: ''})

print('=== FULL COMMUNITY RANKING — ALL 54 CLUSTERS ===')
print(f'{"Rk":<4} {"Community":<42} {"Dist":>6} {"65+%":>6} {"Score":>7} {"Method":<22} {"V?"}')
print('-' * 90)
for rank, row in summary.iterrows():
    flag = '◄' if row['Vulnerable'] == 'Yes' else ''
    print(f'{rank:<4} {row["Cluster"]:<42} {row["Distance_km"]:>5.1f} '
          f'{row["pct_65_plus"]:>5.1f}% {row["vulnerability_score"]:>7.3f}'
          f'  {row["Pop_source"]:<22} {flag}')

print()
print(f'◄ = above-median distance (>{dist_median:.1f} km) and 65+ share (>{share_median:.1f}%)')
print(f'Vulnerable communities: {len(vulnerable)} of {len(final_clean)}')

report_cols = {
    'Cluster':             'Community',
    'Distance_km':         'Distance to Hospital (km)',
    'Nearest_Hospital':    'Nearest Acute Hospital',
    'pct_65_plus':         '65+ Share (%)',
    'Population_65_plus':  '65+ Population',
    'Total_pop_est':       'Total Population (estimate)',
    'vulnerability_score': 'Vulnerability Score',
    'Pop_source':          'Population Method',
    'Vulnerable':          'Vulnerable'
}
report_table = summary.rename(columns=report_cols)[list(report_cols.values())]
report_table.to_csv(f'{OUTPUT_DIR}/report_table.csv', index_label='Rank')
print(f'\nSaved: {OUTPUT_DIR}/report_table.csv')
print()
print('=== VULNERABLE COMMUNITIES ONLY ===')
vuln_only = report_table[report_table['Vulnerable'] == 'Yes'].drop(columns=['Vulnerable'])
print(vuln_only.to_string())

print('\n=== PIPELINE COMPLETE ===')
print(f'All outputs written to: {OUTPUT_DIR}/')
