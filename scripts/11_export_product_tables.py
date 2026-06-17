from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ICPE_TAXONOMY_FILE = (
    PROJECT_ROOT / "data" / "processed" / "icpe" / "icpe_sites_normalized_with_water_taxonomy_7.csv"
)
ICPE_CONTEXT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "icpe" / "icpe_groundwater_context_2005_2025_20km.csv"
)
ICPE_GRID_FILE = (
    PROJECT_ROOT / "data" / "processed" / "grid" / "icpe_exposure_grid_20km.csv"
)
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "product"
GRID_SIZE_METERS = 20_000


GRID_CLASS_LABELS = {
    "high_pressure_declining_groundwater": "Forte pression + nappe en baisse",
    "low_pressure_declining_groundwater": "Faible pression + nappe en baisse",
    "high_pressure_non_declining_groundwater": "Forte pression + nappe non baissière",
    "low_pressure_non_declining_groundwater": "Faible pression + nappe non baissière",
    "unclassified_no_groundwater_data": "Non classé (pas de donnée nappe)",
}


def _critical_percentile(values: pd.Series, *, lower_is_worse: bool = False) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="Float64")
    valid = numeric.dropna()
    if valid.empty:
        return out
    ranking_basis = -valid if lower_is_worse else valid
    out.loc[valid.index] = (ranking_basis.rank(method="average", pct=True) * 100).astype(float)
    return out


def _decile_from_percentile(percentiles: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(percentiles, errors="coerce")
    out = pd.Series(pd.NA, index=percentiles.index, dtype="Int64")
    valid = numeric.dropna().clip(lower=0, upper=100)
    deciles = np.ceil((100 - valid) / 10).clip(1, 10).astype(int)
    out.loc[valid.index] = deciles
    return out


def _physical_score_10(values: pd.Series, *, lower_is_worse: bool = False) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="Float64")
    valid = numeric.dropna().astype(float)
    if valid.empty:
        return out

    if lower_is_worse:
        severity = (-valid).clip(lower=0)
    else:
        severity = valid.clip(lower=0)
    min_value = float(severity.min())
    max_value = float(severity.max())
    if np.isclose(max_value, min_value):
        out.loc[valid.index] = 10.0
        return out

    normalized = 1 + 9 * (severity - min_value) / (max_value - min_value)
    out.loc[valid.index] = normalized.clip(1, 10)
    return out


def _level_10_from_score(scores: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(scores, errors="coerce")
    out = pd.Series(pd.NA, index=scores.index, dtype="Int64")
    valid = numeric.dropna().clip(lower=1, upper=10)
    levels = np.rint(valid).clip(1, 10).astype(int)
    out.loc[valid.index] = levels
    return out


def _clean_siret(value) -> str | None:
    if pd.isna(value):
        return None
    try:
        return f"{int(float(value)):014d}"
    except (TypeError, ValueError):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        return digits.zfill(14) if len(digits) <= 14 and digits else None


def _clean_siren(value) -> str | None:
    siret = _clean_siret(value)
    return siret[:9] if siret else None


def _to_bool(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "oui"}


def _priority_level(grid_class: str | None, dependency_score: float | None) -> str:
    if grid_class == "Forte pression + nappe en baisse":
        if dependency_score is not None and dependency_score >= 7:
            return "Critical"
        return "High"
    if grid_class in {"Forte pression + nappe non baissière", "Faible pression + nappe en baisse"}:
        if dependency_score is not None and dependency_score >= 7:
            return "High"
        return "Moderate"
    if dependency_score is not None and dependency_score >= 8:
        return "Moderate"
    return "Low"


def _dependency_probability(score: float | None, is_water_relevant: bool) -> str:
    if score is None:
        return "Probable" if is_water_relevant else "Unknown"
    if score >= 8:
        return "High"
    if score >= 5:
        return "Probable"
    if score >= 3:
        return "Possible"
    return "Low"


def _dependency_probability_fr(label: str) -> str:
    return {
        "High": "élevée",
        "Probable": "probable",
        "Possible": "possible",
        "Low": "faible",
        "Unknown": "inconnue",
    }.get(label, label.lower())


def _confidence_label(signal_solid: bool, in_scope: bool, is_icpe: bool) -> str:
    if signal_solid and in_scope and is_icpe:
        return "High"
    if is_icpe and (signal_solid or in_scope):
        return "Medium"
    if is_icpe:
        return "Low"
    return "Unknown"


def _risk_explanation(
    grid_class: str | None, dependency_probability: str, signal_solid: bool, city: str | None
) -> str:
    location = f" à {city}" if city else ""
    robustness = "signal nappe robuste" if signal_solid else "signal fondé sur peu de stations locales"
    if not grid_class or grid_class == "Non classé (pas de donnée nappe)":
        return f"Site{location} sans classification nappe robuste ; dépendance à l'eau {_dependency_probability_fr(dependency_probability)}."
    return (
        f"Site{location} classé « {grid_class.lower()} », avec dépendance à l'eau "
        f"{_dependency_probability_fr(dependency_probability)} et {robustness}."
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    taxonomy = pd.read_csv(ICPE_TAXONOMY_FILE, low_memory=False)
    context = pd.read_csv(ICPE_CONTEXT_FILE, low_memory=False)
    grid = pd.read_csv(ICPE_GRID_FILE, low_memory=False)

    taxonomy_subset = taxonomy[
        ["code_aiot", "categorie_eau_7", "score_probabilite_nappe_1_10", "dans_perimetre_eau"]
    ].copy()
    df = context.merge(taxonomy_subset, on="code_aiot", how="left", suffixes=("", "_taxonomy"))
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["grid_xmin"] = (df["x"] // GRID_SIZE_METERS) * GRID_SIZE_METERS
    df["grid_ymin"] = (df["y"] // GRID_SIZE_METERS) * GRID_SIZE_METERS

    grid_subset = grid[
        [
            "grid_xmin",
            "grid_ymin",
            "withdrawal_volume_m3",
        ]
    ].copy()
    grid_subset["grid_xmin"] = pd.to_numeric(grid_subset["grid_xmin"], errors="coerce")
    grid_subset["grid_ymin"] = pd.to_numeric(grid_subset["grid_ymin"], errors="coerce")
    grid_subset["withdrawal_pressure_volume_m3"] = pd.to_numeric(
        grid_subset["withdrawal_volume_m3"], errors="coerce"
    )
    grid_subset = grid_subset.drop(columns=["withdrawal_volume_m3"])

    df = df.merge(grid_subset, on=["grid_xmin", "grid_ymin"], how="left")

    df["siret"] = df["num_siret"].apply(_clean_siret)
    df["siren"] = df["num_siret"].apply(_clean_siren)
    df["site_id"] = df["code_aiot"].apply(lambda v: f"icpe_{int(v)}")
    df["is_icpe"] = df["lib_regime"].fillna("").ne("Non ICPE")
    df["is_geolocated"] = df["x"].notna() & df["y"].notna()
    df["icpe_category"] = df["categorie_eau_7"].where(df["categorie_eau_7"].notna(), None)
    df["grid_class"] = df["local_signal_marker"].map(
        {
            "decline_high_pressure": "Forte pression + nappe en baisse",
            "decline_limited": "Faible pression + nappe en baisse",
            "rise_high_pressure": "Forte pression + nappe non baissière",
            "rise_limited": "Faible pression + nappe non baissière",
        }
    )
    # Prefer the current map/grid nomenclature whenever present.
    df["grid_class"] = df["grid_class"].where(df["grid_class"].notna(), None)
    if "withdrawal_pressure_volume_m3" not in df.columns:
        df["withdrawal_pressure_volume_m3"] = pd.NA

    transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)
    lon_lat = df.apply(
        lambda row: transformer.transform(row["x"], row["y"]) if row["is_geolocated"] else (None, None),
        axis=1,
        result_type="expand",
    )
    df["lon"] = lon_lat[0]
    df["lat"] = lon_lat[1]

    companies = (
        df[df["siren"].notna()][["siren", "nom_ets", "code_naf", "lib_naf"]]
        .rename(columns={"nom_ets": "company_name", "code_naf": "naf_code", "lib_naf": "naf_label"})
        .query("siren != '000000000'")
        .assign(normalized_name=lambda d: d["company_name"].fillna("").str.upper().str.strip())
        .sort_values(["siren", "company_name"])
        .drop_duplicates(subset=["siren"], keep="first")
        .reset_index(drop=True)
    )

    sites = (
        df[
            [
                "site_id",
                "siret",
                "siren",
                "nom_ets",
                "adresse",
                "cd_postal",
                "commune",
                "lat",
                "lon",
                "is_geolocated",
                "is_icpe",
                "icpe_category",
                "code_naf",
                "lib_naf",
                "url_fiche",
            ]
        ]
        .rename(
            columns={
                "nom_ets": "site_name",
                "adresse": "address_line",
                "cd_postal": "postal_code",
                "commune": "city",
                "code_naf": "naf_code",
                "lib_naf": "naf_label",
                "url_fiche": "source_url",
            }
        )
        .copy()
    )
    sites["company_name"] = sites["site_name"]
    sites["geo_score"] = None
    sites["geo_type"] = None
    sites["geoloc_confidence_label"] = "High"

    site_hydro_context = (
        df[
            [
                "site_id",
                "n_stations_20km",
                "median_variation_20y_cm_20km",
                "mean_variation_20y_cm_20km",
                "min_station_distance_km",
                "is_signal_solid",
                "local_signal_class",
                "local_signal_marker",
                "grid_class",
                "withdrawal_pressure_volume_m3",
            ]
        ]
        .rename(
            columns={
                "n_stations_20km": "station_count",
                "median_variation_20y_cm_20km": "aquifer_trend_value_cm_20y",
                "mean_variation_20y_cm_20km": "aquifer_trend_mean_cm_20y",
                "min_station_distance_km": "nearest_station_distance_km",
                "is_signal_solid": "groundwater_signal_robust",
                "local_signal_class": "aquifer_trend_level",
                "local_signal_marker": "aquifer_signal_marker",
            }
        )
        .copy()
    )
    site_hydro_context["pressure_level"] = site_hydro_context["grid_class"].map(
        {
            "Forte pression + nappe en baisse": "High",
            "Forte pression + nappe non baissière": "High",
            "Faible pression + nappe en baisse": "Low",
            "Faible pression + nappe non baissière": "Low",
        }
    ).fillna("Unknown")
    # Keep relative rank fields for later analysis, but build the V2 product score from
    # physical positions on each value scale rather than from population quantiles.
    site_hydro_context["groundwater_decline_percentile"] = _critical_percentile(
        site_hydro_context["aquifer_trend_value_cm_20y"], lower_is_worse=True
    )
    site_hydro_context["withdrawal_volume_percentile"] = _critical_percentile(
        site_hydro_context["withdrawal_pressure_volume_m3"], lower_is_worse=False
    )
    site_hydro_context["groundwater_decline_decile"] = _decile_from_percentile(
        site_hydro_context["groundwater_decline_percentile"]
    )
    site_hydro_context["withdrawal_volume_decile"] = _decile_from_percentile(
        site_hydro_context["withdrawal_volume_percentile"]
    )

    groundwater_score_10 = _physical_score_10(
        site_hydro_context["aquifer_trend_value_cm_20y"], lower_is_worse=True
    )
    withdrawal_score_10 = _physical_score_10(
        site_hydro_context["withdrawal_pressure_volume_m3"], lower_is_worse=False
    )
    criticality_score_10 = (
        (groundwater_score_10 + withdrawal_score_10) / 2
    ).where(groundwater_score_10.notna() & withdrawal_score_10.notna())
    groundwater_level_10 = _level_10_from_score(groundwater_score_10)
    withdrawal_level_10 = _level_10_from_score(withdrawal_score_10)

    risk = df[
        [
            "site_id",
            "grid_class",
            "score_probabilite_nappe_1_10",
            "is_water_relevant",
            "is_signal_solid",
            "commune",
            "dans_perimetre_eau",
        ]
    ].copy()
    risk["dependency_probability"] = risk.apply(
        lambda row: _dependency_probability(
            None if pd.isna(row["score_probabilite_nappe_1_10"]) else float(row["score_probabilite_nappe_1_10"]),
            _to_bool(row["is_water_relevant"]),
        ),
        axis=1,
    )
    risk["priority_level"] = risk.apply(
        lambda row: _priority_level(
            row["grid_class"],
            None if pd.isna(row["score_probabilite_nappe_1_10"]) else float(row["score_probabilite_nappe_1_10"]),
        ),
        axis=1,
    )
    risk["confidence_label"] = risk.apply(
        lambda row: _confidence_label(
            _to_bool(row["is_signal_solid"]),
            _to_bool(row["dans_perimetre_eau"]),
            True,
        ),
        axis=1,
    )
    risk["risk_explanation_short"] = risk.apply(
        lambda row: _risk_explanation(
            row["grid_class"],
            row["dependency_probability"],
            _to_bool(row["is_signal_solid"]),
            row["commune"],
        ),
        axis=1,
    )
    risk["score_version"] = "mvp_v2_physical_levels"
    risk["dependency_score_1_10"] = pd.to_numeric(risk["score_probabilite_nappe_1_10"], errors="coerce")
    risk["is_water_relevant"] = risk["is_water_relevant"].apply(_to_bool)
    risk["within_water_scope"] = risk["dans_perimetre_eau"].apply(_to_bool)
    risk["groundwater_score_10"] = groundwater_score_10.round(1)
    risk["withdrawal_score_10"] = withdrawal_score_10.round(1)
    risk["criticality_score_10"] = criticality_score_10.round(1)
    site_hydro_context["groundwater_level_10"] = groundwater_level_10
    site_hydro_context["withdrawal_level_10"] = withdrawal_level_10
    site_risk_scores = risk[
        [
            "site_id",
            "priority_level",
            "dependency_probability",
            "confidence_label",
            "risk_explanation_short",
            "score_version",
            "dependency_score_1_10",
            "groundwater_score_10",
            "withdrawal_score_10",
            "criticality_score_10",
            "is_water_relevant",
            "within_water_scope",
        ]
    ].copy()

    companies.to_parquet(OUTPUT_DIR / "companies.parquet", index=False)
    sites.to_parquet(OUTPUT_DIR / "sites.parquet", index=False)
    site_hydro_context.to_parquet(OUTPUT_DIR / "site_hydro_context.parquet", index=False)
    site_risk_scores.to_parquet(OUTPUT_DIR / "site_risk_scores.parquet", index=False)

    print("Product parquet exports written to", OUTPUT_DIR)
    print("companies:", len(companies))
    print("sites:", len(sites))
    print("site_hydro_context:", len(site_hydro_context))
    print("site_risk_scores:", len(site_risk_scores))


if __name__ == "__main__":
    main()
