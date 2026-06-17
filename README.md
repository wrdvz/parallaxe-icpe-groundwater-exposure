# Parallaxe ICPE groundwater exposure

This repository turns long-term groundwater trend signals into an operational
exposure layer for French industrial and regulated sites.

It sits between:

- `parallaxe-groundwater-france-trends`: hydrological signal production
- `parallaxe-groundwater-risk-engine`: future search-first product layer

## What this repository produces

This compute repository now has two output families:

### 1. Editorial / analytical outputs

- interactive HTML screening map in `docs/`
- editorial PNG maps by sector
- intermediate processed CSV / GeoJSON files

### 2. Product-facing data exports

For the future app, this repository also exports parquet tables under
`outputs/product/`:

- `companies.parquet`
- `sites.parquet`
- `site_hydro_context.parquet`
- `site_risk_scores.parquet`

These files form the MVP data contract for the future
`parallaxe-groundwater-risk-engine` repository.

## Analytical idea

The current methodology combines:

- a 20 km screening grid
- groundwater trend over 20 years
- territorial pressure based on `AEP + IND + IRR`
- ICPE site enrichment
- a 7-category water-relevance taxonomy

The signal should be read as a contextual operational exposure indicator, not as
an exhaustive financial risk model.

## Current role in the product stack

`parallaxe-icpe-groundwater-exposure` is the **compute and methodology repo**.
It is responsible for:

- ingesting and cleaning source datasets
- building the ICPE and groundwater context tables
- generating cartographic outputs
- exporting product tables for downstream app use

The future `parallaxe-groundwater-risk-engine` app will be responsible for:

- search-first company and site resolution
- portfolio screening UX
- ranking and explanation delivery
- product runtime APIs

## Main sources

- ADES / Hubeau groundwater observations
- BDLISA hydrogeological entities
- BNPE groundwater withdrawals
- ICPE / Georisques facility data
- SIRENE / SIRET enrichment where relevant for product serving

## Product export regeneration

Generate the current MVP parquet exports with:

```bash
python3 scripts/11_export_product_tables.py
```

## Author

Edward Vizard  
Parallaxe processing
