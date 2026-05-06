"""Construit un index léger d'existence SIRENE par SIRET.

But:
- distinguer "SIRET reconnu mais non cartographiable" de "SIRET inconnu"
- stocker un payload minimal dans R2

Sortie:
- shards JSON contenant une simple liste de SIRET par préfixe
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _read_siret_only(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path, columns=["siret"])
    return pd.read_csv(path, usecols=["siret"], dtype={"siret": "string"}, low_memory=False)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "siret": df["siret"].astype("string").str.replace(r"\D+", "", regex=True),
        }
    )
    out = out[out["siret"].str.len() == 14].copy()
    out = out.drop_duplicates(subset=["siret"], keep="first").copy()
    return out


def _build_shards(df: pd.DataFrame, out_dir: Path, prefix_len: int, namespace: str) -> None:
    base_dir = out_dir / namespace
    base_dir.mkdir(parents=True, exist_ok=True)

    data = df.copy()
    data["shard_prefix"] = data["siret"].str.slice(0, prefix_len).str.pad(width=prefix_len, side="right", fillchar="0")

    for prefix, chunk in data.groupby("shard_prefix", dropna=True):
        payload = sorted(chunk["siret"].astype(str).tolist())
        target = base_dir / f"{prefix}.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sirene", required=True, help="Chemin vers le fichier SIRENE source (CSV ou Parquet)")
    parser.add_argument("--out-dir", required=True, help="Dossier de sortie des shards")
    parser.add_argument("--prefix-len", type=int, default=3, help="Longueur du préfixe SIRET pour le sharding")
    parser.add_argument("--namespace", default="sirene-exists/v1", help="Sous-dossier de versionnement des shards")
    args = parser.parse_args()

    sirene = _read_siret_only(Path(args.sirene))
    normalized = _normalize(sirene)

    out_dir = Path(args.out_dir)
    _build_shards(normalized, out_dir, args.prefix_len, args.namespace)

    print("SIRET normalisés:", len(normalized))
    print("Shards output:", out_dir / args.namespace)


if __name__ == "__main__":
    main()
