"""Amenity typing for Overture places from the place *name*, used to correct the
category-based ``functional_group`` from :mod:`sitex.arch.poi_classification`.

Why the name: in Vietnam the Overture category of an individual place is often
wrong (a veterinary clinic tagged ``hospital``, a motorbike dealer tagged
``doctors_office``, an upper-secondary school tagged ``elementary_school``).
Editing the category lookup cannot fix this, because the mistake belongs to the
place, not to the category. Vietnamese facility names follow fixed conventions
("Nha khoa", "Mầm non", "THCS", "Trạm y tế", "UBND" ...), so the name tells us
more reliably which facility types an accessibility analysis is dealing with.

- :func:`classify_amenities` adds ``amenity_type`` and a ``note`` that explains
  why a place was dropped.
- :func:`write_review_file` and :func:`apply_review_file` handle the review in
  QGIS. Every place is written once with a ``keep`` flag. Students correct
  ``keep`` or ``amenity_type`` in QGIS, and the next run applies their edits.
  Edits are matched by the Overture ``id``, never by row position.

The rules are defined in the notebook (``03-NA03-Amenity_Isochrones.ipynb``,
Section 2), where students can read and extend them. This module only applies
them. A rule is a tuple ``(amenity_type, group, include, exclude)``, where
``include`` and ``exclude`` are lists of regex fragments matched as whole words.
A place gets the type of the first rule where an ``include`` fragment matches
and no ``exclude`` fragment does.

How a fragment is matched depends on how it is written:

- A fragment in plain ASCII is matched against the folded name (lowercase, no
  diacritics, ``đ`` -> ``d``), so it also finds names typed without accents.
- A fragment with diacritics is matched against the lowercase name with accents
  kept. Use this where folding would merge different words: ``chợ`` (market),
  ``chó`` (dog) and ``cho`` (for); ``sở`` (department) and ``số`` (number);
  ``gỗ`` (wood) and ``go``.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import geopandas as gpd
import pandas as pd

Rule = tuple[str, str, list[str], list[str]]  # (amenity_type, group, include, exclude)


def normalize_name(name) -> str:
    """Lowercase ASCII with diacritics stripped (``đ`` -> ``d``), punctuation to
    spaces — so ``"Trường THCS Nguyễn Du"`` and ``"Truong Thcs Nguyen Du"`` match
    the same rule."""
    if not isinstance(name, str):
        return ""
    text = name.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _lower_name(name) -> str:
    """Lowercase, punctuation to spaces, diacritics kept (NFC)."""
    if not isinstance(name, str):
        return ""
    text = unicodedata.normalize("NFC", name).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text)).strip()


def _matches(folded: str, accented: str, patterns: list[str]) -> bool:
    for p in patterns:
        text = folded if p.isascii() else accented
        if re.search(rf"(?<!\w){p}(?!\w)", text):
            return True
    return False


def classify_amenities(
    pois: gpd.GeoDataFrame,
    rules: list[Rule],
    organisation_keywords: list[str] = (),
    other_places: list[str] = (),
    official_types: list[str] = (),
    name_col: str = "name",
) -> gpd.GeoDataFrame:
    """Add ``amenity_type``, ``amenity_group`` and ``note`` columns from each place's name.

    ``organisation_keywords`` mark organisations named after a facility (a school's
    trade union, a class reunion group): they're left out, not typed.

    ``other_places`` are normalised town names (e.g. ``"chu se"``, ``"kon tum"``):
    a place whose name mentions one is usually a branch listing that Overture has
    geolocated into the study area by mistake, so it's typed but flagged
    ``note="names another town"`` and ``keep=False`` — except for ``official_types``
    (public schools, hospitals, offices ...), which are routinely *named after* a
    place ("THPT Trần Quốc Tuấn, Phú Thiện"), so a town in their name says nothing
    about where they are. A really distant one is still dropped later by its
    distance to the street network.

    ``keep`` is True only for places that got an ``amenity_type`` and no flag —
    the notebook then filters by the amenity types the student selected.
    """
    pois = pois.copy()

    types, groups, notes = [], [], []
    for name in pois[name_col]:
        folded, accented = normalize_name(name), _lower_name(name)
        if not folded:
            types.append(None); groups.append(None); notes.append("no name")
            continue
        if _matches(folded, accented, list(organisation_keywords)):
            types.append(None); groups.append(None); notes.append("organisation, not a facility")
            continue
        hit = next(
            ((t, g) for t, g, inc, exc in rules
             if _matches(folded, accented, inc) and not _matches(folded, accented, exc)),
            None,
        )
        if hit is None:
            types.append(None); groups.append(None); notes.append("no rule matched")
            continue
        types.append(hit[0]); groups.append(hit[1])
        named_elsewhere = (
            hit[0] not in official_types and other_places and _matches(folded, accented, list(other_places))
        )
        notes.append("names another town" if named_elsewhere else "")

    pois["amenity_type"] = types
    pois["amenity_group"] = groups
    pois["note"] = notes
    pois["keep"] = pois["amenity_type"].notna() & (pois["note"] == "")
    return pois


# ── QGIS review round-trip ───────────────────────────────────────────────────

REVIEW_COLUMNS = ["id", "name", "overture_category", "functional_group", "amenity_type", "amenity_group", "keep", "note"]


def write_review_file(pois: gpd.GeoDataFrame, path: Path, local_epsg: int) -> Path:
    """Write every place (not only the kept ones) for review in QGIS, in the local
    metric CRS. ``keep`` is written as 0/1 so it edits as a plain integer field.

    ``auto_type``/``auto_keep`` record the automatic result next to the editable
    columns, so :func:`apply_review_file` can tell a student's edit from an
    untouched row — untouched rows then follow any later improvement to the rules
    instead of freezing today's automatic result."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in REVIEW_COLUMNS if c in pois.columns] + ["geometry"]
    out = pois[cols].copy()
    out["keep"] = out["keep"].astype(int)
    out["auto_type"] = out["amenity_type"]
    out["auto_keep"] = out["keep"]
    out.to_crs(epsg=local_epsg).to_file(path, layer="pois_review", driver="GPKG")
    return path


def apply_review_file(
    pois: gpd.GeoDataFrame, path: Path, rules: list[Rule],
) -> tuple[gpd.GeoDataFrame, int]:
    """Apply the rows a student edited in a review file (``keep`` or ``amenity_type``
    differs from ``auto_keep``/``auto_type``), matched on ``id``. Every other place
    keeps its fresh automatic values. Returns ``(pois, n_edits_applied)``."""
    rev = pd.DataFrame(gpd.read_file(path, layer="pois_review", ignore_geometry=True)).drop_duplicates("id")
    for col in ("amenity_type", "auto_type"):
        rev[col] = rev[col].replace("", None)
    edited = (rev["keep"].astype(int) != rev["auto_keep"].astype(int)) | (
        rev["amenity_type"].fillna("") != rev["auto_type"].fillna("")
    )
    edits = rev.loc[edited, ["id", "amenity_type", "keep"]].set_index("id")

    pois = pois.copy()
    hit = pois["id"].isin(edits.index)
    ids = pois.loc[hit, "id"]
    pois.loc[hit, "keep"] = edits.loc[ids, "keep"].astype(int).astype(bool).values
    pois.loc[hit, "amenity_type"] = edits.loc[ids, "amenity_type"].values
    group_of = {t: g for t, g, _inc, _exc in rules}
    pois.loc[hit, "amenity_group"] = pois.loc[hit, "amenity_type"].map(group_of).fillna("Other").values
    pois.loc[hit, "note"] = "edited in review"
    pois["keep"] = pois["keep"].astype(bool)
    return pois, int(hit.sum())


def export_amenities(pois: gpd.GeoDataFrame, path: Path, local_epsg: int) -> Path:
    """The cleaned result (every place, with ``amenity_type``, ``amenity_group``,
    ``keep`` as 0/1 and ``note``), rewritten on every run — the input for later
    notebooks, unlike the review file, which is written once and then edited."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in REVIEW_COLUMNS if c in pois.columns] + ["geometry"]
    out = pois[cols].copy()
    out["keep"] = out["keep"].astype(int)
    if path.exists():
        path.unlink()
    out.to_crs(epsg=local_epsg).to_file(path, layer="pois_amenities", driver="GPKG")
    return path


def select_amenities(pois: gpd.GeoDataFrame, amenities: list[str]) -> tuple[gpd.GeoDataFrame, list[str]]:
    """Kept places for each chosen entry, which may be an amenity *group*
    ("Healthcare") or a single amenity *type* ("Kindergarten"). The entry goes into
    an ``amenity`` column — one isochrone set is drawn per entry. A place chosen
    twice (as "Healthcare" and as "Hospital") appears once for each.
    Returns ``(facilities, unknown_entries)``."""
    parts, unknown = [], []
    for entry in amenities:
        hit = pois["keep"] & ((pois["amenity_group"] == entry) | (pois["amenity_type"] == entry))
        if not hit.any():
            unknown.append(entry)
        parts.append(pois[hit].assign(amenity=entry))
    facilities = pd.concat(parts, ignore_index=True) if parts else pois.iloc[0:0].assign(amenity=None)
    return gpd.GeoDataFrame(facilities, geometry="geometry", crs=pois.crs), unknown


def amenity_summary(pois: gpd.GeoDataFrame) -> pd.DataFrame:
    """Kept/dropped counts per amenity type — the table students choose from."""
    typed = pois[pois["amenity_type"].notna()].fillna({"amenity_group": "Other"})
    return (
        typed.groupby(["amenity_group", "amenity_type"])["keep"]
        .agg(kept="sum", total="count")
        .astype(int)
        .reset_index()
        .sort_values(["amenity_group", "kept"], ascending=[True, False])
        .reset_index(drop=True)
    )
