"""Remove manifest-listed measurements before interpreting any raw outcomes."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from release_inputs import validate_precleaned_frame


DEFAULT_EXCLUSIONS = Path(__file__).resolve().parents[1] / "data" / "metadata" / "EXCLUDE_wells.csv"


def _validate_exclusions(exclusions: pd.DataFrame) -> pd.DataFrame:
    required = {"name0", "rxn", "plate"}
    if not required.issubset(exclusions.columns):
        raise ValueError("Exclusion manifest requires name0, rxn, and plate columns")
    if exclusions.empty:
        raise ValueError("Exclusion manifest must not be empty")
    for column in required:
        values = exclusions[column].astype("string")
        if (values.isna() | values.str.strip().eq("")).any():
            raise ValueError(f"Exclusion manifest has blank {column} values")
    if exclusions.name0.duplicated().any():
        raise ValueError("Exclusion manifest has duplicate name0 values")
    return exclusions


def load_exclusions(path: Path | None = None) -> pd.DataFrame:
    """Read the project's local, unmodified exclusion CSV."""
    return _validate_exclusions(pd.read_csv(
        DEFAULT_EXCLUSIONS if path is None else path, dtype="string", low_memory=False))


def _require_columns(frame: pd.DataFrame, reaction_column: str) -> None:
    if not {"name0", reaction_column}.issubset(frame.columns):
        raise ValueError(f"Cleanup requires name0 and {reaction_column} columns")


def apply_cleanup(frame: pd.DataFrame, *, reaction_column: str,
                  exclusions: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict]:
    """Delete exact measurement IDs from a full raw corpus and report counts."""
    exclusions = load_exclusions() if exclusions is None else _validate_exclusions(exclusions)
    _require_columns(frame, reaction_column)
    if validate_precleaned_frame(frame):
        assert_clean(frame, reaction_column=reaction_column, exclusions=exclusions)
        return frame.copy(), {'raw_records': len(frame), 'excluded_records': 0,
                              'cleaned_records': len(frame), 'excluded_plate_ids': [],
                              'by_plate': [], 'input_already_cleaned': True}
    matched = frame.name0.isin(exclusions.name0)
    removed = frame.loc[matched]
    if removed.empty:
        raise ValueError("Exclusion manifest matched zero records")
    if not exclusions.name0.isin(removed.name0).all():
        raise ValueError("Exclusion manifest contains unmatched name0 values")
    if removed.name0.duplicated().any():
        raise ValueError("An exclusion name0 matched multiple records")
    expected = removed.name0.map(exclusions.set_index("name0").rxn).astype("string")
    if removed[reaction_column].astype("string").ne(expected).fillna(True).any():
        raise ValueError("Excluded records have reaction labels that disagree with the manifest")
    cleaned = frame.loc[~matched].copy()
    by_plate = [{"plate_id": str(plate), "records": int(records)}
                for plate, records in exclusions.groupby("plate", sort=True).size().items()]
    report = {
        "raw_records": len(frame), "excluded_records": len(removed),
        "cleaned_records": len(cleaned),
        "excluded_plate_ids": [row["plate_id"] for row in by_plate],
        "by_plate": by_plate,
    }
    assert_clean(cleaned, reaction_column=reaction_column, exclusions=exclusions)
    return cleaned, report


def assert_clean(frame: pd.DataFrame, *, reaction_column: str,
                 exclusions: pd.DataFrame | None = None) -> None:
    """Reject excluded measurements in any cleaned analysis or release subset."""
    exclusions = load_exclusions() if exclusions is None else _validate_exclusions(exclusions)
    _require_columns(frame, reaction_column)
    if frame.name0.isin(exclusions.name0).any():
        raise ValueError("Cleaned data contains excluded name0 values")
