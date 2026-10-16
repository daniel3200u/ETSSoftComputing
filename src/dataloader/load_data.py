
from pathlib import Path
import ast

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "Order Picking Dataset from a Warehouse of a Footwear Manufacturing Company"


def load_positions(file_path=None):
    """
    Load storage locations and parse their coordinates.

    Expected columns:
        Location, Points

    Example:
        H-06-13, "534, 688, 1"
    """
    file_path = Path(file_path) if file_path else DATA_DIR / "Storage_Location.csv"
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()
    required_columns = {"originalLocation", "position"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in Positions file: {sorted(missing)}"
        )

    df = df.copy()
    df["originalLocation"] = df["originalLocation"].astype("string").str.strip()
    coordinates = df["position"].astype("string").str.split(
        ",", expand=True
    )
    if coordinates.shape[1] != 3:
        raise ValueError(
            "Column 'Points' must contain exactly 3 coordinates: X, Y, Z."
        )
    coordinates.columns = ["X", "Y", "Z"]
    for col in ["X", "Y", "Z"]:
        df[col] = pd.to_numeric(
            coordinates[col].str.strip(),
            errors="coerce"
        )

    _validate_coordinates(df, "Storage_Location")

    return df


def load_support_points(file_path=None):
    """
    Load navigation support points and parse coordinates.

    Expected columns:
        points_specified, labels

    Example:
        (66.0, -29.0, 1.0), LC-01
    """
    file_path = (
        Path(file_path)
        if file_path
        else DATA_DIR / "Support_Points_Navigation.csv"
    )

    df = pd.read_csv(file_path, sep=';')
    df.columns = df.columns.str.strip()
    required_columns = {"points_specified", "labels"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in SupportPoints file: {sorted(missing)}"
        )

    df = df.copy()
    df["labels"] = df["labels"].astype("string").str.strip()

    def parse_point(value):
        if isinstance(value, str):
            value = ast.literal_eval(value)

        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError(
                f"Invalid support point coordinate: {value!r}"
            )

        return pd.Series(value, index=["X", "Y", "Z"])

    parsed = df["points_specified"].apply(parse_point)

    for col in ["X", "Y", "Z"]:
        df[col] = pd.to_numeric(parsed[col], errors="coerce")

    _validate_coordinates(df, "SupportPoints")

    return df


def load_picking_wave(
    file_path,
    wave_id,
    wave_column="Wave",
    location_column="Location"
):
    """
    Load locations belonging to a specific picking wave.

    The Excel file must contain a wave ID column and a location column.
    Change the column names above to match your actual file.
    """
    df = pd.read_excel(file_path)
    df.columns = df.columns.str.strip()

    required_columns = {wave_column, location_column}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing columns in PickingWaves file: {sorted(missing)}"
        )
    df = df.copy()
    df[wave_column] = df[wave_column].astype("string").str.strip()
    df[location_column] = df[location_column].astype("string").str.strip()
    selected = df[df[wave_column] == str(wave_id)].copy()
    if selected.empty:
        raise ValueError(f"Wave {wave_id!r} was not found.")
    selected = selected.drop_duplicates(subset=[location_column])

    return selected.reset_index(drop=True)


def _validate_coordinates(df, dataset_name):
    """Check missing coordinates and duplicate identifiers."""

    coordinate_columns = ["X", "Y", "Z"]

    invalid_coordinates = df[coordinate_columns].isna().any(axis=1)
    if invalid_coordinates.any():
        bad_rows = df.loc[invalid_coordinates]
        raise ValueError(
            f"{dataset_name} contains invalid coordinates:\n"
            f"{bad_rows.to_string(index=False)}"
        )
    if "originalLocation" in df.columns and df["originalLocation"].duplicated().any():
        duplicates = df.loc[
            df["originalLocation"].duplicated(keep=False), "originalLocation"
        ].tolist()

        raise ValueError(
            f"Duplicate storage locations found: {duplicates}"
        )
    if "labels" in df.columns and df["labels"].duplicated().any():
        duplicates = df.loc[
            df["labels"].duplicated(keep=False), "labels"
        ].tolist()
        raise ValueError(
            f"Duplicate support point labels found: {duplicates}"
        )


def load_all_data():
    positions = load_positions()
    support_points = load_support_points()
    return {
        "positions": positions,
        "support_points": support_points,
    }
