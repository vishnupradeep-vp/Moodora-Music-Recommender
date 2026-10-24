"""
preprocess.py
-------------
Loading + cleaning helpers shared by the content-based and
collaborative-filtering recommenders.
"""

import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS = [
    "danceability", "energy", "valence",
    "acousticness", "tempo", "loudness",
]


def _ensure_data_exists(path: str) -> None:
    filepath = Path(path)
    if not filepath.exists():
        # Lazy import to avoid circular dependency
        from data.generate_data import generate_ratings, generate_songs
        filepath.parent.mkdir(parents=True, exist_ok=True)
        songs_df = generate_songs()
        ratings_df = generate_ratings(songs_df)
        songs_df.to_csv(filepath.parent / "songs.csv", index=False)
        ratings_df.to_csv(filepath.parent / "ratings.csv", index=False)


def load_songs(path: str = "data/songs.csv") -> pd.DataFrame:
    _ensure_data_exists(path)
    df = pd.read_csv(path, encoding="utf-8")
    df = df.drop_duplicates(subset="song_id").reset_index(drop=True)
    
    # Ensure all feature columns exist and missing numeric values are filled safely
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mean_val = df[col].mean()
        df[col] = df[col].fillna(0.0 if pd.isna(mean_val) else mean_val)
        
    return df


def load_ratings(path: str = "data/ratings.csv") -> pd.DataFrame:
    _ensure_data_exists(path)
    df = pd.read_csv(path, encoding="utf-8")
    if not all(col in df.columns for col in ["user_id", "song_id", "rating"]):
        return pd.DataFrame(columns=["user_id", "song_id", "rating"])
    return df.dropna(subset=["user_id", "song_id", "rating"]).reset_index(drop=True)


def scale_features(df: pd.DataFrame, columns=FEATURE_COLUMNS) -> pd.DataFrame:
    """Min-max scale audio features to [0, 1] so no single feature
    (e.g. tempo, which ranges 60-200) dominates the similarity score."""
    scaled = df.copy()
    if scaled.empty:
        return scaled
    cols_to_scale = [col for col in columns if col in scaled.columns]
    if not cols_to_scale:
        return scaled
    scaler = MinMaxScaler()
    scaled[cols_to_scale] = np.clip(scaler.fit_transform(scaled[cols_to_scale]), 0.0, 1.0)
    return scaled


